# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transfer each incoming PCM sample once and reconstruct a batch of buffered windows on device."""

from collections import defaultdict

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _assemble(suffix, meta, out, width: tl.constexpr, ring: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1) * block + tl.arange(0, block)
    desc = meta + row * 5
    history = tl.load(desc).to(tl.pointer_type(tl.int16))
    first = tl.load(desc + 1)
    length = tl.load(desc + 2)
    split = tl.load(desc + 3)
    offset = tl.load(desc + 4)
    absolute = first + j
    prior = tl.load(history + absolute % ring, (j < length) & (absolute < split), 0)
    recent = tl.load(suffix + offset + absolute - split, (j < length) & (absolute >= split), 0)
    tl.store(out + row * width + j, (prior.to(tl.float32) + recent.to(tl.float32)) * (1.0 / 32768), j < width)


@triton.jit
def _save(suffix, meta, ring: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1) * block + tl.arange(0, block)
    desc = meta + row * 5
    history = tl.load(desc).to(tl.pointer_type(tl.int16))
    start = tl.load(desc + 1)
    length = tl.load(desc + 2)
    offset = tl.load(desc + 4)
    keep = (j < length) & (j >= length - ring)
    pcm = tl.load(suffix + offset + j, keep, 0)
    tl.store(history + (start + j) % ring, pcm, keep)


class PCMWindowRing:
    capacity = 98560

    def __init__(self, device, capacity=98560):
        self.capacity = capacity
        self.device = device
        self.states = {}

    def release(self, stream):
        self.states.pop(stream, None)

    def __call__(self, rows, selected=None, starts=None, trims=None, *, host_lengths=False):
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[row["asr_stream"]].append(i)
        descriptors = [None] * len(rows)
        commits, parts = [], []
        updated = {}
        offset = 0
        for stream, indices in groups.items():
            state = self.states.get(stream)
            if state is None:
                if rows[indices[0]].get("asr_keep", 0):
                    raise ValueError("Missing ASR PCM history for a continuing window")
                state = torch.empty(self.capacity, device=self.device, dtype=torch.int16), 0, 0
            ring, start, previous = state
            ring.record_stream(torch.cuda.current_stream(self.device))
            end = start
            for i in indices:
                row = rows[i]
                pcm = torch.as_tensor(row["asr_pcm"], device="cpu", dtype=torch.int16).reshape(-1)
                keep = int(row.get("asr_keep", 0))
                length = keep + pcm.numel()
                if not 0 <= keep <= previous or length > self.capacity:
                    raise ValueError("Invalid ASR window overlap")
                descriptors[i] = (ring.data_ptr(), end - keep, length, start, offset)
                parts.append(pcm)
                end += pcm.numel()
                previous = length
            commits.append((ring.data_ptr(), start, end - start, 0, offset))
            offset += end - start
            updated[stream] = ring, end, previous
        # The pinned allocator retains the transfer buffers until the nonblocking copy completes.
        cpu = torch.empty(max(1, offset), dtype=torch.int16, device="cpu", pin_memory=True)
        if offset:
            torch.cat(parts, out=cpu[:offset])
        suffix = cpu.to(self.device, non_blocking=True)
        if starts is not None:
            combined = []
            trims = [0] * len(starts) if trims is None else trims
            for first, last, trim in zip(starts, selected, trims, strict=True):
                ptr, latest, length, split, suffix_offset = descriptors[last]
                earliest = descriptors[first][1]
                if rows[first]["asr_stream"] != rows[last]["asr_stream"] or earliest > latest:
                    raise ValueError("Invalid shared PCM window")
                total = length + latest - earliest
                if not 0 <= trim <= total:
                    raise ValueError("Invalid trailing PCM trim")
                combined.append((ptr, earliest, total - trim, split, suffix_offset))
            descriptors = combined
        elif selected is not None:
            descriptors = [descriptors[i] for i in selected]
        meta = torch.tensor(descriptors + commits, dtype=torch.int64, device="cpu", pin_memory=True).to(
            self.device, non_blocking=True
        )
        n = len(descriptors)
        width = max(d[2] for d in descriptors)
        audio = torch.empty(n, width, device=self.device, dtype=torch.float32)
        if width:
            _assemble[(n, triton.cdiv(width, 256))](suffix, meta, audio, width, self.capacity, 256)
        largest = max(d[2] for d in commits)
        if largest:
            _save[(len(commits), triton.cdiv(largest, 256))](suffix, meta[n:], self.capacity, 256)
        for stream, indices in groups.items():
            if rows[indices[-1]]["asr_final"]:
                self.release(stream)
            else:
                self.states[stream] = updated[stream]
        lengths = meta[:n, 2].contiguous()
        if host_lengths:
            return audio, lengths, [d[2] for d in descriptors]
        return audio, lengths
