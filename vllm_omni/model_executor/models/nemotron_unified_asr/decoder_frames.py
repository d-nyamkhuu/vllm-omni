# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pack valid Conformer frames directly into one time sequence per RNNT stream."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather(
    enc,
    lengths,
    meta,
    x,
    lens,
    enc_t,
    length_stride,
    sr,
    sc,
    st,
    channels: tl.constexpr,
    pitch,
    steps,
    hop: tl.constexpr,
    block: tl.constexpr,
):
    b, t = tl.program_id(0), tl.program_id(1)
    desc = meta + b * pitch
    count, final = tl.load(desc), tl.load(desc + 1)
    last = count - 1
    last_row = tl.load(desc + 2 + last * 2)
    last_left = tl.load(desc + 3 + last * 2)
    length = last * hop + tl.where(final != 0, tl.load(lengths + last_row * length_stride) - last_left, hop)
    if t == 0:
        tl.store(lens + b, length)
    step = tl.minimum(t // hop, last)
    row = tl.load(desc + 2 + step * 2)
    left = tl.load(desc + 3 + step * 2)
    frame = tl.minimum(enc_t - 1, left + t - step * hop)
    c = tl.arange(0, block)
    values = tl.load(enc + row * sr + frame * st + c * sc, (t < length) & (c < channels), 0)
    tl.store(x + (b * steps + t) * channels + c, values, c < channels)


class DecoderFramePlan:
    def __init__(self, rows, groups, hop, tail, use_history, device, *, parents=None, offsets=None):
        self.hop = hop
        longest = max(map(len, groups))
        self.steps = max((len(g) - 1) * hop + (tail if rows[g[-1]]["asr_final"] else hop) for g in groups)
        desc = []
        for indices in groups:
            record = [len(indices), int(rows[indices[-1]]["asr_final"])]
            for i in indices:
                row = parents[i] if parents is not None else i
                offset = offsets[i] if offsets is not None else (0 if use_history else rows[i]["asr_left"])
                record.extend((row, offset))
            record.extend([0] * (2 * (longest - len(indices))))
            desc.append(record)
        self.meta = torch.tensor(desc, device="cpu", dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)

    def gather(self, enc, lengths):
        batch, channels = self.meta.shape[0], enc.shape[1]
        x = enc.new_empty((batch, self.steps, channels))
        lens = enc.new_empty((batch,), dtype=torch.int64)
        _gather[(batch, self.steps)](
            enc,
            lengths,
            self.meta,
            x,
            lens,
            enc.shape[2],
            lengths.stride(0),
            *enc.stride(),
            channels,
            self.meta.shape[1],
            self.steps,
            self.hop,
            triton.next_power_of_2(channels),
        )
        return x, lens
