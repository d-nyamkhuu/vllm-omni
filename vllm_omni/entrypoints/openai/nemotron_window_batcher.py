# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coalesce ready WebSocket windows into one native generation request."""

import asyncio
from collections import defaultdict, deque
from itertools import islice
from typing import TypedDict


class AudioWindow(TypedDict):
    asr_stream: str
    asr_lang: int
    asr_keep: int
    asr_left: int
    asr_pcm: bytes
    asr_final: bool


QueuedWindow = tuple[AudioWindow, asyncio.Future[list[int]]]


class NemotronWindowBatcher:
    def __init__(self, submit, *, packed=True):
        self.packed = packed
        self.ready_hops = 24
        self.submit = submit
        self.limit = 128
        self.pending: deque[QueuedWindow] = deque()
        self.task: asyncio.Task | None = None

    async def window(self, info: AudioWindow) -> list[int]:
        future: asyncio.Future[list[int]] = asyncio.get_running_loop().create_future()
        self.pending.append((info, future))
        if self.task is None:
            self.task = asyncio.create_task(self._drain())
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # The connection's abort must reach the worker after its final outstanding update.
            await asyncio.shield(future)
            raise

    def _take_span_batch(self) -> list[QueuedWindow]:
        # Logical RNNT updates from a stream share the same acoustic span.
        # Limit both the actual encoder work and the per-call transport size.
        queues: defaultdict[str, deque[QueuedWindow]] = defaultdict(deque)
        for item in self.pending:
            queues[item[0]["asr_stream"]].append(item)
        self.pending.clear()
        active = deque(queues.values())
        batch: list[QueuedWindow] = []
        frames = spans = 0
        logical_limit = 4 * self.limit
        while active and len(batch) < logical_limit:
            stream = active.popleft()
            n = min(self.ready_hops, len(stream), logical_limit - len(batch))
            run = list(islice(stream, n))
            first, last = run[0][0], run[-1][0]
            samples = first["asr_keep"] + sum(len(info["asr_pcm"]) // 2 for info, _ in run)
            # An unaligned final chunk may require a separate acoustic window.
            split_final = n > 1 and last["asr_final"]
            if split_final:
                samples += last["asr_keep"]
            cost = (samples + 1279) // 1280 + int(split_final)
            groups = 1 + int(split_final)
            if batch and (spans + groups > 32 or frames + cost > 4096):
                active.appendleft(stream)
                break
            batch.extend(stream.popleft() for _ in range(n))
            frames += cost
            spans += groups
            if stream:
                active.append(stream)
        for stream in active:
            self.pending.extend(stream)
        return batch

    async def _drain(self):
        try:
            # Coalesce simultaneous arrivals. Queues also build naturally while the GPU runs.
            await asyncio.sleep(0.0005)
            while self.pending:
                if self.packed:
                    batch = self._take_span_batch()
                else:
                    batch = [self.pending.popleft() for _ in range(min(self.limit, len(self.pending)))]
                rows = [info for info, _ in batch]
                try:
                    audio = [row["asr_pcm"] for row in rows]
                    payload = {
                        "asr_stream": [r["asr_stream"] for r in rows],
                        "asr_lang": [r["asr_lang"] for r in rows],
                        "asr_left": [r["asr_left"] for r in rows],
                        "asr_final": [r["asr_final"] for r in rows],
                        "asr_keep": [r["asr_keep"] for r in rows],
                        "asr_batch_lengths": [len(blob) // 2 for blob in audio],
                        "pcm": audio,
                    }
                    data = await self.submit(payload)
                    parts = data["asr_parts"]
                    if len(parts) != len(batch):
                        raise RuntimeError("Inconsistent batched RNNT response")
                    for (_, future), part in zip(batch, parts, strict=True):
                        if not future.done():
                            future.set_result(part)
                except Exception as exc:
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(exc)
        finally:
            self.task = None
