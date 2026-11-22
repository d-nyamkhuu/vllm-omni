# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replay the buffered ASR encoder on the serving worker's CUDA stream."""

from __future__ import annotations

from collections.abc import Callable

import torch


class BufferedEncoderGraphs:
    """An instance belongs to one encoder and one serial inference worker."""

    def __init__(
        self,
        forward: Callable,
        *,
        frame_buckets: tuple[int, ...],
        batch_buckets: tuple[int, ...] | None = None,
        max_graphs: int | None = None,
        reuse_larger: bool = False,
    ):
        self.forward = forward
        self.frame_buckets = tuple(sorted(frame_buckets))
        self.batch_buckets = tuple(sorted(batch_buckets)) if batch_buckets else None
        self.max_graphs = max_graphs
        self.reuse_larger = reuse_larger
        self.observations = {}
        self.pool = None
        self.graphs = {}
        self.captures = 0
        self.replays = 0

    @torch.inference_mode()
    def __call__(self, audio_signal: torch.Tensor, length: torch.Tensor):
        batch, channels, frames = audio_signal.shape
        bucket = next((size for size in self.frame_buckets if size >= frames), None)
        capacity = next((b for b in self.batch_buckets if b >= batch), None) if self.batch_buckets else batch
        if bucket is None or capacity is None:
            return self.forward(audio_signal=audio_signal, length=length)
        key = (capacity, channels, bucket, audio_signal.device, audio_signal.dtype, length.dtype)
        if self.reuse_larger and key not in self.graphs:
            # Reuse an existing private graph when padding costs less than another
            # eager encoder pass. Compare with the requested bucket, and keep the
            # single-stream latency path on single-row graphs.
            candidates = (
                k
                for k in self.graphs
                if k[1] == channels
                and k[3:] == key[3:]
                and k[0] >= batch
                and k[2] >= frames
                and (batch > 1 or k[0] == 1)
                and k[0] * k[2] <= 1.5 * capacity * bucket
            )
            key = min(candidates, key=lambda k: k[0] * k[2], default=key)
        if key not in self.graphs:
            if self.max_graphs is not None:
                if len(self.graphs) >= self.max_graphs:
                    return self.forward(audio_signal=audio_signal, length=length)
                seen = self.observations.get(key, 0) + 1
                self.observations[key] = seen
                if seen < 3:
                    return self.forward(audio_signal=audio_signal, length=length)
            static = audio_signal.new_zeros((capacity, channels, bucket))
            static[:batch, :, :frames].copy_(audio_signal)
            lengths = length.new_full((capacity,), frames)
            lengths[:batch].copy_(length)
            # cuBLAS, convolution plans, and the positional encoding cache initialize eagerly.
            stream = torch.cuda.Stream(device=audio_signal.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.forward(audio_signal=static, length=lengths)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            if self.batch_buckets and self.max_graphs is None and self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            # This worker replays one graph at a time and copies its outputs before reuse.
            with torch.cuda.graph(graph, stream=stream, pool=self.pool):
                encoded, encoded_lengths = self.forward(audio_signal=static, length=lengths)
            self.graphs[key] = (graph, static, lengths, encoded, encoded_lengths)
            self.captures += 1
        graph, static, lengths, encoded, encoded_lengths = self.graphs[key]
        static[:batch, :, :frames].copy_(audio_signal)
        static[..., frames:].zero_()
        lengths[:batch].copy_(length)
        graph.replay()
        self.replays += 1
        # The checkpoint's stride-8 convolution returns ceil(frames/8) positions.
        # Give downstream RNNT independent tensors as the next replay reuses storage.
        return encoded[:batch, :, : (frames + 7) // 8].clone(), encoded_lengths[:batch].clone()
