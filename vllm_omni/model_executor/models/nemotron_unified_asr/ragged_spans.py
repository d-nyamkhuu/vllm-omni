# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Execute a batch of seven-frame-context Conformer spans on a single token matrix."""

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

from .residual_norm import residual_norm
from .span_attention import _banded_attention


@triton.jit
def _depthwise(
    glu,
    meta,
    weight,
    bias,
    out,
    width: tl.constexpr,
    taps: tl.constexpr,
    left: tl.constexpr,
    has_bias: tl.constexpr,
    channels: tl.constexpr,
):
    tile, block = tl.program_id(0), tl.program_id(1)
    origin = tl.load(meta + 3 * tile)
    pos = tl.load(meta + 3 * tile + 1) + tl.arange(0, 16)
    length = tl.load(meta + 3 * tile + 2)
    c = block * channels + tl.arange(0, channels)
    acc = tl.full((16, channels), 0, tl.float32)
    for tap in tl.static_range(taps):
        j = pos + tap - left
        x = tl.load(
            glu + (origin + j[:, None]) * width + c[None, :],
            ((j >= 0) & (j < length))[:, None] & (c < width)[None, :],
            0,
        )
        w = tl.load(weight + c * taps + tap, c < width, 0)
        acc = tl.fma(x.to(tl.float32), w[None, :].to(tl.float32), acc)
    if has_bias:
        acc += tl.load(bias + c, c < width, 0)[None, :].to(tl.float32)
    tl.store(out + (origin + pos[:, None]) * width + c[None, :], acc, (c < width)[None, :])


def pack_layout(samples):
    # The centered STFT has at most floor(samples/160)+1 frames. The checkpoint's
    # three strided convolutions each round up. GPU lengths mask the final tile.
    starts, tiles = [], []
    for row, n in enumerate(samples):
        start = 16 * len(tiles)
        starts.append(start)
        upper = triton.cdiv(n // 160 + 1, 8)
        for pos in range(0, max(1, upper), 16):
            tiles.append((row, start, pos))
    count = next(
        (size for size in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256) if size >= len(tiles)), len(tiles)
    )
    tiles.extend((-1, i * 16, 0) for i in range(len(tiles), count))
    return starts, tiles


class RaggedSpanEncoder:
    def __init__(self, encoder, native):
        self.encoder = encoder
        self.native = native
        self.samples = None
        self.positions = []
        self.graphs = {}

    def _layers(self, x, meta):
        x = x * self.encoder.xscale if self.encoder.xscale else x
        for layer, pos in zip(self.encoder.layers, self.positions, strict=True):
            residual, normed = residual_norm(
                x, layer.feed_forward1(layer.norm_feed_forward1(x)), layer.norm_self_att, layer.fc_factor
            )
            a = layer.self_attn
            # Keep the released checkpoint's three independent projections.
            q, k, v = a.linear_q(normed), a.linear_k(normed), a.linear_v(normed)
            attended = torch.empty_like(q)
            _banded_attention[(1, a.h, len(meta))](
                q, k, v, pos, a.pos_bias_u, a.pos_bias_v, meta, attended, 128, a.h, a.d_k, 16, 128, True, num_warps=4
            )
            residual, normed = residual_norm(residual, a.linear_out(attended), layer.norm_conv, 1.0)
            conv = layer.conv
            glu = F.glu(conv._apply_pointwise(normed, conv.pointwise_conv1), dim=-1)
            depth = torch.empty_like(glu)
            dw = conv.depthwise_conv
            width = glu.shape[1]
            _depthwise[(len(meta), triton.cdiv(width, 128))](
                glu,
                meta,
                dw.weight,
                dw.bias,
                depth,
                width,
                dw.kernel_size[0],
                dw._left_padding,
                dw.bias is not None,
                128,
            )
            y = conv._apply_pointwise(conv.activation(conv.batch_norm(depth)), conv.pointwise_conv2)
            residual, normed = residual_norm(residual, y, layer.norm_feed_forward2, 1.0)
            _, x = residual_norm(residual, layer.feed_forward2(normed), layer.norm_out, layer.fc_factor, False)
        return x

    def _run(self, x, meta):
        count = len(meta)
        # Use private graph pools. Unusually large batches execute eagerly.
        if count > 256:
            return self._layers(x, meta)
        key = count
        if key not in self.graphs:
            static, desc = x.clone(), meta.clone()
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._layers(static, desc)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self._layers(static, desc)
            self.graphs[key] = graph, static, desc, output
        graph, static, desc, output = self.graphs[key]
        static.copy_(x)
        desc.copy_(meta)
        graph.replay()
        return output

    def __call__(self, audio_signal, length):
        if self.samples is None or (len(self.samples) == 1 and self.samples[0] <= 103680):
            return self.native(audio_signal=audio_signal, length=length)
        enc = self.encoder
        pre, lens = enc.pre_encode(x=audio_signal.transpose(1, 2), lengths=length)
        b, t, width = pre.shape
        starts, tiles = pack_layout(self.samples)
        layout = torch.tensor(tiles, dtype=torch.int32, device="cpu", pin_memory=True).to(pre.device, non_blocking=True)
        row = layout[:, 0].long()
        safe = row.clamp_min(0)
        valid = torch.where(row >= 0, lens.index_select(0, safe), 0).int()
        meta = torch.cat((layout[:, 1:], valid[:, None]), 1).contiguous()
        local = layout[:, 2, None] + torch.arange(16, device=pre.device)
        x = pre[safe[:, None], local.clamp_max(t - 1)].reshape(-1, width)
        if not self.positions:
            _, pos = enc.pos_enc(x=pre.new_zeros((1, 128, width)), cache_len=0)
            self.positions = [layer.self_attn.linear_pos(pos).reshape(-1, width).contiguous() for layer in enc.layers]
        packed = self._run(x, meta)
        base = torch.tensor(starts, device="cpu", pin_memory=True).to(pre.device, non_blocking=True)
        j = torch.arange(t, device=pre.device)
        out = packed[(base[:, None] + j).clamp_max(len(packed) - 1)]
        out = out.masked_fill(j[None, :, None] >= lens[:, None, None], 0)
        if enc.out_proj is not None:
            out = enc.out_proj(out)
        return out.transpose(1, 2), lens.to(torch.int64)
