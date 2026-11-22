# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Banded relative-position attention for the released seven-frame streaming context."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _banded_attention(
    q,
    k,
    v,
    pos,
    bias_u,
    bias_v,
    mask,
    out,
    frames: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    queries: tl.constexpr,
    keys: tl.constexpr,
    packed: tl.constexpr = False,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    local_i = tl.arange(0, queries)
    local_j = tl.arange(0, keys)
    if packed:
        origin = tl.load(mask + tile * 3)
        first_query = tl.load(mask + tile * 3 + 1)
        length = tl.load(mask + tile * 3 + 2)
    else:
        origin = b * frames
        first_query = tile * queries
        length = frames
    i = first_query + local_i
    # Earliest possible key for the first query in this tile. The final query's
    # right context also fits inside 128 keys for the released [70, 7, 4] mask.
    start = tl.maximum(0, (first_query // 7) * 7 - 70)
    j = start + local_j
    d = tl.arange(0, dim)
    width: tl.constexpr = heads * dim
    q_offsets = (origin + i[:, None]) * width + h * dim + d[None, :]
    kv_offsets = (origin + j[:, None]) * width + h * dim + d[None, :]
    query = tl.load(q + q_offsets, i[:, None] < length, 0)
    key = tl.load(k + kv_offsets, j[:, None] < length, 0)
    u = tl.load(bias_u + h * dim + d)
    w = tl.load(bias_v + h * dim + d)
    dt: tl.constexpr = q.dtype.element_ty
    qu = (query.to(tl.float32) + u[None, :].to(tl.float32)).to(dt)
    qv = (query.to(tl.float32) + w[None, :].to(tl.float32)).to(dt)
    content = tl.dot(qu, tl.trans(key)).to(dt)
    # Relative vectors form a contiguous diagonal band across the query tile.
    first = frames - 1 - (first_query + queries - 1) + start
    r = first + local_j
    vectors = tl.load(pos + r[:, None] * width + h * dim + d[None, :], ((r >= 0) & (r < 2 * frames - 1))[:, None], 0)
    relative = tl.dot(qv, tl.trans(vectors)).to(dt)
    column = queries - 1 - local_i[:, None] + local_j[None, :]
    relative = tl.gather(relative, tl.minimum(column, keys - 1), axis=1)
    score = (content.to(tl.float32) + relative.to(tl.float32)).to(dt)
    score = (score.to(tl.float32) * (1.0 / tl.sqrt(float(dim)))).to(dt).to(tl.float32)
    if packed:
        chunk = i // 7 * 7
        blocked = (
            (j[None, :] < chunk[:, None] - 70)
            | (j[None, :] > chunk[:, None] + 10)
            | (j[None, :] >= length)
            | (i[:, None] >= length)
        )
    else:
        blocked = tl.load(
            mask + (b * frames + i[:, None]) * frames + j[None, :], (i[:, None] < frames) & (j[None, :] < frames), 1
        )
    score = tl.where(~blocked & (column < keys), score, -float("inf"))
    maximum = tl.max(score, 1)
    weights = tl.exp(score - tl.where(maximum == -float("inf"), 0.0, maximum)[:, None])
    p = (weights / tl.maximum(tl.sum(weights, 1), 1.0)[:, None]).to(dt)
    value = tl.load(v + kv_offsets, j[:, None] < length, 0)
    if packed:
        tl.store(out + q_offsets, tl.dot(p, value))
    else:
        tl.store(out + q_offsets, tl.dot(p, value), i[:, None] < frames)


def banded_attention(q, k, v, pos, u, w, mask):
    batch, frames, heads, dim = q.shape
    result = torch.empty_like(q, memory_format=torch.contiguous_format)
    _banded_attention[(batch, heads, triton.cdiv(frames, 16))](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        pos.contiguous(),
        u.contiguous(),
        w.contiguous(),
        mask.contiguous(),
        result,
        frames,
        heads,
        dim,
        16,
        128,
        num_warps=4,
    )
    return result


class _SpanAttention:
    def __init__(self, module):
        self.module = module
        self.original = module.forward
        self.positions = {}

    def __call__(self, query, key, value, mask, pos_emb, cache=None):
        m = self.module
        b, t, width = query.shape
        if (
            m.training
            or m.use_pytorch_sdpa
            or cache is not None
            or query is not key
            or key is not value
            or mask is None
            or mask.shape != (b, t, t)
            or pos_emb.shape[:2] != (1, 2 * t - 1)
            or m.d_k != 128
            or query.dtype != torch.bfloat16
            or not query.is_cuda
            or torch.is_autocast_enabled()
        ):
            return self.original(query, key, value, mask, pos_emb, cache)
        q, k, v = m.forward_qkv(query, key, value)
        position_key = pos_emb.data_ptr(), t, pos_emb.stride()
        saved = self.positions.get(position_key)
        if saved is None:
            p = m.linear_pos(pos_emb).reshape(2 * t - 1, m.h, m.d_k)
            # Retain frozen NeMo table views and their projections outside graphs.
            if len(self.positions) < 16 and not torch.cuda.is_current_stream_capturing():
                self.positions[position_key] = pos_emb, p
        else:
            p = saved[1]
        result = banded_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), p, m.pos_bias_u, m.pos_bias_v, mask
        )
        return m.linear_out(result.reshape(b, t, width))


def install_span_attention(encoder):
    if encoder.att_context_style != "chunked_limited_with_rc" or tuple(encoder.att_context_size) != (70, 7, 4):
        raise ValueError("Banded span attention requires the released [70, 7, 4] mask")
    for layer in encoder.layers:
        layer.self_attn.forward = _SpanAttention(layer.self_attn)
