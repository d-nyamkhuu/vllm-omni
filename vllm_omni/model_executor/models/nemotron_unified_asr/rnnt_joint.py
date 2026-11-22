# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Find greedy RNNT labels while keeping vocabulary tiles in registers."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _joint_tiles(
    enc,
    pred,
    times,
    lens,
    weight,
    bias,
    partial,
    ids,
    batch,
    stride_b,
    stride_t,
    stride_c,
    pred_b,
    width: tl.constexpr,
    vocab: tl.constexpr,
    tiles: tl.constexpr,
    steps: tl.constexpr,
    bm: tl.constexpr,
    bn: tl.constexpr,
    bk: tl.constexpr,
):
    m = tl.program_id(0) * bm + tl.arange(0, bm)
    tile = tl.program_id(1)
    n = tile * bn + tl.arange(0, bn)
    row = m // steps
    time = tl.load(times + row, row < batch, 0) + m % steps
    last = tl.load(lens + row, row < batch, 1) - 1
    time = tl.maximum(0, tl.minimum(time, last))
    k = tl.arange(0, bk)
    acc = tl.full((bm, bn), 0, tl.float32)
    for i in range(triton.cdiv(width, bk)):
        col = i * bk + k
        f = tl.load(
            enc + row[:, None] * stride_b + time[:, None] * stride_t + col[None, :] * stride_c,
            (row < batch)[:, None] & (col < width)[None, :],
            0,
        )
        g = tl.load(pred + row[:, None] * pred_b + col[None, :], (row < batch)[:, None] & (col < width)[None, :], 0)
        # The release rounds the BF16 sum before applying ReLU.
        h = tl.maximum((f.to(tl.float32) + g.to(tl.float32)).to(tl.bfloat16), 0).to(tl.bfloat16)
        w = tl.load(weight + n[:, None] * width + col[None, :], (n < vocab)[:, None] & (col < width)[None, :], 0)
        acc = tl.dot(h, tl.trans(w), acc)
    b = tl.load(bias + n, n < vocab, 0)
    logits = (acc + b[None, :]).to(tl.bfloat16).to(tl.float32)
    logits = tl.where(n[None, :] < vocab, logits, -float("inf"))
    best = tl.max(logits, 1)
    label = tl.min(tl.where(logits == best[:, None], n[None, :], vocab), 1)
    offset = m * tiles + tile
    tl.store(partial + offset, best, row < batch)
    tl.store(ids + offset, label, row < batch)


@triton.jit
def _joint_winners(partial, ids, scores, labels, tiles: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, block)
    x = tl.load(partial + row * tiles + k, k < tiles, -float("inf")).to(tl.float32)
    indices = tl.load(ids + row * tiles + k, k < tiles, 2147483647)
    top = tl.max(x, 0)
    tl.store(scores + row, top)
    tl.store(labels + row, tl.min(tl.where(x == top, indices, 2147483647), 0))


def greedy_joint(enc, pred, times, lengths, linear, steps):
    batch, _, width = enc.shape
    vocab = linear.out_features
    tiles = triton.cdiv(vocab, 64)
    partial = enc.new_empty((batch * steps, tiles))
    ids = torch.empty_like(partial, dtype=torch.int32)
    scores = enc.new_empty((batch, steps))
    labels = torch.empty_like(scores, dtype=torch.int32)
    _joint_tiles[(triton.cdiv(batch * steps, 16), tiles)](
        enc,
        pred,
        times,
        lengths,
        linear.weight,
        linear.bias,
        partial,
        ids,
        batch,
        *enc.stride(),
        pred.stride(0),
        width,
        vocab,
        tiles,
        steps,
        16,
        64,
        64,
        num_warps=4,
    )
    _joint_winners[(batch * steps,)](partial, ids, scores, labels, tiles, triton.next_power_of_2(tiles))
    return scores, labels
