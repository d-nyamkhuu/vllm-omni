# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _residual_norm(
    skip,
    branch,
    weight,
    bias,
    norm_out,
    skip_out,
    width: tl.constexpr,
    scale: tl.constexpr,
    eps: tl.constexpr,
    save_skip: tl.constexpr,
    has_bias: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, block)
    mask = c < width
    x = tl.load(skip + row * width + c, mask, 0)
    delta = tl.load(branch + row * width + c, mask, 0)
    dt: tl.constexpr = skip.dtype.element_ty
    # Match the BF16 rounding points in scaled branch, residual and norm.
    scaled = (delta.to(tl.float32) * scale).to(dt)
    value = (x.to(tl.float32) + scaled.to(tl.float32)).to(dt)
    if save_skip:
        tl.store(skip_out + row * width + c, value, mask)
    z = value.to(tl.float32)
    centered = z - tl.sum(z, 0) / width
    var = tl.sum(tl.where(mask, centered * centered, 0), 0) / width
    w = tl.load(weight + c, mask, 0).to(tl.float32)
    result = centered * tl.rsqrt(var + eps) * w
    if has_bias:
        result += tl.load(bias + c, mask, 0).to(tl.float32)
    tl.store(norm_out + row * width + c, result, mask)


def residual_norm(skip, branch, norm, scale, save_skip=True):
    output = torch.empty_like(skip)
    summed = torch.empty_like(skip) if save_skip else output
    width = skip.shape[-1]
    _residual_norm[(skip.shape[0],)](
        skip,
        branch,
        norm.weight,
        norm.bias,
        output,
        summed,
        width,
        scale,
        norm.eps,
        save_skip,
        norm.bias is not None,
        triton.next_power_of_2(width),
    )
    return summed, output
