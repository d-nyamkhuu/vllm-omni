# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton CUDA attention")
@pytest.mark.parametrize("t", [7, 77, 129, 256])
def test_banded_attention_matches_dense_relative_mask_and_graph(t):
    from vllm_omni.model_executor.models.nemotron_unified_asr.span_attention import banded_attention

    b, h, d = 2, 2, 128
    rng = torch.Generator(device="cuda").manual_seed(793 + t)

    def rand(*shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)

    q, k, v = [rand(b, t, h, d) for _ in range(3)]
    pos, u, w = rand(2 * t - 1, h, d), rand(h, d), rand(h, d)
    idx = torch.arange(t, device=q.device)
    chunk = (idx // 7 * 7)[:, None]
    mask = ((idx < chunk - 70) | (idx > chunk + 10)).expand(b, t, t).clone()
    rel = (t - 1 - idx[:, None] + idx).expand(b, h, t, t)
    banded_attention(q, k, v, pos, u, w, mask)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = banded_attention(q, k, v, pos, u, w, mask)
    for length in (t - 2, t // 2):
        mask[1, length:, :] = True
        mask[1, :, length:] = True
        graph.replay()
        ac = (q + u).transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)
        bd = ((q + w).transpose(1, 2) @ pos.permute(1, 2, 0)).gather(-1, rel)
        score = ((ac + bd) / d**0.5).masked_fill(mask[:, None], -10000)
        expected = (score.softmax(-1).masked_fill(mask[:, None], 0) @ v.transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.016)
        assert not actual[1, length:].count_nonzero()
