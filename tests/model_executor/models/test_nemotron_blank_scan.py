# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.nemotron_unified_asr.rnnt_blank_scan import install_blank_scan

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph RNNT scan")
def test_blank_scan_stops_at_first_nonblank_in_each_live_stream():
    labels = [
        [3, 3, 2, 0, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 2, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3],
        [3, 1, 2, 2, 0, 0, 0, 0],
        [3, 3, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    def tensor(x):
        return torch.tensor(x, device="cuda")

    f = torch.nn.functional.one_hot(tensor(labels), 4).to(torch.bfloat16) * 8
    batch = len(labels)
    lengths = tensor([8, 8, 8, 3, 3, 0])
    state = SimpleNamespace(
        encoder_output_projected=f,
        encoder_output_length=lengths,
        decoder_output=f.new_zeros((batch, 1, 4)),
        time_indices=tensor([0, 1, 2, 0, 1, 0]),
        time_indices_current_labels=tensor([-1] * batch),
        safe_time_indices=tensor([0, 1, 2, 0, 1, -1]),
        labels=tensor([3, 3, 3, 1, 3, 3]),
        scores=f.new_full((batch,), 5),
        blank_mask=tensor([True, True, True, False, True, True]),
        active_mask=tensor([True, True, True, True, True, False]),
        advance_mask=tensor([True, True, True, False, True, False]),
        advance_mask_any=tensor(True),
        need_nb_logits=False,
    )
    original = []
    decoder = SimpleNamespace(
        state=state,
        joint=SimpleNamespace(joint_after_projection=lambda x, y: x.unsqueeze(2) + y.unsqueeze(1)),
        record_all_steps=False,
        has_fusion_models=lambda: False,
        _blank_index=3,
        _inner_loop_step_find_next_non_blank=lambda: original.append(True),
    )
    install_blank_scan(decoder)
    call = decoder._inner_loop_step_find_next_non_blank
    initial = {k: v.clone() for k, v in vars(state).items() if isinstance(v, torch.Tensor)}
    call()
    assert state.time_indices.tolist() == [2, 5, 6, 0, 3, 0]
    assert state.advance_mask.tolist() == [False, True, True, False, False, False]
    for k, v in initial.items():
        getattr(state, k).copy_(v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    graph.replay()
    assert state.labels.tolist() == [2, 2, 3, 1, 3, 3]
    assert state.time_indices.tolist() == [2, 5, 8, 0, 3, 0]
    assert state.time_indices_current_labels.tolist() == [2, 5, 7, -1, 2, -1]
    assert state.scores.tolist() == [8, 8, 8, 5, 8, 5]
    assert state.active_mask.tolist() == [True, True, False, True, False, False]
    assert not state.advance_mask_any.item()
    decoder.record_all_steps = True
    call()
    assert original == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton joint projection and graph replay")
@pytest.mark.parametrize("width,vocab", [(65, 129), (640, 4096)])
def test_fused_joint_matches_bf16_native_max_with_cross_tile_ties(width, vocab):
    from vllm_omni.model_executor.models.nemotron_unified_asr.rnnt_joint import greedy_joint

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(927)
    batch, steps = 9, 4
    enc = torch.randint(-2, 3, (batch, 8, width), device=device, generator=g).bfloat16()[:, ::2]
    predictor = torch.randint(-2, 3, (batch, 1, width), device=device, generator=g).bfloat16()
    lengths = torch.arange(batch, device=device) % 5
    times = lengths // 2
    linear = torch.nn.Linear(width, vocab, device=device, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        linear.weight.copy_(torch.randint(-1, 2, linear.weight.shape, device=device, generator=g))
        linear.bias.zero_()
        # These rows tie across different vocabulary tiles.
        linear.weight[[1, 128]] = 0
        linear.bias[[1, 128]] = 2048
        greedy_joint(enc, predictor, times, lengths, linear, steps)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scores, labels = greedy_joint(enc, predictor, times, lengths, linear, steps)
        for round_no in range(2):
            if round_no:
                linear.bias.zero_()
                times.zero_()
            graph.replay()
            t = torch.minimum(times[:, None] + torch.arange(steps, device=device), lengths[:, None] - 1).clamp_min(0)
            frames = enc[torch.arange(batch, device=device)[:, None], t]
            native_scores, native_labels = linear((frames + predictor).relu()).max(-1)
            torch.testing.assert_close(scores, native_scores, rtol=0, atol=0)
            torch.testing.assert_close(labels.long(), native_labels)
