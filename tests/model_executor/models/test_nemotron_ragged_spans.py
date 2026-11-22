# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton CUDA kernels")
def test_packed_attention_and_convolution_keep_span_boundaries_on_graph_replay():
    from vllm_omni.model_executor.models.nemotron_unified_asr.ragged_spans import _depthwise, pack_layout
    from vllm_omni.model_executor.models.nemotron_unified_asr.span_attention import _banded_attention, banded_attention

    torch.manual_seed(31)
    lengths = [95, 130, 21]
    starts, tiles = pack_layout([n * 1280 - 1 for n in lengths])
    dims = 128
    x = torch.randn(len(tiles) * 16, dims, dtype=torch.bfloat16, device="cuda")
    k, v, u, w = torch.randn_like(x), torch.randn_like(x), torch.randn_like(x[0]), torch.randn_like(x[0])
    pos = torch.randn(255, dims, device="cuda", dtype=x.dtype)
    weight, bias = torch.randn(dims, 1, 9, device=x.device, dtype=x.dtype), torch.randn_like(u)
    meta = torch.empty((len(tiles), 3), dtype=torch.int32, device=x.device)
    y, z = torch.empty_like(x), torch.empty_like(x)

    def update(sizes):
        meta.copy_(torch.tensor([(base, i, sizes[r] if r >= 0 else 0) for r, base, i in tiles], device=x.device))

    def run():
        _banded_attention[(1, 1, len(tiles))](x, k, v, pos, u, w, meta, y, 128, 1, dims, 16, 128, True, num_warps=4)
        _depthwise[(len(tiles), 1)](v, meta, weight, bias, z, dims, 9, 8, True, 128)

    update(lengths)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for sizes in (lengths, [80, 118, 1], [95, 130, 0]):
        update(sizes)
        graph.replay()
        for start, n in zip(starts, sizes, strict=True):
            if not n:
                continue
            sl = slice(start, start + n)
            idx = torch.arange(n, device=x.device)
            chunk = (idx // 7 * 7)[:, None]
            mask = (idx < chunk - 70) | (idx > chunk + 10)
            # The dense and packed kernels must use the same keys for every query,
            # even where a span wraps across attention tiles or another stream follows it.
            offset = 128 - n
            table = pos[offset : 255 - offset] if offset >= 0 else F.pad(pos, (0, 0, -offset, -offset))
            dense = banded_attention(
                x[sl][None, :, None],
                k[sl][None, :, None],
                v[sl][None, :, None],
                table[:, None],
                u[None],
                w[None],
                mask[None],
            )
            peer = dense[0, :, 0]
            # The dense specialization uses compile-time strides. One BF16 ULP
            # covers rounding between those two tensor-core layouts.
            step = (torch.nextafter(peer, y[sl]) - peer).abs().float()
            assert torch.all((y[sl].float() - peer.float()).abs() <= step + 1.0e-6)
            ac = ((x[sl] + u).float() @ k[sl].float().T).to(x.dtype)
            bd = ((x[sl] + w).float() @ pos.float().T).to(x.dtype).gather(-1, (127 - idx[:, None] + idx).clamp(0, 254))
            probs = ((ac + bd) / dims**0.5).masked_fill(mask, -10000).softmax(-1)
            torch.testing.assert_close(y[sl], (probs.float() @ v[sl].float()).to(x.dtype), atol=0.008, rtol=0.016)
            expected = F.conv1d(F.pad(v[sl].T[None].float(), (8, 0)), weight.float(), bias.float(), groups=dims)
            torch.testing.assert_close(z[sl], expected[0].T.to(x.dtype), atol=0.008, rtol=0.008)

    # Modifying every other stream's keys and values must have exactly zero
    # effect on the first stream on the same captured graph.
    update(lengths)
    graph.replay()
    first_attention, first_conv = y[: lengths[0]].clone(), z[: lengths[0]].clone()
    k[starts[1] :].fill_(42)
    v[starts[1] :].fill_(-17)
    graph.replay()
    torch.testing.assert_close(y[: lengths[0]], first_attention, rtol=0, atol=0)
    torch.testing.assert_close(z[: lengths[0]], first_conv, rtol=0, atol=0)


def test_native_span_graph_budget_and_interleaved_replays():
    import pytest

    from vllm_omni.model_executor.models.nemotron_unified_asr.encoder_graph import BufferedEncoderGraphs

    if not torch.cuda.is_available():
        pytest.skip("CUDA graph replay")

    def native(audio_signal, length):
        return audio_signal[..., ::8].contiguous().square(), (length + 7) // 8

    graphs = BufferedEncoderGraphs(native, frame_buckets=(16, 32, 64), batch_buckets=(1, 2, 4), max_graphs=2)
    batches = [(1, 11), (2, 27)] * 3 + [(3, 53)] * 4 + [(1, 9), (2, 17)] * 3
    for i, (batch, width) in enumerate(batches):
        signal = torch.randn(batch, 32, width, device="cuda")
        lengths = torch.arange(width - batch + 1, width + 1, device="cuda")
        got = graphs(signal, lengths)
        for output, expected in zip(got, native(signal, lengths), strict=True):
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if i < 4:
            assert graphs.captures == 0
    assert graphs.captures == 2 and graphs.replays == 8 and graphs.pool is None


def test_covering_native_span_graph_uses_larger_shape_and_keeps_single_stream_separate():
    import pytest

    from vllm_omni.model_executor.models.nemotron_unified_asr.encoder_graph import BufferedEncoderGraphs

    if not torch.cuda.is_available():
        pytest.skip("CUDA graph replay")

    def native(audio_signal, length):
        return audio_signal[..., ::8].contiguous().square(), (length + 7) // 8

    graphs = BufferedEncoderGraphs(
        native, frame_buckets=(24, 32, 64), batch_buckets=(1, 2, 4), max_graphs=2, reuse_larger=True
    )
    for batch, width in [(4, 31)] * 3 + [(3, 23)] * 3 + [(1, 22)] * 3 + [(2, 62)] * 3:
        signal = torch.randn(batch, 32, width, device="cuda")
        lengths = torch.arange(width - batch + 1, width + 1, device="cuda")
        for value, expected in zip(graphs(signal, lengths), native(signal, lengths), strict=True):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert {(k[0], k[2]) for k in graphs.graphs} == {(4, 32), (1, 24)}
    assert graphs.replays == 5
