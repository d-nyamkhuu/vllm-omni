# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.nemotron_unified_asr.pcm_ring import PCMWindowRing

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton PCM assembly")
def test_pcm_ring_matches_buffered_windows_bit_for_bit():
    ring = PCMWindowRing("cuda")
    ref: dict[str, torch.Tensor] = {}
    gen = torch.Generator().manual_seed(291)

    def send(spec, selected=None):
        rows, windows = [], []
        for stream, n, final in spec:
            old = ref.get(stream, torch.empty(0, dtype=torch.int16))
            add = torch.randint(-32768, 32768, (n,), generator=gen, dtype=torch.int16)
            keep = min(len(old), ring.capacity - n)
            window = torch.cat((old[len(old) - keep :], add))
            ref[stream] = window
            rows.append(dict(asr_stream=stream, asr_pcm=add, asr_keep=keep, asr_final=final))
            windows.append(window)
            if final:
                del ref[stream]
        got, length = ring(rows, selected)
        if selected is not None:
            windows = [windows[i] for i in selected]
        assert length.tolist() == [len(w) for w in windows]
        expected = torch.nn.utils.rnn.pad_sequence(windows, batch_first=True).float() / 32768
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)

    send([("a", 8960, False), ("b", 8960, False)])
    # A batched burst exceeds the entire ring. Old rows still see their own prefix.
    send([("a", 2560, False), ("b", 2560, False)] * 43, selected=[1, 0, 43, 85, 84])
    send([("a", 131, True), ("b", 0, True), ("c", 713, True)])
    assert not ring.states
    send([("a", 8960, False)])
    ring.release("a")
    assert not ring.states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton PCM window assembly")
def test_borrowed_pcm_keeps_history_when_the_selected_end_wraps():
    ring = PCMWindowRing("cuda", capacity=32)
    data = torch.arange(120, dtype=torch.int16)

    def batch(spec, selected=None, starts=None, trims=None):
        rows = [
            dict(asr_stream="x", asr_pcm=data[end - new : end], asr_keep=keep, asr_final=False)
            for end, new, keep in spec
        ]
        return ring(rows, selected, starts, trims)

    batch([(32, 32, 0)])
    for base in (32, 64):
        spec = [(base + i * 4, 4, 28) for i in range(1, 9)]
        audio, sizes = batch(spec, [7, 7, 3], [0, 7, 2], [5, 0, 0])
        assert sizes.tolist() == [55, 32, 36]
        for i, (first, end) in enumerate(((base - 28, base + 27), (base, base + 32), (base - 20, base + 16))):
            torch.testing.assert_close(audio[i, : end - first].cpu(), data[first:end].float() / 32768, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device PCM ring")
@pytest.mark.parametrize(
    ("regular", "remainder", "joined"), [(3, 1, True), (17, 0, True), (17, 1280, True), (17, 1279, False)]
)
def test_full_ready_flush_preserves_native_final_pcm_and_frame_grid(regular, remainder, joined):
    from vllm_omni.model_executor.models.nemotron_unified_asr.pcm_ring import PCMWindowRing
    from vllm_omni.model_executor.models.nemotron_unified_asr.window_spans import pack_ready_windows

    samples = [11 * 1280] + [7 * 1280] * (regular - 1) + [remainder]
    pcm = torch.arange(sum(samples), dtype=torch.int32).to(torch.int16)
    rows = []
    history = end = 0
    for i, n in enumerate(samples):
        excess = max(0, history + n - 81 * 1280)
        rows.append(
            dict(
                asr_final=i == regular,
                asr_left=max(0, history - 4 * 1280 - excess) // 1280,
                asr_pcm=pcm[end : end + n],
                asr_keep=history - excess,
                asr_stream="voice",
            )
        )
        end += n
        history = min(81 * 1280, history + n)
    starts, selected, parent, offsets = pack_ready_windows(rows)
    final = parent[-1]
    assert rows[selected[final]]["asr_final"]
    assert (len(starts) == 1) == joined
    ring = PCMWindowRing("cuda", capacity=81 * 1280)
    audio, lengths = ring(rows, selected, starts)
    final_len = int(lengths[final])
    assert not ring.states
    # The decoder sees the same absolute beginning and end of the final hop.
    start = 1280 * offsets[-1]
    native_start = len(pcm) - history + rows[-1]["asr_left"] * 1280
    torch.testing.assert_close(audio[final, start:final_len].cpu(), pcm[native_start:].float() / 32768, rtol=0, atol=0)
