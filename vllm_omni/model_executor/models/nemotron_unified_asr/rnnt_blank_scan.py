# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Search several acoustic frames with one unchanged greedy RNNT predictor state."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _read_frames(enc, times, lengths, out, sr, st, sc, width: tl.constexpr, steps: tl.constexpr, block: tl.constexpr):
    row, k = tl.program_id(0), tl.program_id(1)
    time = tl.minimum(tl.load(times + row) + k, tl.load(lengths + row) - 1)
    c = tl.arange(0, block)
    x = tl.load(enc + row * sr + tl.maximum(time, 0) * st + c * sc, c < width, 0)
    tl.store(out + (row * steps + k) * width + c, x, c < width)


@triton.jit
def _take_first_label(
    window_labels,
    window_scores,
    times,
    lengths,
    selected_times,
    safe_times,
    labels,
    scores,
    blank_mask,
    active_mask,
    advance_mask,
    batch,
    blank: tl.constexpr,
    from_first: tl.constexpr,
    steps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0) * block + tl.arange(0, block)
    k = tl.arange(0, steps)
    live = row < batch
    if from_first:
        advance = tl.load(active_mask + row, live, 0)
    else:
        advance = tl.load(advance_mask + row, live, 0)
    start = tl.load(times + row, live, 0)
    end = tl.load(lengths + row, live, 0)
    n = tl.minimum(steps, tl.maximum(0, end - start))
    ids = tl.load(window_labels + row[:, None] * steps + k[None, :], live[:, None], blank)
    first = tl.min(tl.where((k[None, :] < n[:, None]) & (ids != blank), k[None, :], steps), axis=1)
    hit = first < steps
    choice = tl.where(hit, first, tl.maximum(n - 1, 0))
    index = row * steps + choice
    label = tl.load(window_labels + index, live, blank)
    score = tl.load(window_scores + index, live, 0.0)
    time = start + tl.where(hit, first, n)
    write = live & advance
    tl.store(labels + row, label, write)
    tl.store(scores + row, score, write)
    tl.store(selected_times + row, start + choice, write)
    tl.store(times + row, time, write)
    tl.store(safe_times + row, tl.minimum(time, end - 1), write)
    tl.store(blank_mask + row, ~hit, write)
    tl.store(active_mask + row, time < end, write)
    tl.store(advance_mask + row, advance & (~hit) & (time < end), live)


def install_blank_scan(decoder, *, from_first=False, fuse_joint=False):
    original = decoder._inner_loop_step_find_next_non_blank

    joint = decoder.joint
    if fuse_joint:
        modules = tuple(joint.joint_net)
        linear = modules[-1]
        if not (
            len(modules) == 3
            and isinstance(modules[0], torch.nn.ReLU)
            and isinstance(modules[1], torch.nn.Dropout)
            and not modules[1].training
            and isinstance(linear, torch.nn.Linear)
            and linear.bias is not None
            and linear.weight.dtype == torch.bfloat16
            and linear.weight.is_contiguous()
            and not joint.log_softmax
            and not joint.is_adapter_available()
        ):
            raise ValueError("Fused RNNT joint requires the released BF16 ReLU greedy head")

    def scan(original_step, initial):
        state = decoder.state
        if decoder.record_all_steps or state.need_nb_logits or decoder.has_fusion_models():
            return original_step()
        enc = state.encoder_output_projected
        batch, _, width = enc.shape
        steps = 4
        if (
            fuse_joint
            and enc.dtype == state.decoder_output.dtype == torch.bfloat16
            and state.decoder_output.stride(-1) == 1
        ):
            from .rnnt_joint import greedy_joint

            scores, labels = greedy_joint(
                enc, state.decoder_output, state.time_indices, state.encoder_output_length, linear, steps
            )
        else:
            frames = enc.new_empty((batch, steps, width))
            _read_frames[(batch, steps)](
                enc,
                state.time_indices,
                state.encoder_output_length,
                frames,
                *enc.stride(),
                width,
                steps,
                triton.next_power_of_2(width),
            )
            # A blank advances acoustic time without updating the predictor. Frames
            # past the first nonblank are discarded before NeMo advances the LSTM.
            logits = decoder.joint.joint_after_projection(frames, state.decoder_output).squeeze(2)
            scores, labels = logits.max(-1)
        _take_first_label[(triton.cdiv(batch, 32),)](
            labels,
            scores,
            state.time_indices,
            state.encoder_output_length,
            state.time_indices_current_labels,
            state.safe_time_indices,
            state.labels,
            state.scores,
            state.blank_mask,
            state.active_mask,
            state.advance_mask,
            batch,
            decoder._blank_index,
            initial,
            steps,
            32,
        )
        torch.any(state.advance_mask, out=state.advance_mask_any)

    decoder._inner_loop_step_find_next_non_blank = lambda: scan(original, False)
    if from_first:
        before = decoder._before_inner_loop_get_joint_output
        decoder._before_inner_loop_get_joint_output = lambda: scan(before, True)
