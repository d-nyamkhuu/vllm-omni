# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One independently scheduled RNNT window per active speech stream."""

from bisect import bisect_left
from pathlib import Path

import torch
from torch import nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .encoder_graph import BufferedEncoderGraphs


class NemotronUnifiedASR(nn.Module):
    have_multimodal_outputs = True
    enable_update_additional_information = True
    requires_raw_input_tokens = True
    has_preprocess = False
    has_postprocess = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
        from omegaconf import OmegaConf

        mc = vllm_config.model_config
        self.config = mc.hf_config
        self.model_dir = Path(mc.model)
        self.device = vllm_config.device_config.device
        if not mc.trust_remote_code:
            raise ValueError("Nemotron ASR requires --trust-remote-code for its NeMo component configuration")
        cfg = OmegaConf.create(self.config.asr_config)
        root = self.model_dir
        cfg.tokenizer.dir = str(root)
        for field in ("model_path", "vocab_path", "spe_tokenizer_vocab"):
            cfg.tokenizer[field] = str(root / cfg.tokenizer[field])
        cfg.train_ds = cfg.validation_ds = None
        # Reproduce NeMo's FP32 construction and subsequent BF16 inference cast.
        dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            with torch.device("cpu"):
                self.asr = EncDecRNNTBPEModelWithPrompt(cfg=cfg)
        finally:
            torch.set_default_dtype(dtype)
        self.asr.to(device=self.device, dtype=mc.dtype).eval()
        decoding = OmegaConf.structured(RNNTDecodingConfig())
        decoding.strategy = "greedy_batch"
        decoding.fused_batch_size = -1
        self.asr.change_decoding_strategy(decoding)
        self.asr.preprocessor.featurizer.dither = 0.0
        self.asr.preprocessor.featurizer.pad_to = 0
        self.asr.encoder.sync_max_audio_length = False
        self.asr.encoder.set_default_att_context_size([70, 7, 4])
        from .pcm_ring import PCMWindowRing

        self.device_pcm = PCMWindowRing(self.device, (70 + 7 + 4) * 1280)
        self.decoder_states = {}

    def load_weights(self, weights):
        params = self.state_dict()
        loaded = set()
        for name, weight in weights:
            if name not in params or params[name].shape != weight.shape:
                raise ValueError(f"Unexpected ASR checkpoint tensor {name} {weight.shape}")
            params[name].copy_(weight)
            loaded.add(name)
        if missing := params.keys() - loaded:
            raise ValueError(f"Missing ASR checkpoint tensors: {sorted(missing)}")
        native = BufferedEncoderGraphs(
            self.asr.encoder.forward,
            frame_buckets=(64, 128, 256, 384, 512, 624, 736, 896, 1024, 1280, 1536, 1792, 2048),
            max_graphs=12,
            reuse_larger=True,
            batch_buckets=(1, 2, 4, 8, 16, 32, 48, 64, 96, 128),
        )
        if self.config.packed_streaming:
            from .ragged_spans import RaggedSpanEncoder
            from .rnnt_blank_scan import install_blank_scan
            from .span_attention import install_span_attention

            install_blank_scan(self.asr.decoding.decoding.decoding_computer, from_first=True, fuse_joint=True)
            install_span_attention(self.asr.encoder)
            self.ragged_encoder = RaggedSpanEncoder(self.asr.encoder, native)
            self.asr.encoder.forward = self.ragged_encoder
        else:
            self.asr.encoder.forward = native
        return loaded

    def embed_input_ids(self, input_ids, **kwargs):
        return torch.zeros((input_ids.numel(), self.config.hidden_size), device=input_ids.device)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return None

    def make_omni_output(self, output, **kwargs):
        return output

    @torch.inference_mode()
    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        model_intermediate_buffer=None,
        **kwargs,
    ):
        # MRv2 runs a dummy forward during initialization. Audio arrives by worker RPC.
        return OmniOutput(text_hidden_states=None, multimodal_outputs={})

    @torch.inference_mode()
    def process_realtime_audio_batch(self, payload):
        if "asr_abort" in payload:
            stream = payload["asr_abort"]
            self.decoder_states.pop(stream, None)
            self.device_pcm.release(stream)
            return {"parts": []}
        audio = payload["pcm"]
        if not 0 < len(audio) <= 512:
            raise ValueError("Invalid number of ASR windows")
        rows = []
        for i, (blob, n) in enumerate(zip(audio, payload["asr_batch_lengths"], strict=True)):
            if len(blob) != n * 2:
                raise ValueError("Inconsistent ASR PCM byte length")
            row = {k: payload[k][i] for k in ("asr_stream", "asr_lang", "asr_left", "asr_final", "asr_keep")}
            row["asr_pcm"] = (
                torch.frombuffer(bytearray(blob), dtype=torch.int16) if blob else torch.empty(0, dtype=torch.int16)
            )
            rows.append(row)
        return {"parts": self.decode_rows(rows)}

    def decode_rows(self, rows):
        from .decoder_frames import DecoderFramePlan
        from .window_spans import pack_ready_windows

        streams = {}
        for i, row in enumerate(rows):
            streams.setdefault(row["asr_stream"], []).append(i)
        groups = list(streams.values())
        hop, tail = 7, 12
        temporal_batch = len(groups) < len(rows)
        if self.config.packed_streaming:
            starts, selected, parent, left = pack_ready_windows(rows)
            audio, lengths, samples = self.device_pcm(rows, selected, starts, host_lengths=True)
        else:
            selected = range(len(rows))
            parent, left = None, None
            audio, lengths = self.device_pcm(rows)
        frame_plan = DecoderFramePlan(rows, groups, hop, tail, False, self.device, parents=parent, offsets=left)
        langs = torch.tensor([rows[i]["asr_lang"] for i in selected], device=self.device)
        if self.config.packed_streaming:
            self.ragged_encoder.samples = samples
        try:
            enc, enc_len = self.asr(input_signal=audio, input_signal_length=lengths, language_prompt_indices=langs)
        finally:
            if self.config.packed_streaming:
                self.ragged_encoder.samples = None
        x, lens = frame_plan.gather(enc, enc_len)
        decoder = self.asr.decoding.decoding.decoding_computer
        previous = [self.decoder_states.get(stream) for stream in streams]
        state = decoder.merge_to_batched_state(previous) if any(s is not None for s in previous) else None
        begin = state.decoded_lengths if state is not None else torch.zeros_like(lens)
        hyps, state = decoder(x=x, out_len=lens, prev_batched_state=state, multi_biasing_ids=None)
        states = decoder.split_batched_state(state)
        counts, labels, timestamps, *_ = hyps.get_data_without_blank()
        # NeMo may retain a larger decoder graph from the previous invocation.
        counts, timestamps = counts[: len(groups)], timestamps[: len(groups)]
        tokens = [None] * len(rows)
        # NeMo's history buffers can hold hundreds of frames. A hop can
        # emit at most max_symbols IDs per valid decoder frame.
        symbols = getattr(decoder, "max_symbols", None)
        limit = labels.shape[1] if symbols is None else min(labels.shape[1], symbols * x.shape[1])
        fields = [counts[:, None]]
        if temporal_batch:
            fields.append(timestamps[:, :limit] - begin[:, None])
        fields.append(labels[: len(groups), :limit])
        bounds = torch.cat(fields, dim=1).tolist()
        label_start = 1 + (limit if temporal_batch else 0)
        for j, (stream, indices) in enumerate(streams.items()):
            count = bounds[j][0]
            times = bounds[j][1 : count + 1]
            first = 0
            for step, i in enumerate(indices):
                end = count if i == indices[-1] else bisect_left(times, hop * (step + 1), lo=first)
                tokens[i] = bounds[j][label_start + first : label_start + end]
                first = end
            if rows[indices[-1]]["asr_final"]:
                self.decoder_states.pop(stream, None)
            else:
                self.decoder_states[stream] = states[j]
        return tokens
