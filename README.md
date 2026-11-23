# Ming-3.0-TTS-0.5B Support & Dense Streaming Optimizations (vLLM-Omni Fork)

This fork of [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) adds support for [`inclusionAI/Ming-3.0-tts-0.5b`](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5b) (`patch_size=1` dense 0.5B checkpoint) and a high-throughput, low-latency dense streaming profile on CUDA.

## Branches & Upstream PRs

| Branch | Status | Scope |
| --- | --- | --- |
| [`pr/ming3-dense-0.5b`](https://github.com/d-nyamkhuu/vllm-omni/tree/pr/ming3-dense-0.5b) | Open PR [#10977](https://github.com/vllm-project/vllm-omni/pull/10977) | Model support for `inclusionAI/Ming-3.0-tts-0.5b` (offline `end2end.py` + online `/v1/audio/speech`, voice cloning, seed determinism), keeping legacy `Ming-omni-tts` (`patch_size=4`) unchanged |
| [`pr/ming3-dense-streaming-perf`](https://github.com/d-nyamkhuu/vllm-omni/tree/pr/ming3-dense-streaming-perf) | Stacked on `#10977` (contains **both PR 1 + PR 2**) | GPU-resident MRV2 acoustic feedback, vectorized + CUDA-graphed CFM decode, single batched D2H stop check, batched streaming AudioVAE, and bundled `vllm_omni/deploy/ming_tts_dense_streaming.yaml` |

> Want to try both PRs together right now? Check out [`pr/ming3-dense-streaming-perf`](https://github.com/d-nyamkhuu/vllm-omni/tree/pr/ming3-dense-streaming-perf).

## Measured Results (`inclusionAI/Ming-3.0-tts-0.5b`)

Comparable unseeded raw PCM streaming (`stream=true`, `stream_format="audio"`, 44.1 kHz mono 16-bit PCM, 3 measured rounds per concurrency after warmup) on **1× NVIDIA RTX PRO 6000 Blackwell (96 GB)** with `vLLM 0.34.0`, `PyTorch 2.14.0+cu130`, `CUDA 13.0`, `Transformers 5.22.2`:

- **Baseline (PR 1, `vllm_omni/deploy/ming_tts.yaml`):** `max_num_seqs: 1`, `enforce_eager: true`, `latent_chunk_size: 25`
- **Dense Streaming (PR 2, `vllm_omni/deploy/ming_tts_dense_streaming.yaml`):** `max_num_seqs: 32`, `ming_dense_streaming: true`, Stage-0 PIECEWISE + CFM CUDA graphs, batched streaming VAE, `latent_chunk_size: 8`

| Concurrency | Profile | TTFA mean ± std (ms) | TTFA p95 (ms) | E2E mean (ms) | RTF mean | RPS mean | Audio-sec/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Baseline (`ming_tts.yaml`) | 141.78 ± 1.65 | 142.94 | 1496.59 | 0.3716 | 0.669 | 2.69 |
| 1 | **Dense Streaming** | **45.60 ± 0.43 (3.1×)** | **46.04** | **409.79 (3.65×)** | **0.1045 (3.6×)** | **2.438 (3.6×)** | **9.56 (3.6×)** |
| 4 | Baseline (`ming_tts.yaml`) | 2186.46 ± 82.88 | 4422.93 | 3584.91 | 0.8148 | 0.683 | 2.91 |
| 4 | **Dense Streaming** | **63.02 ± 2.24 (34.7×)** | **72.34** | **532.17 (6.7×)** | **0.1244** | **6.446 (9.4×)** | **27.58 (9.5×)** |
| 8 | Baseline (`ming_tts.yaml`) | 5312.65 ± 175.27 | 10587.45 | 6764.77 | 1.5298 | 0.663 | 2.91 |
| 8 | **Dense Streaming** | **71.13 ± 1.29 (74.7×)** | **80.07** | **617.74 (11.0×)** | **0.1370** | **10.964 (16.5×)** | **49.52 (17.0×)** |
| 16 | **Dense Streaming** | **89.74 ± 0.79** | **102.90** | **703.52** | **0.1567** | **19.475** | **87.57** |

- **Quality & accuracy:** Identical Whisper `base` WER (`0.102`) between PR 1 and PR 2, byte-exact `seed=42` determinism, matching English transcripts and durations against official upstream [`inclusionAI/Ming-3.0-tts`](https://github.com/inclusionAI/Ming-3.0-tts), and live non-regression verified on legacy `inclusionAI/Ming-omni-tts-0.5B`.
- **Memory & startup:** Peak GPU VRAM `45,034 MiB` (baseline) vs. `49,984 MiB` (dense streaming with `max_num_seqs: 32` + CUDA graphs); `enforce_eager: true` cleanly disables CFM/PIECEWISE graph capture.

## Quickstart: Install & Run with Latest vLLM-Omni (`vllm==0.34.0`)

Requires Python 3.11–3.13 (3.12 recommended) on a CUDA BF16 GPU (tested on NVIDIA L4 24 GB and RTX PRO 6000 Blackwell 96 GB).

### Option A — Fresh `uv` environment

```bash
git clone --depth 1 -b pr/ming3-dense-streaming-perf https://github.com/d-nyamkhuu/vllm-omni.git
cd vllm-omni

uv venv .venv --python 3.12 --seed
source .venv/bin/activate

uv pip install vllm==0.34.0 setuptools_scm wheel
uv pip install --no-build-isolation --no-deps -e .
uv pip install soundfile soxr x-transformers einops onnxruntime omegaconf \
    diffusers==0.42.0 accelerate==1.13.0 prettytable aenum janus pyyaml msgpack av scipy
```

### Option B — Inside an existing `vllm-omni` (`vllm==0.34.0`) environment or Docker image

```bash
git clone --depth 1 -b pr/ming3-dense-streaming-perf https://github.com/d-nyamkhuu/vllm-omni.git
cd vllm-omni
uv pip install --no-build-isolation --no-deps -e .
```

### Serve Online (Dense Streaming Profile)

```bash
vllm serve inclusionAI/Ming-3.0-tts-0.5b --omni \
    --deploy-config vllm_omni/deploy/ming_tts_dense_streaming.yaml \
    --port 8091
```

*(For PR 1 baseline / eager mode only, pass `--deploy-config vllm_omni/deploy/ming_tts.yaml` instead.)*

**Synthesize WAV:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "inclusionAI/Ming-3.0-tts-0.5b",
      "input": "Streaming text to speech should start playing quickly while keeping natural rhythm.",
      "response_format": "wav",
      "seed": 42
    }' --output ming3_sample.wav
```

**Stream raw 44.1 kHz 16-bit mono PCM (omit `seed` for concurrent batching):**

```bash
curl -N -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "inclusionAI/Ming-3.0-tts-0.5b",
      "input": "Streaming text to speech should start playing quickly while keeping natural rhythm.",
      "response_format": "pcm",
      "stream": true,
      "stream_format": "audio"
    }' --output ming3_stream.pcm
```

### Offline Inference

```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --model inclusionAI/Ming-3.0-tts-0.5b \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --case style
```

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png">
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png" width=55%>
  </picture>
</p>
<h3 align="center">
Easy, fast, and cheap omni-modality model serving for everyone
</h3>

<p align="center">
| <a href="https://vllm-omni.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://deepwiki.com/vllm-project/vllm-omni"><b>DeepWiki</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> | <a href="docs/assets/WeChat.jpg"><b>WeChat</b></a> | <a href="https://arxiv.org/abs/2610.09307"><b>Technical Report</b></a> | <a href="https://arxiv.org/abs/2602.02204"><b>Paper</b></a> | <a href="https://docs.google.com/presentation/d/1SV7-4AoSrnK25syLG9XmnEgC0D4lcjP7/edit?usp=sharing"><b>Slides</b></a> |
</p>

---

*Latest News* 🔥

- [2026/10] We released [0.32.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.32.0), aligned with vLLM 0.32.0, featuring optional Ray-based multi-node inference, session-aware full-duplex serving, server VAD for turn-based realtime requests, and expanded image, music, and speech generation with YuE2-3B, Qwen-Image-2.1, Kandinsky6, MOSS-Image, and TaoMate-H3. See the release notes for compatibility changes and installation instructions.
- [2026/10] We released the [vLLM-Omni Technical Report](https://arxiv.org/abs/2610.09307), describing a unified serving runtime for text, audio, image, video, and action generation.
- [2026/09] We released [0.30.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.30.0), rebased onto vLLM 0.30.0, featuring a unified full-duplex serving framework around engine-owned sessions for [MiniCPM-o 4.5](recipes/OpenBMB/MiniCPM-o-4_5.md) and AURA, native cross-stage KV and multimodal payload transfer (Mooncake AR-to-DiT handoff and NIXL connectors), interactive world-model serving with [LingBot World](recipes/Robbyant/LingBot-World-2.0.md), and realtime [MiniMax H3](recipes/MiniMaxAI/MiniMax-H3.md) Turbo inference on NVIDIA Blackwell and Ascend 950.
- [2026/08] We released [0.28.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.28.0), featuring production-ready [MiniMax H3](recipes/MiniMaxAI/MiniMax-H3.md) serving on GPU and NPU, a unified AR/DiT paged KV cache runtime, and enhanced realtime full-duplex serving for the [MiniCPM-o series](recipes/OpenBMB/MiniCPM-o-4_5.md).
- [2026/08] [VeRL-Omni](https://github.com/verl-project/verl-omni) `v0.2.0` is released: faster diffusion RL powered by vLLM-Omni (request-level/step-wise batching with FA3), rebuilt Qwen3-Omni multimodal training (DPO & GSPO), plus LTX-2.3, Qwen-Image-Edit support and more. See the [release notes](https://github.com/verl-project/verl-omni/releases/tag/v0.2.0).
- [2026/08] We released [0.26.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.26.0) - aligned with the vLLM 0.26 release line, featuring [MiniMax H3](recipes/MiniMaxAI/MiniMax-H3.md) joint video/audio generation, an experimental full-duplex realtime runtime for [MiniCPM-o 4.5](recipes/OpenBMB/MiniCPM-o-4_5.md), distributed layerwise diffusion offload, and broader model, hardware, streaming, TTS, and quantization support.
- [2026/07] We released [0.24.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.24.0) - aligned with the vLLM 0.24 release line, expanding production-ready coverage across TTS, speech, diffusion, image/video generation, and robot-policy serving, with major Omni stage runtime refactoring, diffusion request-level batching, async output materialization, quantization/cache/memory improvements, and broad CUDA/ROCm/XPU/NPU support.
- [2026/06] Stable releases [0.14.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.14.0), [0.16.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.16.0), [0.18.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.18.0), [0.20.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.20.0), and [0.22.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.22.0) aligned with every even-numbered upstream vLLM minor version, expanding omni and world-model serving, TTS, diffusion, distributed execution, quantization, RL integration through [VeRL-Omni](https://github.com/verl-project/verl-omni), and CUDA/ROCm/MUSA/NPU/XPU support.
- [2026/03] Check out our first public [project deepdive](https://youtu.be/sgwNfsNnR9I) at the vLLM Hong Kong Meetup!
- [2025/11] vLLM community officially released [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) in order to support omni-modality models serving.

---

## About

[vLLM](https://github.com/vllm-project/vllm) was originally designed to support large language models for text-based autoregressive generation tasks. vLLM-Omni is a framework that extends its support for omni-modality model inference and serving:

- **Omni-modality**: Text, image, audio, video, and action data processing
- **Non-autoregressive Architectures**: extend the AR support of vLLM to Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous outputs**: from traditional text generation to multimodal and action outputs

<p align="center">
  <picture>
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/omni-modality-model-architecture.png" width=55%>
  </picture>
</p>

vLLM-Omni is fast with:

- State-of-the-art AR support by leveraging efficient KV cache management from vLLM
- Pipelined stage execution overlapping for high throughput performance
- Full disaggregation based on OmniConnector and dynamic resource allocation across stages

vLLM-Omni is flexible and easy to use with:

- Heterogeneous pipeline abstraction to manage complex model workflows
- Seamless integration with popular Hugging Face models
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- [Full-duplex realtime serving](docs/serving/realtime_duplex_api.md) with streaming audio input and output

vLLM-Omni seamlessly supports most popular open-source models on HuggingFace, including:

- **Omni-modality models** (e.g. Qwen3-Omni, MiniCPM-o 4.5, Cosmos3, HunyuanImage, BAGEL)
- **TTS models** (e.g. Qwen3-TTS, Tencent AuK, Breeze-TTS-2, CosyVoice3)
- **Diffusion models** — image, video, and audio generation (e.g. MiniMax H3, LingBot World, MAGI-2, LTX-2.5, Wan2.2)
- **Robot-policy and action models** (e.g. π0.5, GR00T-N1.7, DreamZero-DROID, InternVLA-A1)

## Getting Started

Visit our [documentation](https://vllm-omni.readthedocs.io/en/latest/) to learn more.

- [Installation](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)
- [Quickstart](https://vllm-omni.readthedocs.io/en/latest/getting_started/quickstart/)
- [Nightly CUDA Docker images](https://hub.docker.com/r/vllm/vllm-omni/tags?name=nightly) built from the latest `main`
- [List of Supported Models](https://vllm-omni.readthedocs.io/en/latest/models/supported_models/)
- [Deployment Recipes](https://recipes.vllm.ai) for vLLM-Omni model serving

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM-Omni](https://vllm-omni.readthedocs.io/en/latest/contributing/) for how to get involved.

## Citation

If you use vLLM-Omni for your research, please cite our [technical report](https://arxiv.org/abs/2610.09307) and [original paper](https://arxiv.org/abs/2602.02204):

```bibtex
@article{vllmomni2026technicalreport,
  title={vLLM-Omni Technical Report: A Unified Serving Runtime for Omni-Modality Generation},
  author={{vLLM-Omni Team}},
  journal={arXiv preprint arXiv:2610.09307},
  year={2026}
}

@article{yin2026vllmomni,
  title={vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models},
  author={Peiqi Yin, Jiangyun Zhu, Han Gao, Chenguang Zheng, Yongxiang Huang, Taichang Zhou, Ruirui Yang, Weizhi Liu, Weiqing Chen, Canlin Guo, Didan Deng, Zifeng Mo, Cong Wang, James Cheng, Roger Wang, Hongsheng Liu},
  journal={arXiv preprint arXiv:2602.02204},
  year={2026}
}
```

## Join the Community

Feel free to ask questions, provide feedbacks and discuss with fellow users of vLLM-Omni in `#sig-omni` slack channel at [slack.vllm.ai](https://slack.vllm.ai) or vLLM user forum at [discuss.vllm.ai](https://discuss.vllm.ai).

## Star History

<a href="https://www.star-history.com/?repos=vllm-project%2Fvllm-omni&type=date&legend=top-left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=vllm-project/vllm-omni&type=date&theme=dark&legend=top-left&sealed_token=ExgLDZJoQEg27Zfhhut2LqN0GYO6Fw2PWLwPE6JYBUp2BgM3hmsYlwaIVopnUEfbRXidQ4nisumrTdKYydiKhy1SZXipw47qY2_tiUDhCpsPXeXtPuEVKVzBwKs3pw0tiHsJgtSfwXx5yjHXck0Y2SblzFWeJYCkTe1WLGTbUAOIETjXXQJjyCGZvKz5" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=vllm-project/vllm-omni&type=date&legend=top-left&sealed_token=ExgLDZJoQEg27Zfhhut2LqN0GYO6Fw2PWLwPE6JYBUp2BgM3hmsYlwaIVopnUEfbRXidQ4nisumrTdKYydiKhy1SZXipw47qY2_tiUDhCpsPXeXtPuEVKVzBwKs3pw0tiHsJgtSfwXx5yjHXck0Y2SblzFWeJYCkTe1WLGTbUAOIETjXXQJjyCGZvKz5" />
    <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=vllm-project/vllm-omni&type=date&legend=top-left&sealed_token=ExgLDZJoQEg27Zfhhut2LqN0GYO6Fw2PWLwPE6JYBUp2BgM3hmsYlwaIVopnUEfbRXidQ4nisumrTdKYydiKhy1SZXipw47qY2_tiUDhCpsPXeXtPuEVKVzBwKs3pw0tiHsJgtSfwXx5yjHXck0Y2SblzFWeJYCkTe1WLGTbUAOIETjXXQJjyCGZvKz5" />
  </picture>
</a>

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
