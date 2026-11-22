# Stream speech to text with Nemotron Unified ASR

Serve NVIDIA’s language-prompted, greedy RNNT on one BF16 CUDA GPU. Each connection holds its own waveform history, language and decoder state. The worker admits 32 concurrent transcription sessions. It can combine acoustic windows when several packets have already arrived. A live stream advances in seven-frame steps with four frames of right context.

## Prepare the checkpoint

The adapter uses the ASR components from NVIDIA NeMo Speech revision `c02239ab43822ea837b21fb5c0c29e45fbd063fd`. Install that revision with its ASR dependencies in the vLLM-Omni environment. Convert the official archive once into a new local directory. The conversion retains the original tensor values and SentencePiece vocabulary.

```bash
pip install 'nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@c02239ab43822ea837b21fb5c0c29e45fbd063fd'
hf download nvidia/Nemotron-Labs-Unified-ASR-0.6B nemotron-labs-unified-asr-0.6b.nemo \
    --revision 023ce907267a09a308a5cb69239447cb3d0c1597 --local-dir checkpoints/nemotron-release
python tools/prepare_nemotron_asr.py checkpoints/nemotron-release/nemotron-labs-unified-asr-0.6b.nemo checkpoints/nemotron
vllm serve checkpoints/nemotron --omni --served-model-name nemotron --trust-remote-code \
    --deploy-config vllm_omni/deploy/nemotron_unified_asr.yaml
```

The NeMo model constructs components from the trusted checkpoint configuration. For a same-context accuracy or performance control, convert the archive into a second directory with `--reference`. It uses the released NeMo Conformer layers and greedy decoder with the same audio transport, frame cadence and decoder-frame assembly.

## Transcribe a stream

Connect to `ws://localhost:8000/v1/realtime` and wait for `session.created`. English is the default. To select another checkpoint language before sending audio, send a session update, for example:

```json
{"type":"session.update","session":{"input_audio_transcription":{"language":"en"}}}
```

Send `input_audio_buffer.append` events with a base64 `audio` field containing mono 16 kHz little-endian signed PCM16. Each append accepts up to ten seconds. Send `{"type":"input_audio_buffer.commit"}` at the end of each utterance. Intermediate `conversation.item.input_audio_transcription.delta` events include the full transcript so far. The `conversation.item.input_audio_transcription.completed` event contains the final transcript. The same socket can start a new utterance after completion.

Validation covers English speech on a single RTX 4090. Each server runs one API process, one stage replica and one GPU rank.

## Measure streaming throughput and latency

The load generator reads a JSONL manifest with `audio_filepath`, `text` and `duration` in seconds for each recording. Use identical manifests and client settings for both servers. Install `jiwer`, `websockets`, `soundfile` and `numpy` on the load-generator host. The client sends 160 ms packets. A burst run measures how quickly the server processes queued audio. A paced run sends packets in real time and measures time to the first nonempty partial and from commit to final.

```bash
python benchmarks/asr/benchmark_realtime_transcription.py --manifest clips.jsonl --limit 32 \
    --workers 4 --compression none --concurrency 32 16 8 --output warm
python benchmarks/asr/benchmark_realtime_transcription.py --manifest clips.jsonl --limit 256 \
    --workers 4 --compression none --concurrency 32 16 8 --output burst
python benchmarks/asr/benchmark_realtime_transcription.py --manifest clips.jsonl --limit 32 \
    --workers 4 --compression none --concurrency 32 --mode paced --output paced-warm
python benchmarks/asr/benchmark_realtime_transcription.py --manifest clips.jsonl --limit 32 \
    --workers 4 --compression none --concurrency 32 16 8 --mode paced --output paced
```
