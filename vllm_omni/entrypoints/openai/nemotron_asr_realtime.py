# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Buffered, language-prompted Nemotron transcription over 16 kHz mono PCM16."""

import asyncio
import uuid
from collections import deque
from pathlib import Path

import pybase64 as base64
from fastapi import WebSocket, WebSocketDisconnect
from sentencepiece import SentencePieceProcessor

from .nemotron_window_batcher import NemotronWindowBatcher

LEFT = 89600


class NemotronTranscription:
    chunk_samples = 7 * 1280
    right_samples = 4 * 1280
    window_backlog = 32

    def __init__(self, engine, config):
        self.slots = asyncio.Semaphore(32)
        self.engine = engine
        self.batcher = NemotronWindowBatcher(self.submit, packed=config.hf_config.packed_streaming)
        self.languages = config.hf_config.asr_config["model_defaults"]["prompt_dictionary"]
        self.tokenizer = SentencePieceProcessor(model_file=str(Path(config.model) / "tokenizer.model"))

    async def submit(self, info):
        # This path is for the dedicated, single-rank ASR stage. Broadcasting
        # stateful batches across replicas would make session placement ambiguous.
        pools = self.engine.engine.stage_pools
        if len(pools) != 1 or pools[0].live_num_replicas != 1:
            raise RuntimeError("Stage audio RPC requires a single dedicated ASR replica")
        result = await self.engine.collective_rpc("process_realtime_audio_batch", args=(info,), stage_ids=[0])
        while isinstance(result, (tuple, list)) and len(result) == 1:
            result = result[0]
        if not isinstance(result, dict) or "parts" not in result:
            raise RuntimeError(f"Unexpected streaming ASR stage response: {str(result)[:200]}")
        return {"asr_parts": result["parts"]}

    async def window(self, info):
        if "asr_abort" in info:
            await self.submit(info)
            return []
        return await self.batcher.window(info)

    async def handle(self, ws: WebSocket):
        async with self.slots:
            await self._session(ws)

    async def _session(self, ws: WebSocket):
        await ws.accept()
        stream = uuid.uuid4().hex
        lang = self.languages["en"]
        queued = bytearray()
        window_samples = 0
        tokens = []
        previous_text = ""
        active = False
        outstanding = deque()
        await ws.send_json({"type": "session.created", "session": {"id": stream}})

        async def decode(nbytes, final):
            nonlocal active, window_samples
            prior = window_samples
            total = prior + nbytes // 2
            window_samples = min(total, LEFT + self.chunk_samples + self.right_samples)
            excess = total - window_samples
            left = max(0, prior - self.right_samples - excess)
            pcm = bytes(memoryview(queued)[:nbytes])
            del queued[:nbytes]

            active = True
            info = dict(
                asr_stream=stream,
                asr_lang=lang,
                asr_final=final,
                asr_left=left // 1280,
                asr_keep=prior - excess,
                asr_pcm=pcm,
            )
            # Only already-arrived audio can enter the queue. Independent acoustic windows
            # may run together; RNNT state and client-visible partials remain in stream order.
            prior_task = outstanding[-1] if outstanding else None
            outstanding.append(asyncio.create_task(emit(info, prior_task)))
            if final:
                await asyncio.gather(*outstanding)
                outstanding.clear()
            elif len(outstanding) >= self.window_backlog:
                await outstanding.popleft()

        async def emit(info, prior_task):
            nonlocal active, previous_text
            ids = await self.window(info)
            if prior_task is not None:
                await prior_task
            text = previous_text
            if ids:
                tokens.extend(ids)
                text = self.tokenizer.decode(tokens)
            if text != previous_text:
                await ws.send_json(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "transcript": text,
                        "delta": text[len(previous_text) :] if text.startswith(previous_text) else text,
                    }
                )
                previous_text = text
            if info["asr_final"]:
                active = False
                await ws.send_json(
                    {"type": "conversation.item.input_audio_transcription.completed", "transcript": text}
                )

        try:
            while True:
                msg = await ws.receive_json()
                if not isinstance(msg, dict):
                    raise ValueError("Expected a JSON object")
                kind = msg.get("type")
                if kind == "session.update":
                    session = msg.get("session", {})
                    options = session.get("input_audio_transcription", {}) if isinstance(session, dict) else None
                    if not isinstance(options, dict):
                        raise ValueError("Invalid transcription session")
                    name = options.get("language", "en")
                    if not isinstance(name, str) or name not in self.languages or active or queued:
                        raise ValueError("Choose a supported language before sending audio")
                    lang = self.languages[name]
                    await ws.send_json({"type": "session.updated"})
                elif kind == "input_audio_buffer.append":
                    encoded = msg.get("audio")
                    if not isinstance(encoded, str) or len(encoded) > 426668:
                        raise ValueError("Expected base64-encoded PCM16")
                    pcm = base64.b64decode(encoded, validate=True)
                    if len(pcm) % 2 or len(pcm) > 320000:
                        raise ValueError("Expected at most ten seconds of 16 kHz mono PCM16")
                    queued.extend(pcm)
                    while len(queued) >= (
                        n := 2 * (self.chunk_samples if window_samples else self.chunk_samples + self.right_samples)
                    ):
                        await decode(n, False)
                elif kind == "input_audio_buffer.commit":
                    if window_samples or len(queued) >= 2 * 320:
                        await decode(len(queued), True)
                    else:
                        await ws.send_json(
                            {"type": "conversation.item.input_audio_transcription.completed", "transcript": ""}
                        )
                    stream = uuid.uuid4().hex
                    queued.clear()
                    window_samples = 0
                    tokens.clear()
                    previous_text = ""
                else:
                    raise ValueError("Unknown transcription event")
        except WebSocketDisconnect:
            pass
        except (ValueError, KeyError) as exc:
            await ws.send_json({"type": "error", "error": str(exc)})
            await ws.close(code=1008)
        finally:
            if outstanding:
                await asyncio.gather(*outstanding, return_exceptions=True)
            if active:
                await self.window({"asr_abort": stream})
