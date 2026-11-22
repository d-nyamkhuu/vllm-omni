# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import WebSocketDisconnect

from vllm_omni.entrypoints.openai.nemotron_asr_realtime import LEFT, NemotronTranscription

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def exercise(events, replies=None):
    async def run():
        ws = SimpleNamespace(
            accept=AsyncMock(),
            send_json=AsyncMock(),
            receive_json=AsyncMock(side_effect=[*events, WebSocketDisconnect()]),
        )
        handler = object.__new__(NemotronTranscription)
        handler.slots = asyncio.Semaphore(32)
        handler.languages = {"en": 0}
        handler.tokenizer = SimpleNamespace(decode=lambda tokens: "word" if tokens else "")
        handler.window = AsyncMock(return_value=[], side_effect=replies)
        await handler.handle(ws)
        return [c.args[0] for c in handler.window.await_args_list], [c.args[0] for c in ws.send_json.await_args_list]

    return asyncio.run(run())


def packet(pcm):
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm.tobytes()).decode()}


def test_unaligned_appends_and_repeated_commits():
    chunk, right = 7 * 1280, 4 * 1280
    pcm = np.arange(LEFT + 3 * chunk + 777, dtype=np.int16)
    packets = [packet(pcm[i : i + 999]) for i in range(0, len(pcm), 999)]
    commit = {"type": "input_audio_buffer.commit"}
    rows, events = exercise([*packets, commit, *packets, commit])
    middle = len(rows) // 2
    assert rows[0]["asr_stream"] != rows[middle]["asr_stream"]
    for utterance in (rows[:middle], rows[middle:]):
        assert len({r["asr_stream"] for r in utterance}) == 1
        assert [r["asr_final"] for r in utterance] == [False] * (len(utterance) - 1) + [True]
        assert b"".join(r["asr_pcm"] for r in utterance) == pcm.tobytes()
        total = 0
        for row in utterance:
            n = len(row["asr_pcm"]) // 2
            keep = min(total, LEFT + chunk + right - n)
            assert row["asr_keep"] == keep
            assert row["asr_left"] == max(0, keep - right) // 1280
            total += n
    assert sum(e["type"].endswith(".completed") for e in events) == 2


def test_disconnect_releases_inflight_stream():
    rows, _ = exercise([packet(np.zeros(11 * 1280, dtype=np.int16))])
    assert rows[-1] == {"asr_abort": rows[0]["asr_stream"]}


def test_sub_frame_commit_returns_empty_transcript():
    rows, events = exercise([packet(np.zeros(159, dtype=np.int16)), {"type": "input_audio_buffer.commit"}])
    assert not rows
    assert events[-1] == {"type": "conversation.item.input_audio_transcription.completed", "transcript": ""}


def test_stage_batch_round_trip_uses_unpadded_pcm():
    from vllm_omni.entrypoints.openai.nemotron_window_batcher import NemotronWindowBatcher

    async def execute():
        handler = object.__new__(NemotronTranscription)

        async def rpc(method, *, args, stage_ids):
            assert method == "process_realtime_audio_batch" and stage_ids == [0]
            info = args[0]
            assert "asr_pcm" not in info
            assert [len(b) for b in info["pcm"]] == [6, 0, 2]
            return [[{"parts": [[3, 4], [], [9]]}]]

        handler.engine = SimpleNamespace(
            engine=SimpleNamespace(stage_pools=[SimpleNamespace(live_num_replicas=1)]), collective_rpc=rpc
        )
        batcher = NemotronWindowBatcher(handler.submit)
        rows = [
            dict(
                asr_stream=str(i),
                asr_left=0,
                asr_lang=0,
                asr_final=False,
                asr_keep=0,
                asr_pcm=bytes(2 * n),
            )
            for i, n in enumerate([3, 0, 1])
        ]
        return await asyncio.gather(*(batcher.window(row) for row in rows))

    assert asyncio.run(execute()) == [[3, 4], [], [9]]


def test_span_batcher_bounds_acoustic_work_and_preserves_stream_order():
    from collections import defaultdict

    from vllm_omni.entrypoints.openai.nemotron_window_batcher import NemotronWindowBatcher

    batcher = NemotronWindowBatcher(None)
    for hop in range(18):
        for stream in range(48):
            info = dict(
                asr_stream=stream,
                asr_keep=min(hop * 8960, 89600),
                asr_pcm=bytes(2 * (8960 if hop != 17 else 517)),
                asr_final=hop == 17,
            )
            batcher.pending.append((info, hop))
    seen = defaultdict(list)
    sizes = []
    while batcher.pending:
        items = batcher._take_span_batch()
        sizes.append(len(items))
        distinct = set()
        groups = work = 0
        for info, hop in items:
            s = info["asr_stream"]
            if s not in distinct or (info["asr_final"] and seen[s] and seen[s][-1] == hop - 1):
                groups += 1
                work += info["asr_keep"]
            work += len(info["asr_pcm"]) // 2
            distinct.add(s)
            seen[s].append(hop)
        assert len(items) <= 512 and groups <= 32 and work <= 4096 * 1280
    assert max(sizes) > 32 and set(seen) == set(range(48))
    assert all(order == list(range(18)) for order in seen.values())
