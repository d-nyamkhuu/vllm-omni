# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drive the same closed-loop WebSocket workload from several independent CPU processes."""

import argparse
import asyncio
import base64
import hashlib
import json
import multiprocessing
import re
import traceback
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import soundfile as sf
import websockets
from jiwer import wer


async def stream(uri, case, paced, compression):
    pcm, sr = sf.read(case["audio_filepath"], dtype="int16")
    assert sr == 16000 and pcm.ndim == 1
    async with websockets.connect(
        uri, open_timeout=30, close_timeout=5, ping_timeout=None, compression=compression
    ) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "session.created", hello
        start = perf_counter()

        async def send():
            for offset in range(0, len(pcm), 2560):
                part = pcm[offset : offset + 2560]
                if paced:
                    await asyncio.sleep(max(0, start + (offset + len(part)) / sr - perf_counter()))
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(part.astype("<i2").tobytes()).decode(),
                        }
                    )
                )
            commit_start = perf_counter() - start
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            return commit_start

        sender = asyncio.create_task(send())
        first = None
        partials = 0
        partial_hash = hashlib.sha256()
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
                elapsed = perf_counter() - start
                if msg["type"].endswith(".delta"):
                    if first is None and msg["transcript"].strip():
                        first = elapsed
                    partials += 1
                    encoded = msg["transcript"].encode("utf-8")
                    partial_hash.update(len(encoded).to_bytes(4, "big") + encoded)
                elif msg["type"].endswith(".completed"):
                    commit_start = await sender
                    return dict(
                        file=case["audio_filepath"],
                        reference=case["text"],
                        duration=len(pcm) / sr,
                        transcript=msg["transcript"],
                        first_partial_s=first,
                        final_s=elapsed,
                        commit_to_final_ms=(elapsed - commit_start) * 1000,
                        partials=partials,
                        partial_transcripts_sha256=partial_hash.hexdigest(),
                    )
                elif msg["type"] == "error":
                    raise RuntimeError(msg)
        finally:
            if not sender.done():
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)


def summarize(rows, concurrency, mode, elapsed):
    def normalized(key):
        return [" ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", row[key].lower())) for row in rows]

    partials = [r["first_partial_s"] for r in rows if r["first_partial_s"] is not None]
    return dict(
        concurrency=concurrency,
        mode=mode,
        wall_s=elapsed,
        streams=len(rows),
        audio_seconds_per_s=sum(r["duration"] for r in rows) / elapsed,
        p95_first_nonempty_partial_ms=float(np.percentile(partials, 95) * 1000) if partials else None,
        streams_without_partial=len(rows) - len(partials),
        p95_final_s=float(np.percentile([r["final_s"] for r in rows], 95)),
        p95_commit_to_final_ms=float(np.percentile([r["commit_to_final_ms"] for r in rows], 95)),
        wer=wer(normalized("reference"), normalized("transcript")),
        rows=rows,
    )


def client(queue, gate, pipe, slots, cases, uri, paced, compression):
    rows = []

    async def lane():
        while (i := queue.get()) is not None:
            rows.append((i, await stream(uri, cases[i], paced, compression)))

    async def run():
        await asyncio.gather(*(lane() for _ in range(slots)))

    try:
        pipe.send(("ready", None))
        gate.wait()
        cpu = process_time()
        asyncio.run(run())
        pipe.send(("finished", (perf_counter(), process_time() - cpu, rows)))
    except Exception:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()


def receive(pipe, expected, timeout):
    if not pipe.poll(timeout):
        raise TimeoutError(f"Client process timed out before {expected}")
    event, payload = pipe.recv()
    if event != expected:
        raise RuntimeError(f"Unexpected client event {event}: {payload}")
    return payload


def run(cases, uri, c, workers, mode, compression):
    ctx = multiprocessing.get_context("spawn")
    gate, queue = ctx.Event(), ctx.Queue()
    count = min(workers, c, len(cases))
    processes, pipes = [], []
    try:
        for i in range(len(cases)):
            queue.put(i)
        for _ in range(c):
            queue.put(None)
        for i in range(count):
            rx, tx = ctx.Pipe(duplex=False)
            slots = c // count + int(i < c % count)
            proc = ctx.Process(target=client, args=(queue, gate, tx, slots, cases, uri, mode == "paced", compression))
            proc.start()
            tx.close()
            processes.append(proc)
            pipes.append(rx)
        for rx in pipes:
            receive(rx, "ready", 60)
        start = perf_counter()
        gate.set()
        timeout = 180 + sum(case["duration"] for case in cases)
        parts = [receive(rx, "finished", timeout) for rx in pipes]
        elapsed = max(end for end, _, _ in parts) - start
        indexed = sorted((pair for _, _, rows in parts for pair in rows), key=lambda p: p[0])
        if [i for i, _ in indexed] != list(range(len(cases))):
            raise ValueError("Incomplete or repeated input indices")
        result = summarize([row for _, row in indexed], c, mode, elapsed)
        result.update(
            client_workers=count,
            compression=compression or "none",
            client_cpu_s=sum(cpu for _, cpu, _ in parts),
            requests_per_worker=[len(rows) for _, _, rows in parts],
        )
        return result
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
            proc.join()
        for rx in pipes:
            rx.close()
        queue.cancel_join_thread()
        queue.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uri", default="ws://127.0.0.1:8000/v1/realtime")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--concurrency", type=int, nargs="+", default=(1, 8, 16, 32))
    p.add_argument("--mode", choices=("burst", "paced"), default="burst")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--compression", choices=("none", "deflate"), default="none")
    p.add_argument("--limit", type=int, default=96)
    args = p.parse_args()
    cases = [json.loads(line) for line in args.manifest.read_text().splitlines()][: args.limit]
    if not cases or min(args.workers, *args.concurrency) < 1:
        p.error("A nonempty manifest, positive worker count and positive concurrency are required")
    args.output.mkdir(parents=True, exist_ok=True)
    for c in args.concurrency:
        result = run(cases, args.uri, c, args.workers, args.mode, None if args.compression == "none" else "deflate")
        path = args.output / f"{args.mode}-c{c}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2))
        temp.replace(path)
        print({k: v for k, v in result.items() if k != "rows"}, flush=True)
