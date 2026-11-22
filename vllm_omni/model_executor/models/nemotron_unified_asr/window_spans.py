# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Combine ready hops from one stream and map logits back to their original windows."""


def _aligned_flush(rows, members, chunk):
    # Subsampling advances one encoder frame per 1,280 input samples. A
    # partial final append can shift the native crop off that grid.
    first, last = rows[members[0]], rows[members[-1]]
    shift = first["asr_keep"] + sum(rows[i]["asr_pcm"].numel() for i in members[:-1]) - last["asr_keep"]
    return shift + 1280 * last["asr_left"] == 1280 * (first["asr_left"] + (len(members) - 1) * chunk)


def pack_ready_windows(rows, limit=24, chunk=7):
    by_stream = {}
    for i, row in enumerate(rows):
        by_stream.setdefault(row["asr_stream"], []).append(i)
    spans = []
    for members in by_stream.values():
        start = 0
        while start < len(members):
            part = members[start : start + limit]
            if len(part) > 1 and rows[part[-1]]["asr_final"] and not _aligned_flush(rows, part, chunk):
                part = part[:-1]
            spans.append(part)
            start += len(part)
    spans.sort(key=lambda part: part[0])
    parents, offsets = [0] * len(rows), [0] * len(rows)
    for index, part in enumerate(spans):
        for position, i in enumerate(part):
            parents[i] = index
            offsets[i] = rows[part[0]]["asr_left"] + position * chunk
    return [part[0] for part in spans], [part[-1] for part in spans], parents, offsets
