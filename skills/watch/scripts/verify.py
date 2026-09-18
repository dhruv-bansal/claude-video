#!/usr/bin/env python3
"""Verify a Whisper transcript for the decoder's silent failure modes.

Whisper returns fluent, complete-looking text even when it has failed:

* Repetition loop — the model conditions each 30s window on its own previous
  output; on a low-confidence patch it latches onto what it just emitted and
  repeats it for minutes while the real speech in that window is discarded.
  Observed in the wild: 11 minutes of loud, clear speech replaced by "Thank
  you." repeated.
* Early stop — decoding ends well before the audio does.

Neither raises. These checks are the only way a reader finds out. Pure
functions over the {start, end, text} segment list every backend produces.
"""
from __future__ import annotations

import sys

# A run of this many consecutive identical segments is the loop signature —
# real speech never repeats a cue verbatim three times in a row.
LOOP_MIN_RUN = 3
# A transcript whose last cue ends more than this many seconds before the
# audio does stopped early.
TAIL_LIMIT_SECONDS = 30.0


def find_repetition_runs(segments: list[dict], min_run: int = LOOP_MIN_RUN) -> list[dict]:
    """Locate runs of consecutive segments with identical (normalized) text.

    Returns one dict per run: {start, end, count, text} in source seconds.
    """
    runs: list[dict] = []
    if not segments:
        return runs

    def _norm(text: str) -> str:
        return " ".join(text.split()).strip().lower()

    run_start = 0
    for i in range(1, len(segments) + 1):
        same = i < len(segments) and _norm(segments[i]["text"]) == _norm(segments[run_start]["text"])
        if same:
            continue
        count = i - run_start
        if count >= min_run:
            runs.append({
                "start": float(segments[run_start]["start"]),
                "end": float(segments[i - 1]["end"]),
                "count": count,
                "text": segments[run_start]["text"],
            })
        run_start = i
    return runs


def verify_segments(
    segments: list[dict],
    duration_seconds: float,
    min_run: int = LOOP_MIN_RUN,
    tail_limit: float = TAIL_LIMIT_SECONDS,
) -> dict:
    """Scan a transcript. Returns {"ok", "loops", "ends_early_seconds"}.

    `ok` is False only for repetition loops — those are definite: speech was
    replaced. `ends_early_seconds` is advisory: Whisper emits nothing for outro
    music, end cards or silence, so a transcript that stops early usually means
    "nobody spoke", and the reader should check the frames rather than assume
    loss. An unknown duration (0), or the text-only {0, 0} fallback segment,
    skips the tail check.
    """
    loops = find_repetition_runs(segments, min_run=min_run)
    ends_early = 0.0
    if segments and duration_seconds > 0:
        last_end = max(float(seg["end"]) for seg in segments)
        missing = duration_seconds - last_end
        if last_end > 0 and missing > tail_limit:
            ends_early = round(missing, 1)
    return {
        "ok": not loops,
        "loops": loops,
        "ends_early_seconds": ends_early,
    }


def filter_quality_range(quality: dict, lo: float | None, hi: float | None) -> dict:
    """Keep only loops overlapping [lo, hi] (a --start/--end focus window)."""
    if lo is None and hi is None:
        return quality
    lo_v = lo if lo is not None else float("-inf")
    hi_v = hi if hi is not None else float("inf")
    loops = [l for l in quality["loops"] if l["end"] >= lo_v and l["start"] <= hi_v]
    return {**quality, "ok": not loops, "loops": loops}


def splice_segments(
    segments: list[dict],
    lo: float,
    hi: float,
    replacement: list[dict],
) -> list[dict]:
    """Replace every segment starting inside [lo, hi] with `replacement`.

    `replacement` must already be in source time. The result stays
    chronological.
    """
    kept = [seg for seg in segments if not (lo <= float(seg["start"]) <= hi)]
    merged = kept + list(replacement)
    merged.sort(key=lambda seg: (float(seg["start"]), float(seg["end"])))
    return merged


def report_quality(quality: dict, backend: str) -> None:
    """One stderr line per remaining problem, for the run log."""
    for loop in quality["loops"]:
        print(
            f"[watch] WARNING: transcript repetition loop {loop['start']:.0f}s-{loop['end']:.0f}s "
            f"(×{loop['count']} {loop['text'][:40]!r}) — speech in that window may be lost",
            file=sys.stderr,
        )
    if quality["ends_early_seconds"]:
        print(
            f"[watch] note: no speech recognised in the last {quality['ends_early_seconds']:.0f}s "
            f"({backend}) — silence/music, or an early stop",
            file=sys.stderr,
        )


def quality_warning_lines(quality: dict, format_time) -> list[str]:
    """Markdown blockquote lines for the report, or [] when the transcript is
    clean. `format_time` renders seconds as the report's `MM:SS`."""
    lines: list[str] = []
    if quality.get("loops"):
        lines += [
            "",
            "> **Warning: transcript verification failed.** Whisper's failures are silent — "
            "the text reads fluently but speech in the flagged windows was lost. Tell the user, "
            "and do not draw conclusions about those windows from the transcript alone:",
        ]
        for loop in quality["loops"]:
            lines.append(
                f">   - repetition loop {format_time(loop['start'])} → {format_time(loop['end'])} "
                f"(×{loop['count']} \"{loop['text'][:50]}\")"
            )
    if quality.get("ends_early_seconds"):
        lines += [
            "",
            f"> **Note:** no speech was recognised in the last {quality['ends_early_seconds']:.0f}s "
            "of the audio. Usually outro music or silence; check the frames for that span before "
            "assuming the transcript is complete.",
        ]
    return lines
