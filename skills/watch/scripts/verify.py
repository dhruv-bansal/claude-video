#!/usr/bin/env python3
"""Verify a Whisper transcript for the decoder's silent failure modes.

Whisper returns fluent, complete-looking text even when it has failed:

* Repetition loop: the model conditions each 30s window on its own previous
  output; on a low-confidence patch it latches onto what it just emitted and
  repeats it for minutes while the real speech in that window is discarded.
  Observed in the wild: 11 minutes of loud, clear speech replaced by "Thank
  you." repeated.
* Truncated window: the decoder hits its per-window token cap and stops
  mid-window. The rest of that window's speech is simply gone. Observed on a
  Hindi reel: one segment 0-30s whose text ends in U+FFFD (a Devanagari
  character cut in half), with the speech from ~16-30s missing.
* Early stop: decoding ends well before the audio does.

None of these raise. These checks are the only way a reader finds out. Pure
functions over the {start, end, text} segment list every backend produces.
"""
from __future__ import annotations

import sys

# A run of this many consecutive identical segments is the loop signature;
# real speech never repeats a cue verbatim three times in a row.
LOOP_MIN_RUN = 3
# A transcript whose last cue ends more than this many seconds before the
# audio does stopped early.
TAIL_LIMIT_SECONDS = 30.0
# Whisper decodes 30s windows and closes each segment with a timestamp token.
# A decoder that runs into its per-window token cap (224 sampled tokens in
# mlx-whisper; byte-level BPE spends ~2-3 tokens per Devanagari/CJK character)
# never emits that closing timestamp, so the segment is stretched to the
# window edge: one cue spanning the whole 30s. Real speech cues are a few
# seconds to ~15s. 29s (not 30) leaves slack for the window's seek rounding.
# Only meaningful for mlx's sequential decode: WhisperX (--no_align) reports
# merged VAD chunks of up to 30s as segments, and the APIs may also return long
# cues, so for those backends only the U+FFFD ending is evidence.
TRUNCATED_MIN_SECONDS = 29.0
# U+FFFD is what a UTF-8 decode produces for a character whose bytes were cut
# off: the cap landed mid-character. Never a legitimate transcript ending.
REPLACEMENT_CHAR = "�"

CLEAN = {"ok": True, "loops": [], "truncated": [], "ends_early_seconds": 0.0}


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


def find_truncated(segments: list[dict], min_seconds: float | None = TRUNCATED_MIN_SECONDS) -> list[dict]:
    """Segments that look cut off by the decoder's token cap.

    A segment qualifies when its text ends in U+FFFD, or (unless `min_seconds`
    is None) when it spans at least `min_seconds`: a whole decode window with
    no closing timestamp. Returns {start, end, text} per segment, in source
    seconds.
    """
    out: list[dict] = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        text = str(seg.get("text", ""))
        long_window = min_seconds is not None and end - start >= min_seconds
        if text.rstrip().endswith(REPLACEMENT_CHAR) or long_window:
            out.append({"start": start, "end": end, "text": text})
    return out


def verify_segments(
    segments: list[dict],
    duration_seconds: float,
    min_run: int = LOOP_MIN_RUN,
    tail_limit: float = TAIL_LIMIT_SECONDS,
    *,
    check_window_length: bool = False,
) -> dict:
    """Scan a transcript. Returns {"ok", "loops", "truncated", "ends_early_seconds"}.

    `ok` is False for repetition loops and truncated windows: in both, speech
    was dropped. `ends_early_seconds` is advisory: Whisper emits nothing for
    outro music, end cards or silence, so a transcript that stops early usually
    means "nobody spoke", and the reader should check the frames rather than
    assume loss. An unknown duration (0), or the text-only {0, 0} fallback
    segment, skips the tail check. `check_window_length` adds the whole-window
    duration rule (TRUNCATED_MIN_SECONDS); pass it only for mlx.
    """
    loops = find_repetition_runs(segments, min_run=min_run)
    truncated = find_truncated(segments, TRUNCATED_MIN_SECONDS if check_window_length else None)
    ends_early = 0.0
    if segments and duration_seconds > 0:
        last_end = max(float(seg["end"]) for seg in segments)
        missing = duration_seconds - last_end
        if last_end > 0 and missing > tail_limit:
            ends_early = round(missing, 1)
    return {
        "ok": not loops and not truncated,
        "loops": loops,
        "truncated": truncated,
        "ends_early_seconds": ends_early,
    }


def filter_quality_range(quality: dict, lo: float | None, hi: float | None) -> dict:
    """Keep only problems overlapping [lo, hi] (a --start/--end focus window)."""
    if lo is None and hi is None:
        return quality
    lo_v = lo if lo is not None else float("-inf")
    hi_v = hi if hi is not None else float("inf")

    def keep(items):
        return [item for item in items if item["end"] >= lo_v and item["start"] <= hi_v]

    loops = keep(quality.get("loops", []))
    truncated = keep(quality.get("truncated", []))
    return {**quality, "ok": not loops and not truncated, "loops": loops, "truncated": truncated}


def splice_segments(
    segments: list[dict],
    lo: float,
    hi: float,
    replacement: list[dict],
) -> list[dict]:
    """Replace every segment overlapping [lo, hi] with `replacement`.

    Overlap, not start-inside: the window is padded, so a good segment that
    starts before `lo` but runs into it would otherwise be kept *and* have its
    tail re-emitted by the retry. `replacement` must already be in source
    time. The result stays chronological.
    """
    kept = [seg for seg in segments if float(seg["end"]) <= lo or float(seg["start"]) >= hi]
    merged = kept + list(replacement)
    merged.sort(key=lambda seg: (float(seg["start"]), float(seg["end"])))
    return clamp_ends(merged)


def clamp_ends(segments: list[dict]) -> list[dict]:
    """Copies of chronological segments with each end clamped to the next
    segment's start. A repaired slice's last word can end a few ms past the
    first kept word of its neighbour (observed: 20.05 vs 19.99)."""
    out = [dict(seg) for seg in segments]
    for seg, nxt in zip(out, out[1:]):
        if float(seg["end"]) > float(nxt["start"]):
            seg["end"] = max(float(seg["start"]), float(nxt["start"]))
    return out


def report_quality(quality: dict, backend: str) -> None:
    """One stderr line per remaining problem, for the run log."""
    for loop in quality.get("loops", []):
        print(
            f"[watch] WARNING: transcript repetition loop {loop['start']:.0f}s-{loop['end']:.0f}s "
            f"(×{loop['count']} {loop['text'][:40]!r}) — speech in that window may be lost",
            file=sys.stderr,
        )
    for seg in quality.get("truncated", []):
        print(
            f"[watch] WARNING: transcript segment {seg['start']:.0f}s-{seg['end']:.0f}s looks truncated "
            f"({backend} hit its per-window limit) — speech in that window may be lost",
            file=sys.stderr,
        )
    if quality.get("ends_early_seconds"):
        print(
            f"[watch] note: no speech recognised in the last {quality['ends_early_seconds']:.0f}s "
            f"({backend}) — silence/music, or an early stop",
            file=sys.stderr,
        )


def quality_warning_lines(quality: dict, format_time) -> list[str]:
    """Markdown blockquote lines for the report, or [] when the transcript is
    clean. `format_time` renders seconds as the report's `MM:SS`."""
    lines: list[str] = []
    loops = quality.get("loops", [])
    truncated = quality.get("truncated", [])
    if loops or truncated:
        lines += [
            "",
            "> **Warning: transcript verification failed.** Whisper's failures are silent — "
            "the text reads fluently but speech in the flagged windows was lost. Tell the user, "
            "and do not draw conclusions about those windows from the transcript alone:",
        ]
        for loop in loops:
            lines.append(
                f">   - repetition loop {format_time(loop['start'])} → {format_time(loop['end'])} "
                f"(×{loop['count']} \"{loop['text'][:50]}\")"
            )
        for seg in truncated:
            lines.append(
                f">   - truncated segment {format_time(seg['start'])} → {format_time(seg['end'])} "
                "(decoder stopped mid-window; later speech in it is missing)"
            )
    if quality.get("ends_early_seconds"):
        lines += [
            "",
            f"> **Note:** no speech was recognised in the last {quality['ends_early_seconds']:.0f}s "
            "of the audio. Usually outro music or silence; check the frames for that span before "
            "assuming the transcript is complete.",
        ]
    return lines
