#!/usr/bin/env python3
"""Local Whisper backend: the `mlx_whisper` CLI on the Apple Silicon GPU.

Nothing leaves the machine, no API key, no per-minute cost. The model
(~1.5 GB) is fetched from Hugging Face on first use into
~/.cache/huggingface/hub/. Install the CLI with `pipx install mlx-whisper`.

Pure stdlib here; mlx itself lives in the CLI's own environment. Segments
come back in the same {start, end, text} shape as the API backends, so
whisper.py treats this as just another source.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

from runtime import diagnostic, run_text
from verify import splice_segments

MLX_BIN = "mlx_whisper"
MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
# `pipx install` lands here; it is only on PATH after `pipx ensurepath` and a
# new shell, so look there explicitly rather than fail on a fresh machine.
PIPX_BIN_DIR = Path.home() / ".local" / "bin"
# Whisper conditions each 30s window on its own previous output. On a
# low-confidence patch it can latch onto what it just emitted and repeat it for
# minutes while discarding the real speech: fluent, complete-looking, wrong.
# Disabling the conditioning stops the loop feeding itself; a little
# temperature lets the decoder escape a greedy repetition already underway.
MLX_TEMPERATURE = 0.2
MLX_RETRY_TEMPERATURE = 0.4
# Padding around a looped window when re-transcribing it in isolation.
LOOP_RETRY_PAD_SECONDS = 5.0
# Each repair re-runs the decoder (a truncated window costs ~11 s, a loop
# window more), so a pathological transcript must not turn one run into many
# minutes. Windows past the cap stay in the verify result and are reported.
MAX_REPAIR_WINDOWS = 10
# A truncated window ran out of decoder tokens before it ran out of speech.
# Re-decoding the same 30s reproduces the cut; shorter slices each carry less
# speech than the cap. 10s slices hold well under the cap even for dense
# Devanagari (15s halves already recovered the observed Hindi case).
TRUNCATED_SLICE_SECONDS = 10.0
# A hard cut through a word garbles it in both slices ("…25, 20." / "26…").
# Cuts are first moved to the nearest pause within SNAP_SECONDS. Each slice is
# then decoded over a slightly wider window, and only words whose midpoint
# lies in the slice's own range are kept, so a word at a cut is whole in one
# slice and kept exactly once. A window edge goes to the nearest pause within
# EDGE_SEARCH_SECONDS (none needed when the cut itself is a pause), else
# SLICE_OVERLAP_SECONDS / 2 past the cut: starting a window mid-word can make
# Whisper skip seconds of speech after it (observed live: 3.6 s dropped).
SLICE_OVERLAP_SECONDS = 1.5
SNAP_SECONDS = 2.5
EDGE_SEARCH_SECONDS = 1.5
# Snapping may lengthen a slice, but never past this (the point of slicing is
# to stay well under the decoder's per-window token cap).
MAX_SLICE_SECONDS = 12.0
SILENCE_NOISE = "-35dB"
SILENCE_MIN_SECONDS = 0.2
SILENCE_RE = re.compile(r"silence_(start|end): (-?[0-9.]+)")


def mlx_path() -> str | None:
    """Absolute path of the `mlx_whisper` CLI (PATH, then pipx's bin dir), or None."""
    found = shutil.which(MLX_BIN)
    if found:
        return found
    candidate = PIPX_BIN_DIR / MLX_BIN
    return str(candidate) if os.access(candidate, os.X_OK) else None


def mlx_available() -> bool:
    return mlx_path() is not None


def model_label(model: str = MLX_MODEL) -> str:
    """Short model name for the report, e.g. `large-v3-turbo`."""
    return model.rsplit("/", 1)[-1].removeprefix("whisper-")


def model_cached(model: str = MLX_MODEL) -> bool:
    """True once the model weights are fully in the Hugging Face cache.

    huggingface_hub creates the model folder (and `blobs/*.incomplete`) at the
    *start* of a download; the `snapshots/<rev>/weights.*` link only appears on
    completion, so that is what we test. Otherwise an interrupted `--warm`
    would look cached and the remaining download would run silently inside a
    watch run, the exact failure warm() exists to prevent.
    """
    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    hub = Path(os.environ.get("HF_HUB_CACHE") or hf_home / "hub")
    snapshots = hub / f"models--{model.replace('/', '--')}" / "snapshots"
    return any(p.exists() for p in snapshots.glob("*/weights.*")) if snapshots.is_dir() else False


def warm(work_dir: Path) -> None:
    """Fetch the model by transcribing one second of silence. Idempotent."""
    work_dir.mkdir(parents=True, exist_ok=True)
    silence = work_dir / "silence.wav"
    result = run_text(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "lavfi", "-t", "1", "-i", "anullsrc=r=16000:cl=mono",
                       "-acodec", "pcm_s16le", str(silence)])
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg could not create warm-up audio: {diagnostic(result.stderr)}")
    transcribe_mlx(silence, work_dir)


def transcribe_mlx(
    audio_path: Path,
    out_dir: Path,
    temperature: float = MLX_TEMPERATURE,
    model: str = MLX_MODEL,
    *,
    words: bool = False,
) -> list[dict]:
    """Run `mlx_whisper` on one audio file and return its 0-based segments.

    Uses JSON output, which carries the same `segments` shape as the APIs'
    verbose_json, so whisper.py's parser serves every backend. The
    repetition-safe decoding flags are always on (see MLX_TEMPERATURE).
    With `words`, the CLI also emits word timestamps and each returned segment
    carries a `words` list of {start, end, text} (only when every word is valid).
    """
    from whisper import _segments_from_response  # lazy: whisper imports this module

    binary = mlx_path()
    if binary is None:
        raise SystemExit(
            f"{MLX_BIN} is not installed. Install with: pipx install mlx-whisper "
            "(Apple Silicon only)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mlx_{audio_path.stem}"
    json_path = out_dir / f"{stem}.json"
    json_path.unlink(missing_ok=True)  # never parse a previous run's output
    cmd = [
        binary,
        str(audio_path.resolve()),
        "--model", model,
        "--output-format", "json",
        "--output-dir", str(out_dir.resolve()),
        "--output-name", stem,
        "--condition-on-previous-text", "False",
        "--temperature", f"{temperature:g}",
        "--verbose", "False",
    ]
    if words:
        cmd += ["--word-timestamps", "True"]
    result = run_text(cmd)
    if result.returncode != 0:
        tail = diagnostic(result.stderr or result.stdout, 1500)
        raise SystemExit(f"{MLX_BIN} failed (exit {result.returncode}): {tail}")

    if not json_path.exists():
        raise SystemExit(f"{MLX_BIN} produced no output at {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{MLX_BIN} wrote unreadable JSON: {type(exc).__name__}") from None
    # mlx emits an occasional blank-text segment (a window it decoded as
    # silence); the shared parser rightly rejects those, so drop them first.
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        data = {**data, "segments": [
            seg for seg in data["segments"]
            if not (isinstance(seg, dict) and isinstance(seg.get("text"), str) and not seg["text"].strip())
        ]}
    segments = _segments_from_response(data)
    if words:
        _attach_words(segments, data)
    return segments


def _attach_words(segments: list[dict], data: dict) -> None:
    """Copy valid word timestamps from the raw JSON onto the parsed segments
    (matched by start/end). A segment with any malformed word gets none."""
    by_span = {}
    for raw in data.get("segments", []):
        if isinstance(raw, dict) and isinstance(raw.get("words"), list):
            by_span[(round(float(raw.get("start", -1)), 6), round(float(raw.get("end", -1)), 6))] = raw["words"]
    for seg in segments:
        raw_words = by_span.get((seg["start"], seg["end"]))
        parsed = []
        for word in raw_words or []:
            if not (isinstance(word, dict) and isinstance(word.get("word"), str)
                    and all(isinstance(word.get(k), (int, float)) and not isinstance(word.get(k), bool)
                            for k in ("start", "end"))):
                parsed = []
                break
            parsed.append({"start": float(word["start"]), "end": float(word["end"]), "text": word["word"]})
        if parsed:
            seg["words"] = parsed


def slice_audio(full_audio: Path, out_path: Path, start: float, end: float) -> Path:
    """Cut [start, end] out of an audio file as 16kHz mono WAV (re-encoded, so
    the cut is sample-accurate rather than snapped to an mp3 frame)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(full_audio.resolve()),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_path.resolve()),
    ]
    result = run_text(cmd)
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(f"ffmpeg failed to slice audio window: {diagnostic(result.stderr)}")
    return out_path


def retry_window(
    segments: list[dict],
    problem: dict,
    duration_seconds: float,
    pad: float = LOOP_RETRY_PAD_SECONDS,
) -> tuple[float, float]:
    """The [lo, hi] span to re-transcribe for one flagged window.

    Pad the window, then widen to the full extent of every segment the padded
    span overlaps. splice_segments drops overlapping segments whole, so the
    retry must cover everything it removes or the words outside the pad
    would be lost.
    """
    lo = max(0.0, problem["start"] - pad)
    hi = problem["end"] + pad
    while True:  # to a fixpoint: widening can pull in further overlapping segments
        before = (lo, hi)
        for seg in segments:
            start, end = float(seg["start"]), float(seg["end"])
            if end > lo and start < hi:
                lo, hi = min(lo, start), max(hi, end)
        if (lo, hi) == before:
            break
    lo = max(0.0, lo)
    if duration_seconds:
        hi = min(duration_seconds, hi)
    return lo, hi


def repair_loops(
    segments: list[dict],
    loops: list[dict],
    audio_path: Path,
    work_dir: Path,
    duration_seconds: float,
) -> list[dict]:
    """Re-transcribe each looped window in isolation and splice it back in.

    Cutting the window out resets the decoder state that produced the loop, and
    the higher retry temperature lets it escape a greedy repetition. A window
    whose retry fails keeps its original (looped) segments so the verify pass
    can still report it.
    """
    from whisper import shift_segments  # lazy: whisper imports this module

    loops = _capped(loops, "repetition loop")
    for index, loop in enumerate(loops):
        lo, hi = retry_window(segments, loop, duration_seconds)
        print(
            f"[watch] repetition loop {index + 1}/{len(loops)} at "
            f"{lo:.0f}s-{hi:.0f}s (×{loop['count']} {loop['text'][:40]!r}) — re-transcribing window…",
            file=sys.stderr,
        )
        try:
            window = slice_audio(audio_path, work_dir / f"loop_{index:02d}.wav", lo, hi)
            redo = transcribe_mlx(window, work_dir, temperature=MLX_RETRY_TEMPERATURE)
        except SystemExit as exc:
            print(f"[watch] window retry failed — keeping original ({exc})", file=sys.stderr)
            continue
        if not redo:
            # An empty retry would delete the window without a trace; keep the
            # looped originals so the re-verify still reports it.
            print("[watch] window retry returned nothing — keeping original", file=sys.stderr)
            continue
        segments = splice_segments(segments, lo, hi, shift_segments(redo, lo))
    return segments


def detect_silences(audio_path: Path, lo: float, hi: float) -> list[tuple[float, float]]:
    """Pauses inside [lo, hi] as (start, end) source seconds; [] if ffmpeg fails."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{lo:.3f}", "-to", f"{hi:.3f}",
           "-i", str(audio_path.resolve()),
           "-af", f"silencedetect=noise={SILENCE_NOISE}:d={SILENCE_MIN_SECONDS}", "-f", "null", "-"]
    try:
        result = run_text(cmd)
    except SystemExit:
        return []
    if result.returncode != 0:
        return []
    out, start = [], None
    for kind, value in SILENCE_RE.findall(result.stderr or ""):
        t = lo + max(0.0, float(value))  # -ss before -i: timestamps restart at 0
        if kind == "start":
            start = t
        elif start is not None:
            out.append((start, min(t, hi)))
            start = None
    if start is not None:
        out.append((start, hi))
    return out


def snap_cuts(plan: list[tuple[float, float]], silences: list[tuple[float, float]],
              snap: float = SNAP_SECONDS, max_slice: float = MAX_SLICE_SECONDS,
              ) -> list[tuple[float, float]]:
    """Move each inner cut of a contiguous plan to the nearest pause midpoint
    within `snap` seconds, keeping cuts ordered and no slice longer than
    `max_slice`."""
    if len(plan) < 2:
        return plan
    lo, hi = plan[0][0], plan[-1][1]
    mids = [(a + b) / 2 for a, b in silences]
    cuts = []
    inner = [end for _, end in plan[:-1]]
    for i, nominal in enumerate(inner):
        prev = cuts[-1] if cuts else lo
        nxt = inner[i + 1] if i + 1 < len(inner) else hi
        near = [m for m in mids if abs(m - nominal) <= snap and prev + 1.0 < m < hi - 1.0
                and m - prev <= max_slice and nxt - m <= max_slice]
        cuts.append(round(min(near, key=lambda m: abs(m - nominal)), 3) if near else nominal)
    bounds = [lo, *cuts, hi]
    return list(zip(bounds, bounds[1:]))


def decode_window(start: float, end: float, silences: list[tuple[float, float]],
                  duration_seconds: float = 0.0) -> tuple[float, float]:
    """The audio span decoded for the slice owning [start, end]: each edge moves
    out to the nearest pause within EDGE_SEARCH_SECONDS (staying put when the
    edge is itself in a pause), else by SLICE_OVERLAP_SECONDS / 2."""
    half = SLICE_OVERLAP_SECONDS / 2
    before = [p for p in silences if p[0] <= start and p[1] >= start - EDGE_SEARCH_SECONDS]
    after = [p for p in silences if p[1] >= end and p[0] <= end + EDGE_SEARCH_SECONDS]
    a = min(start, max((p[0] + p[1]) / 2 for p in before)) if before else start - half
    b = max(end, min((p[0] + p[1]) / 2 for p in after)) if after else end + half
    a = max(0.0, a)
    b = min(duration_seconds, b) if duration_seconds else b
    return round(a, 3), round(b, 3)


def keep_owned(segments: list[dict], start: float, end: float, last: bool) -> list[dict]:
    """Of a slice's (source-time) segments, keep what this slice owns:
    words whose midpoint is in [start, end) ([start, end] for the last slice),
    regrouped into their segments; segments without word timestamps are kept
    or dropped whole by their own midpoint."""
    def owned(a: float, b: float) -> bool:
        mid = (a + b) / 2
        return start <= mid < end or (last and mid == end)

    out = []
    for seg in segments:
        words = seg.get("words")
        if not words:
            if owned(seg["start"], seg["end"]):
                out.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            continue
        kept = [w for w in words if owned(w["start"], w["end"])]
        text = "".join(w["text"] for w in kept).strip()
        if text:
            out.append({"start": round(kept[0]["start"], 2), "end": round(kept[-1]["end"], 2), "text": text})
    return out


def slice_plan(lo: float, hi: float, slice_seconds: float = TRUNCATED_SLICE_SECONDS) -> list[tuple[float, float]]:
    """Split [lo, hi] into equal contiguous slices no longer than slice_seconds."""
    span = hi - lo
    if span <= 0:
        return []
    count = max(1, math.ceil(span / slice_seconds - 1e-9))
    step = span / count
    return [(round(lo + i * step, 3), round(hi if i == count - 1 else lo + (i + 1) * step, 3))
            for i in range(count)]


def repair_truncated(
    segments: list[dict],
    truncated: list[dict],
    audio_path: Path,
    work_dir: Path,
    duration_seconds: float,
) -> list[dict]:
    """Re-transcribe each truncated window as short slices and splice them in.

    The cut comes from the per-window token cap, so decoding the same 30s again
    reproduces it; slices of TRUNCATED_SLICE_SECONDS each carry less speech than
    the cap. Cuts snap to nearby pauses, each slice is decoded with overlap and
    word timestamps, and only the words a slice owns are kept (keep_owned), so
    no word is cut in half or emitted twice. If any slice fails, or every slice
    comes back empty, the original segment is kept so the re-verify still
    reports the window.
    """
    truncated = _capped(truncated, "truncated segment")
    for index, seg in enumerate(truncated):
        lo, hi = retry_window(segments, seg, duration_seconds, pad=0.0)
        silences = detect_silences(audio_path, max(0.0, lo - EDGE_SEARCH_SECONDS), hi + EDGE_SEARCH_SECONDS)
        plan = snap_cuts(slice_plan(lo, hi), silences)
        print(
            f"[watch] truncated segment {index + 1}/{len(truncated)} at {lo:.0f}s-{hi:.0f}s — "
            f"re-transcribing as {len(plan)} slices…",
            file=sys.stderr,
        )
        redo: list[dict] = []
        try:
            for part, (start, end) in enumerate(plan):
                a, b = decode_window(start, end, silences, duration_seconds)
                window = slice_audio(audio_path, work_dir / f"trunc_{index:02d}_{part:02d}.wav", a, b)
                decoded = _shift_with_words(transcribe_mlx(window, work_dir, words=True), a)
                redo.extend(keep_owned(decoded, start, end, last=part == len(plan) - 1))
        except SystemExit as exc:
            print(f"[watch] slice retry failed — keeping original ({exc})", file=sys.stderr)
            continue
        if not redo:
            print("[watch] slice retry returned nothing — keeping original", file=sys.stderr)
            continue
        segments = splice_segments(segments, lo, hi, redo)
    return segments


def _shift_with_words(segments: list[dict], offset: float) -> list[dict]:
    """shift_segments, but carrying word timestamps along."""
    out = []
    for seg in segments:
        moved = {"start": seg["start"] + offset, "end": seg["end"] + offset, "text": seg["text"]}
        if seg.get("words"):
            moved["words"] = [{"start": w["start"] + offset, "end": w["end"] + offset, "text": w["text"]}
                              for w in seg["words"]]
        out.append(moved)
    return out


def _capped(problems: list[dict], label: str) -> list[dict]:
    """The first MAX_REPAIR_WINDOWS problems; logs how many are left as-is."""
    if len(problems) > MAX_REPAIR_WINDOWS:
        print(f"[watch] {len(problems) - MAX_REPAIR_WINDOWS} more {label} window(s) left unrepaired "
              f"(repair cap {MAX_REPAIR_WINDOWS}); they stay flagged in the report", file=sys.stderr)
    return problems[:MAX_REPAIR_WINDOWS]
