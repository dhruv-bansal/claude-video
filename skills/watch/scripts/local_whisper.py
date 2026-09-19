#!/usr/bin/env python3
"""Local Whisper backend: the `mlx_whisper` CLI on the Apple Silicon GPU.

Nothing leaves the machine, no API key, no per-minute cost. The model
(~1.5 GB) is fetched from Hugging Face on first use into
~/.cache/huggingface/hub/. Install the CLI with `pipx install mlx-whisper`.

Pure stdlib here — mlx itself lives in the CLI's own environment. Segments
come back in the same {start, end, text} shape as the API backends, so
whisper.py treats this as just another source.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from verify import splice_segments

MLX_BIN = "mlx_whisper"
MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
# `pipx install` lands here; it is only on PATH after `pipx ensurepath` and a
# new shell, so look there explicitly rather than fail on a fresh machine.
PIPX_BIN_DIR = Path.home() / ".local" / "bin"
# Whisper conditions each 30s window on its own previous output. On a
# low-confidence patch it can latch onto what it just emitted and repeat it for
# minutes while discarding the real speech — fluent, complete-looking, wrong.
# Disabling the conditioning stops the loop feeding itself; a little
# temperature lets the decoder escape a greedy repetition already underway.
MLX_TEMPERATURE = 0.2
MLX_RETRY_TEMPERATURE = 0.4
# Padding around a looped window when re-transcribing it in isolation.
LOOP_RETRY_PAD_SECONDS = 5.0


def mlx_path() -> str | None:
    """Absolute path of the `mlx_whisper` CLI (PATH, then pipx's bin dir), or None."""
    found = shutil.which(MLX_BIN)
    if found:
        return found
    candidate = PIPX_BIN_DIR / MLX_BIN
    return str(candidate) if os.access(candidate, os.X_OK) else None


def mlx_available() -> bool:
    return mlx_path() is not None


def model_cached(model: str = MLX_MODEL) -> bool:
    """True once the model weights are fully in the Hugging Face cache.

    huggingface_hub creates the model folder (and `blobs/*.incomplete`) at the
    *start* of a download; the `snapshots/<rev>/weights.*` link only appears on
    completion, so that is what we test. Otherwise an interrupted `--warm`
    would look cached and the remaining download would run silently inside a
    watch run — the exact failure warm() exists to prevent.
    """
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
    snapshots = hub / f"models--{model.replace('/', '--')}" / "snapshots"
    return any(p.exists() for p in snapshots.glob("*/weights.*")) if snapshots.is_dir() else False


def warm(work_dir: Path) -> None:
    """Fetch the model by transcribing one second of silence. Idempotent."""
    work_dir.mkdir(parents=True, exist_ok=True)
    silence = work_dir / "silence.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-t", "1", "-i", "anullsrc=r=16000:cl=mono",
         "-acodec", "pcm_s16le", str(silence)],
        check=True,
    )
    transcribe_mlx(silence, work_dir)


def transcribe_mlx(
    audio_path: Path,
    out_dir: Path,
    temperature: float = MLX_TEMPERATURE,
    model: str = MLX_MODEL,
) -> list[dict]:
    """Run `mlx_whisper` on one audio file and return its 0-based segments.

    Uses JSON output, which carries the same `segments` shape as the APIs'
    verbose_json, so whisper.py's parser serves every backend. The
    repetition-safe decoding flags are always on — see MLX_TEMPERATURE.
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-1500:]
        raise SystemExit(f"{MLX_BIN} failed (exit {result.returncode}): {tail}")

    json_path = out_dir / f"{stem}.json"
    if not json_path.exists():
        raise SystemExit(f"{MLX_BIN} produced no output at {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{MLX_BIN} wrote unreadable JSON: {exc}")
    return _segments_from_response(data)


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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(f"ffmpeg failed to slice audio window: {result.stderr.strip()}")
    return out_path


def retry_window(segments: list[dict], loop: dict, duration_seconds: float) -> tuple[float, float]:
    """The [lo, hi] span to re-transcribe for one loop.

    Pad the loop, then widen to the full extent of every segment the padded
    span overlaps. splice_segments drops overlapping segments whole, so the
    retry must cover everything it removes or the words outside the pad
    would be lost.
    """
    lo = max(0.0, loop["start"] - LOOP_RETRY_PAD_SECONDS)
    hi = loop["end"] + LOOP_RETRY_PAD_SECONDS
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
