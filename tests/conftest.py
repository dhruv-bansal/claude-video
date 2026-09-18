"""Shared pytest fixtures: ffmpeg-synthesized clips and scripts/ on sys.path."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Make the bundled scripts importable (mirrors watch.py's sys.path insert).
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# 14 visually distinct fills → 14 abrupt cuts → x264 emits a keyframe per cut.
COLORS = [
    "red", "green", "blue", "white", "black", "yellow", "cyan",
    "magenta", "gray", "orange", "purple", "brown", "navy", "olive",
]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def build_cut_clip(
    path: Path,
    n: int = 14,
    seg: float = 0.4,
    size: str = "320x240",
    fps: int = 10,
) -> None:
    """Concatenate ``n`` solid-color segments into one clip with ``n`` cuts.

    Each color change is a hard scene cut, so the scene selector finds ~n-1
    changes. x264's own scenecut detection is unreliable on flat fills, so we
    force a keyframe at every ``seg`` boundary — giving ~n real keyframes for
    the keyframe engine to find.
    """
    inputs: list[str] = []
    for i in range(n):
        color = COLORS[i % len(COLORS)]
        inputs += ["-f", "lavfi", "-t", str(seg), "-i", f"color=c={color}:s={size}:r={fps}"]
    streams = "".join(f"[{i}:v]" for i in range(n))
    filt = f"{streams}concat=n={n}:v=1:a=0[out]"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", filt, "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-force_key_frames", f"expr:gte(t,n_forced*{seg})",
        str(path),
    ])


def build_static_clip(
    path: Path,
    duration: float = 3.0,
    size: str = "320x240",
    fps: int = 10,
) -> None:
    """One solid color: 1 keyframe, no scene changes → triggers both fallbacks."""
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", str(duration), "-i", f"color=c=blue:s={size}:r={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "600",
        str(path),
    ])


def build_deck_clip(
    path: Path,
    n_slides: int = 3,
    seg: float = 1.0,
    size: str = "1280x720",
    fps: int = 10,
) -> None:
    """A slide deck: white slides, each adding one dark "text line".

    Consecutive slides differ by a whole-frame mean of ~2/255 — the same
    signal a real deck on a shared template produces — which is exactly what
    whole-frame dedup at threshold 2.0 collapsed. Tiled dedup must keep all.
    """
    w = int(size.split("x")[0])
    inputs: list[str] = []
    for i in range(n_slides):
        boxes = "".join(
            f",drawbox=x=100:y={100 + 60 * k}:w={w // 5}:h=24:c=black:t=fill"
            for k in range(i + 1)
        )
        inputs += ["-f", "lavfi", "-t", str(seg), "-i", f"color=c=white:s={size}:r={fps}{boxes}"]
    streams = "".join(f"[{i}:v]" for i in range(n_slides))
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", f"{streams}concat=n={n_slides}:v=1:a=0[out]", "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-force_key_frames", f"expr:gte(t,n_forced*{seg})",
        str(path),
    ])


@pytest.fixture(scope="session")
def deck_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "deck.mp4"
    build_deck_clip(path)
    return path


@pytest.fixture(scope="session")
def cut_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "cuts.mp4"
    build_cut_clip(path)
    return path


@pytest.fixture(scope="session")
def static_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "static.mp4"
    build_static_clip(path)
    return path
