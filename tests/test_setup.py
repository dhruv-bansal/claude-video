"""setup.py --json surfaces the resolved watch detail and Whisper backend."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"
REQUIRED = ("ffmpeg", "ffprobe", "yt-dlp")


def _shim_path(root: Path, *, mlx: bool = False) -> str:
    """A PATH containing only the required binaries (symlinked from the real
    ones) plus, optionally, a fake `mlx_whisper`. Keeps the test independent
    of whether the developer's machine has mlx-whisper installed."""
    shim = root / "bin"
    shim.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        real = shutil.which(name)
        if real and not (shim / name).exists():
            os.symlink(real, shim / name)
    if mlx:
        fake = shim / "mlx_whisper"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
    return str(shim)


def _run(args, *, home=None, extra_env=None, mlx=False):
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    env.pop("WATCH_WHISPER", None)
    # Don't let a real key in the developer's shell env leak into the test.
    env.pop("GROQ_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("SETUP_COMPLETE", None)
    if home is not None:
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)  # Windows
        env["PATH"] = _shim_path(Path(home), mlx=mlx)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SETUP), *args],
        capture_output=True, text=True, env=env,
    )


def _write_env(home: Path, body: str) -> None:
    cfg = home / ".config" / "watch"
    cfg.mkdir(parents=True, exist_ok=True)
    f = cfg / ".env"
    f.write_text(body, encoding="utf-8")
    f.chmod(0o600)


def test_json_reports_watch_detail():
    proc = _run(["--json"])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["watch_detail"] == "balanced"


def test_keyless_completed_setup_proceeds_silently(tmp_path):
    """A user who finished setup without a key must NOT be nagged forever."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\nSETUP_COMPLETE=true\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, f"keyless-complete should pass --check; got {chk.returncode}: {chk.stderr}"
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is True
    assert js["first_run"] is False
    assert js["setup_complete"] is True
    # status still encourages a key even though we can proceed
    assert js["status"] == "needs_key"


def test_keyless_first_run_is_encouraged(tmp_path):
    """Genuine first run with no key: --check reports exit 3 (encourage a key)."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is False
    assert js["first_run"] is True


def test_key_present_is_ready(tmp_path):
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["whisper_backend"] == "groq"
    assert js["has_mlx_whisper"] is False


# --- local mlx-whisper backend -------------------------------------------------

def test_mlx_installed_is_ready_without_key(tmp_path):
    """mlx_whisper on PATH is a complete Whisper backend: no key, no nag."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    chk = _run(["--check"], home=tmp_path, mlx=True)
    assert chk.returncode == 0, chk.stderr
    assert chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path, mlx=True).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["has_api_key"] is False
    assert js["has_mlx_whisper"] is True
    assert js["whisper_backend"] == "mlx"


def test_mlx_preferred_over_api_key_in_auto(tmp_path):
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    js = json.loads(_run(["--json"], home=tmp_path, mlx=True).stdout)
    assert js["whisper_backend"] == "mlx"


def test_watch_whisper_pins_api_backend_over_mlx(tmp_path):
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\nWATCH_WHISPER=groq\n")
    js = json.loads(_run(["--json"], home=tmp_path, mlx=True).stdout)
    assert js["whisper_backend"] == "groq"


def test_watch_whisper_mlx_without_binary_is_needs_key(tmp_path):
    """Pinning mlx on a machine without it must not silently fall back to a key."""
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\nWATCH_WHISPER=mlx\n")
    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["whisper_backend"] is None
    assert js["status"] == "needs_key"


def test_installer_marks_complete_with_mlx_and_no_key(tmp_path):
    proc = _run([], home=tmp_path, mlx=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "whisper backend: mlx" in proc.stdout
    env_text = (tmp_path / ".config" / "watch" / ".env").read_text()
    assert "SETUP_COMPLETE=true" in env_text
    assert "WATCH_WHISPER" in env_text  # scaffolded template documents the preference


def test_json_reports_model_cache_and_warm_is_noop_without_mlx(tmp_path):
    js = json.loads(_run(["--json"], home=tmp_path / "with-mlx", mlx=True).stdout)
    assert js["mlx_model_cached"] is False  # fresh HOME → no HF cache

    proc = _run(["--warm"], home=tmp_path / "without-mlx")  # no mlx on this shim PATH
    assert proc.returncode == 2
    assert "not installed" in proc.stderr
