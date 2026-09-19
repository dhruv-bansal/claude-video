"""Local mlx-whisper backend: drives a fake CLI, parses JSON, repairs loops."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import local_whisper
import verify


def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


# --- local mlx runner: drives a fake CLI, parses its JSON -----------------------

def _fake_mlx(bin_dir: Path, segments: list[dict], record: Path | None = None) -> None:
    """Install a stand-in `mlx_whisper` that writes the given segments as JSON
    to --output-dir/--output-name and, optionally, records its argv."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "mlx_whisper"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, pathlib\n"
        "a = sys.argv[1:]\n"
        f"segs = {json.dumps(segments)}\n"
        "out = pathlib.Path(a[a.index('--output-dir') + 1]) / (a[a.index('--output-name') + 1] + '.json')\n"
        "out.write_text(json.dumps({'text': ' '.join(s['text'] for s in segs), 'segments': segs}))\n"
        + (f"pathlib.Path({str(record)!r}).write_text(json.dumps(a))\n" if record else "")
    )
    script.chmod(0o755)


class TestTranscribeMlx:
    def test_parses_json_and_passes_safe_flags(self, tmp_path: Path, monkeypatch):
        record = tmp_path / "argv.json"
        _fake_mlx(tmp_path / "bin", [{"start": 0, "end": 1.5, "text": " hello "}], record)
        monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")

        segs = local_whisper.transcribe_mlx(audio, tmp_path / "mlx")

        assert segs == [{"start": 0.0, "end": 1.5, "text": "hello"}]
        argv = json.loads(record.read_text())
        assert argv[0] == str(audio.resolve())
        assert argv[argv.index("--model") + 1] == local_whisper.MLX_MODEL
        assert argv[argv.index("--condition-on-previous-text") + 1] == "False"
        assert argv[argv.index("--temperature") + 1] == "0.2"
        assert argv[argv.index("--output-format") + 1] == "json"

    def test_missing_binary_is_clear_error(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(local_whisper, "mlx_path", lambda: None)
        with pytest.raises(SystemExit, match="pipx install mlx-whisper"):
            local_whisper.transcribe_mlx(tmp_path / "a.wav", tmp_path)

    def test_nonzero_exit_surfaces_stderr(self, tmp_path: Path, monkeypatch):
        bin_dir = tmp_path / "bin"; bin_dir.mkdir()
        bad = bin_dir / "mlx_whisper"
        bad.write_text("#!/bin/sh\necho 'boom: no metal device' >&2\nexit 7\n"); bad.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
        with pytest.raises(SystemExit, match="exit 7.*no metal device"):
            local_whisper.transcribe_mlx(tmp_path / "a.wav", tmp_path)


def _make_wav(path: Path, seconds: float) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-t", str(seconds), "-i", "sine=frequency=440:sample_rate=16000",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(path)],
        check=True,
    )


class TestRepairLoopsMlx:
    def test_looped_window_is_resliced_and_spliced(self, tmp_path: Path, monkeypatch):
        """The fake CLI returns the fix for the window; the loop must be gone
        and the replacement shifted into source time."""
        _fake_mlx(tmp_path / "bin", [{"start": 1.0, "end": 3.0, "text": "real speech"}])
        monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 40.0)

        segs = [_seg(0, 5, "intro")] + [_seg(10 + i, 11 + i, "Thank you.") for i in range(4)] + [_seg(30, 35, "outro")]
        loops = verify.find_repetition_runs(segs)
        assert len(loops) == 1

        out = local_whisper.repair_loops(segs, loops, audio, tmp_path / "mlx", duration_seconds=40.0)

        assert verify.find_repetition_runs(out) == []
        texts = [s["text"] for s in out]
        assert texts == ["intro", "real speech", "outro"]
        # window lo = 10 - 5 = 5 → replacement 1.0..3.0 lands at 6.0..8.0 source time
        assert out[1]["start"] == 6.0 and out[1]["end"] == 8.0
        assert (tmp_path / "mlx" / "loop_00.wav").exists()


# --- discovery on a fresh machine ------------------------------------------------

class TestMlxPath:
    def test_falls_back_to_pipx_bin_dir_when_not_on_path(self, tmp_path: Path, monkeypatch):
        """`pipx install` lands in ~/.local/bin, which is not on PATH until
        `pipx ensurepath` + a new shell. Must still be found."""
        pipx_bin = tmp_path / ".local" / "bin"
        _fake_mlx(pipx_bin, [])
        monkeypatch.setattr(local_whisper, "PIPX_BIN_DIR", pipx_bin)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        assert local_whisper.mlx_path() == str(pipx_bin / "mlx_whisper")
        assert local_whisper.mlx_available() is True

    def test_none_when_absent_everywhere(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(local_whisper, "PIPX_BIN_DIR", tmp_path / "nope")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        assert local_whisper.mlx_path() is None


class TestModelCache:
    def test_respects_hf_home(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        assert local_whisper.model_cached() is False
        repo = tmp_path / "hf" / "hub" / "models--mlx-community--whisper-large-v3-turbo"
        # An interrupted download: folder + blobs/*.incomplete exist, no snapshot yet.
        (repo / "blobs").mkdir(parents=True)
        (repo / "blobs" / "abc.incomplete").write_bytes(b"\0")
        assert local_whisper.model_cached() is False
        snap = repo / "snapshots" / "deadbeef"
        snap.mkdir(parents=True)
        (snap / "weights.safetensors").write_bytes(b"\0")
        assert local_whisper.model_cached() is True

    def test_warm_runs_cli_on_one_second_of_silence(self, tmp_path: Path, monkeypatch):
        record = tmp_path / "argv.json"
        _fake_mlx(tmp_path / "bin", [], record)
        monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
        local_whisper.warm(tmp_path / "warm")
        argv = json.loads(record.read_text())
        assert argv[0].endswith("silence.wav")
        assert (tmp_path / "warm" / "silence.wav").stat().st_size > 0


class TestRepairLoopsGuards:
    def test_empty_retry_keeps_original_window(self, tmp_path: Path, monkeypatch):
        _fake_mlx(tmp_path / "bin", [])  # retry returns nothing
        monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 40.0)
        segs = [_seg(0, 5, "intro")] + [_seg(10 + i, 11 + i, "Thank you.") for i in range(4)]
        loops = verify.find_repetition_runs(segs)
        out = local_whisper.repair_loops(segs, loops, audio, tmp_path / "mlx", duration_seconds=40.0)
        assert out == segs  # nothing silently deleted
        assert verify.find_repetition_runs(out) == loops  # re-verify still reports it


class TestTranscribeVideoMlx:
    def test_full_local_path_returns_quality(self, tmp_path: Path, monkeypatch):
        """transcribe_video(backend="mlx"): wav extraction → CLI → verify → 3-tuple."""
        import whisper
        record = tmp_path / "argv.json"
        _fake_mlx(tmp_path / "bin", [{"start": 0.0, "end": 1.5, "text": "hello"}], record)
        monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
        video = tmp_path / "v.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-t", "2", "-i", "color=c=blue:s=64x64:r=5",
             "-f", "lavfi", "-t", "2", "-i", "sine=frequency=440:sample_rate=16000",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)],
            check=True,
        )
        segs, backend, quality = whisper.transcribe_video(str(video), tmp_path / "audio.mp3", backend="mlx", api_key="")
        assert backend == "mlx"
        assert segs == [{"start": 0.0, "end": 1.5, "text": "hello"}]
        assert quality["ok"] is True
        assert (tmp_path / "audio.wav").exists() and not (tmp_path / "audio.mp3").exists()
        assert json.loads(record.read_text())[0].endswith("audio.wav")

    def test_auto_falls_back_to_api_key_when_mlx_fails(self, tmp_path: Path, monkeypatch):
        import whisper
        bin_dir = tmp_path / "bin"; bin_dir.mkdir()
        bad = bin_dir / "mlx_whisper"
        bad.write_text("#!/bin/sh\necho 'ModuleNotFoundError: mlx' >&2\nexit 1\n"); bad.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
        monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
        monkeypatch.setattr(whisper, "extract_audio", lambda video, out: out)
        monkeypatch.setattr(whisper, "audio_duration", lambda p: 3.0)
        calls = []
        monkeypatch.setattr(whisper, "_transcribe_file", lambda b, k, p: calls.append((b, k)) or [_seg(0, 1, "api")])
        (tmp_path / "audio.mp3").write_bytes(b"x")
        (tmp_path / "audio.wav").write_bytes(b"x")

        segs, backend, _ = whisper.transcribe_video("v.mp4", tmp_path / "audio.mp3", backend="mlx", api_key="", api_fallback=True)
        assert backend == "groq" and calls == [("groq", "sk-groq")]

        with pytest.raises(SystemExit, match="mlx_whisper failed"):  # pinned: no fallback
            whisper.transcribe_video("v.mp4", tmp_path / "audio.mp3", backend="mlx", api_key="")


class TestRetryWindow:
    def test_widens_to_cover_straddling_segments(self):
        """Loop 10-13s → pad 5-18s. A good segment 3-7s straddles lo, and one
        16-21s straddles hi; both are dropped whole by the splice, so the retry
        window must grow to 3-21s or their outer words would be lost."""
        segs = [_seg(3, 7, "a"), _seg(10, 11, "x"), _seg(11, 12, "x"), _seg(12, 13, "x"), _seg(16, 21, "z")]
        loop = verify.find_repetition_runs(segs)[0]
        assert local_whisper.retry_window(segs, loop, duration_seconds=60.0) == (3.0, 21.0)

    def test_chained_overlaps_reach_fixpoint(self):
        """1-3.5 overlaps 3-7 which overlaps the pad: the window must reach 1.0,
        otherwise the splice drops 1-3.5 while the retry starts at 3.0."""
        segs = [_seg(1, 3.5, "p"), _seg(3, 7, "q"), _seg(10, 11, "x"), _seg(11, 12, "x"), _seg(12, 13, "x")]
        loop = verify.find_repetition_runs(segs)[0]
        assert local_whisper.retry_window(segs, loop, duration_seconds=60.0) == (1.0, 18.0)

    def test_clamps_to_audio_bounds(self):
        segs = [_seg(0, 1, "x"), _seg(1, 2, "x"), _seg(2, 3, "x")]
        loop = verify.find_repetition_runs(segs)[0]
        assert local_whisper.retry_window(segs, loop, duration_seconds=6.0) == (0.0, 6.0)

    def test_repair_keeps_words_before_pad(self, tmp_path: Path, monkeypatch):
        """End to end with a fake CLI: the CLI must receive the widened window
        and the pre-pad words must survive (re-emitted by the retry, not lost)."""
        record = tmp_path / "argv.json"
        # The fake returns 0-based segments for whatever window it is given;
        # lo will be 3.0, so 0-4 → 3-7 source time reproduces "A B C D".
        _fake_mlx(tmp_path / "bin", [{"start": 0.0, "end": 4.0, "text": "A B C D"},
                                     {"start": 7.0, "end": 10.0, "text": "real speech"}], record)
        monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 40.0)
        segs = [_seg(3, 7, "A B C D")] + [_seg(10 + i, 11 + i, "Thank you.") for i in range(3)] + [_seg(30, 35, "outro")]
        loops = verify.find_repetition_runs(segs)

        out = local_whisper.repair_loops(segs, loops, audio, tmp_path / "mlx", duration_seconds=40.0)

        assert [s["text"] for s in out] == ["A B C D", "real speech", "outro"]
        assert out[0]["start"] == 3.0 and out[0]["end"] == 7.0
        # ffmpeg sliced exactly the widened window (3.0 → 18.0)
        wav = tmp_path / "mlx" / "loop_00.wav"
        import subprocess as sp
        dur = float(sp.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav)],
                           capture_output=True, text=True).stdout.strip())
        assert abs(dur - 15.0) < 0.1
