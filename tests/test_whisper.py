"""Whisper auto-chunking: plan, split, and timestamp stitching."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

import whisper


MB = 1024 * 1024


class TestPlanChunks:
    def test_under_limit_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=600.0, total_bytes=5 * MB, max_bytes=24 * MB)
        assert plan == [(0.0, 600.0)]

    def test_at_limit_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=600.0, total_bytes=24 * MB, max_bytes=24 * MB)
        assert plan == [(0.0, 600.0)]

    def test_over_limit_splits_into_enough_chunks(self):
        # 71 MB against a 24 MB cap → ceil(71/24) = 3 chunks.
        plan = whisper.plan_chunks(total_seconds=3600.0, total_bytes=71 * MB, max_bytes=24 * MB)
        assert len(plan) == 3

    def test_chunks_are_contiguous_and_cover_full_duration(self):
        total = 3600.0
        plan = whisper.plan_chunks(total_seconds=total, total_bytes=71 * MB, max_bytes=24 * MB)
        # Offsets start at 0 and each picks up where the previous ended.
        assert plan[0][0] == 0.0
        for (off, dur), (next_off, _) in zip(plan, plan[1:]):
            assert math.isclose(off + dur, next_off)
        last_off, last_dur = plan[-1]
        assert math.isclose(last_off + last_dur, total)

    def test_each_chunk_estimated_under_limit(self):
        total_seconds, total_bytes, cap = 3600.0, 71 * MB, 24 * MB
        plan = whisper.plan_chunks(total_seconds, total_bytes, cap)
        bytes_per_second = total_bytes / total_seconds
        for _off, dur in plan:
            assert dur * bytes_per_second <= cap

    def test_zero_duration_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=0.0, total_bytes=0, max_bytes=24 * MB)
        assert plan == [(0.0, 0.0)]


class TestShiftSegments:
    def test_adds_offset_to_start_and_end(self):
        segs = [{"start": 0.0, "end": 2.5, "text": "hi"}, {"start": 2.5, "end": 4.0, "text": "there"}]
        shifted = whisper.shift_segments(segs, 1800.0)
        assert shifted == [
            {"start": 1800.0, "end": 1802.5, "text": "hi"},
            {"start": 1802.5, "end": 1804.0, "text": "there"},
        ]

    def test_zero_offset_is_identity(self):
        segs = [{"start": 1.0, "end": 2.0, "text": "x"}]
        assert whisper.shift_segments(segs, 0.0) == segs

    def test_does_not_mutate_input(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "x"}]
        whisper.shift_segments(segs, 10.0)
        assert segs[0]["start"] == 0.0


def _make_mp3(path: Path, seconds: float) -> None:
    """Synthesize a mono 16k 64k mp3 of a sine tone — mirrors extract_audio's format."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-t", str(seconds), "-i", "sine=frequency=440:sample_rate=16000",
            "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
            str(path),
        ],
        check=True,
    )


class TestSplitAudio:
    def test_creates_one_file_per_plan_entry(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        assert len(chunks) == 2
        for chunk_path, _offset in chunks:
            assert chunk_path.exists() and chunk_path.stat().st_size > 0

    def test_returns_plan_offsets(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        assert [offset for _path, offset in chunks] == [0.0, 3.0]

    def test_chunks_are_smaller_than_full(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        full_size = full.stat().st_size
        for chunk_path, _offset in chunks:
            assert chunk_path.stat().st_size < full_size


class TestAudioDuration:
    def test_reads_duration_of_synthesized_clip(self, tmp_path: Path):
        audio = tmp_path / "audio.mp3"
        _make_mp3(audio, 5.0)
        assert whisper.audio_duration(audio) == pytest.approx(5.0, abs=0.5)


class TestTranscribeChunks:
    def test_shifts_and_concatenates_each_chunk(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def fake_transcribe(path: Path) -> list[dict]:
            return [{"start": 0.0, "end": 2.0, "text": path.stem}]

        out = whisper.transcribe_chunks(chunks, fake_transcribe)

        assert out == [
            {"start": 0.0, "end": 2.0, "text": "a"},
            {"start": 100.0, "end": 102.0, "text": "b"},
        ]

    def test_keeps_successful_chunks_when_one_fails(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def flaky(path: Path) -> list[dict]:
            if path.stem == "b":
                raise SystemExit("chunk b failed")
            return [{"start": 1.0, "end": 2.0, "text": "a"}]

        out = whisper.transcribe_chunks(chunks, flaky)

        assert out == [{"start": 1.0, "end": 2.0, "text": "a"}]

    def test_raises_when_every_chunk_fails(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def always_fail(path: Path) -> list[dict]:
            raise SystemExit("boom")

        with pytest.raises(SystemExit):
            whisper.transcribe_chunks(chunks, always_fail)


# --- backend resolution ------------------------------------------------------

class TestResolveBackend:
    @pytest.fixture(autouse=True)
    def _no_keys(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("WATCH_WHISPER", raising=False)
        # Keep the developer's real ~/.config/watch/.env and cwd .env out of it.
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.chdir(tmp_path)
        import config
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")

    def test_auto_prefers_mlx_when_installed(self, monkeypatch):
        monkeypatch.setattr(whisper, "mlx_available", lambda: True)
        monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
        assert whisper.resolve_backend() == ("mlx", "")

    def test_auto_falls_back_to_groq_then_openai(self, monkeypatch):
        monkeypatch.setattr(whisper, "mlx_available", lambda: False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
        assert whisper.resolve_backend() == ("openai", "sk-oa")
        monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
        assert whisper.resolve_backend() == ("groq", "sk-groq")

    def test_auto_nothing_available(self, monkeypatch):
        monkeypatch.setattr(whisper, "mlx_available", lambda: False)
        assert whisper.resolve_backend() == (None, None)

    def test_pinned_mlx_without_binary_does_not_fall_back(self, monkeypatch):
        monkeypatch.setattr(whisper, "mlx_available", lambda: False)
        monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
        assert whisper.resolve_backend("mlx") == (None, None)

    def test_pinned_api_ignores_mlx(self, monkeypatch):
        monkeypatch.setattr(whisper, "mlx_available", lambda: True)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
        assert whisper.resolve_backend("openai") == ("openai", "sk-oa")
        assert whisper.resolve_backend("groq") == (None, None)

    def test_config_preference_is_honored(self, monkeypatch):
        monkeypatch.setattr(whisper, "mlx_available", lambda: True)
        monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
        monkeypatch.setenv("WATCH_WHISPER", "groq")
        assert whisper.resolve_backend() == ("groq", "sk-groq")


class TestExtractAudioWav:
    def test_wav_suffix_yields_pcm(self, tmp_path: Path):
        video = tmp_path / "v.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-t", "2", "-i", "color=c=blue:s=64x64:r=5",
             "-f", "lavfi", "-t", "2", "-i", "sine=frequency=440:sample_rate=16000",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)],
            check=True,
        )
        out = whisper.extract_audio(str(video), tmp_path / "audio.wav")
        assert out.read_bytes()[:4] == b"RIFF"
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name,sample_rate,channels", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert probe == "pcm_s16le,16000,1"
