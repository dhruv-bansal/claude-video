"""Local mlx-whisper backend: drives a fake CLI, parses JSON, repairs loops and
truncated windows. Never needs the real mlx_whisper: conftest hides it, and the
tests here install a fake CLI and restore the real lookup (`real_mlx_lookup`)."""
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

def _fake_mlx(bin_dir: Path, segments: list[dict], record: Path | None = None,
              by_name: dict[str, list[dict]] | None = None) -> None:
    """Install a stand-in `mlx_whisper` that writes segments as JSON to
    --output-dir/--output-name. `record` gets the last call's argv, and every
    call is appended to `<record>.all`. `by_name` maps a substring of the input
    file name to the segments returned for it (else `segments`)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "mlx_whisper"
    lines = [
        "#!/usr/bin/env python3",
        "import json, sys, pathlib",
        "a = sys.argv[1:]",
        f"segs = json.loads({json.dumps(json.dumps(segments))})",
        f"for key, value in json.loads({json.dumps(json.dumps(by_name or {}))}).items():",
        "    if key in pathlib.Path(a[0]).name:",
        "        segs = value",
        "out = pathlib.Path(a[a.index('--output-dir') + 1]) / (a[a.index('--output-name') + 1] + '.json')",
        "out.write_text(json.dumps({'text': ' '.join(s['text'] for s in segs), 'segments': segs}), encoding='utf-8')",
    ]
    if record:
        lines += [
            f"pathlib.Path({str(record)!r}).write_text(json.dumps(a))",
            f"with open({str(record) + '.all'!r}, 'a') as f:",
            "    f.write(json.dumps(a) + chr(10))",
        ]
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)


def _use_bin(monkeypatch, bin_dir: Path) -> None:
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture(autouse=True)
def _real_lookup(real_mlx_lookup):
    """Every test here drives a fake CLI through the real PATH lookup."""


def _make_wav(path: Path, seconds: float) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-t", str(seconds), "-i", "sine=frequency=440:sample_rate=16000",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(path)],
        check=True,
    )


def _make_video(path: Path, seconds: float) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-t", str(seconds), "-i", "color=c=blue:s=64x64:r=5",
         "-f", "lavfi", "-t", str(seconds), "-i", "sine=frequency=440:sample_rate=16000",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )


def _duration(path: Path) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip())


class TestTranscribeMlx:
    def test_parses_json_and_passes_safe_flags(self, tmp_path: Path, monkeypatch):
        record = tmp_path / "argv.json"
        _fake_mlx(tmp_path / "bin", [{"start": 0, "end": 1.5, "text": " hello "}], record)
        _use_bin(monkeypatch, tmp_path / "bin")
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
        _use_bin(monkeypatch, bin_dir)
        with pytest.raises(SystemExit, match="exit 7.*no metal device"):
            local_whisper.transcribe_mlx(tmp_path / "a.wav", tmp_path)

    def test_blank_text_segments_are_dropped_not_fatal(self, tmp_path: Path, monkeypatch):
        _fake_mlx(tmp_path / "bin", [{"start": 0, "end": 1, "text": " "}, {"start": 1, "end": 2, "text": "word"}])
        _use_bin(monkeypatch, tmp_path / "bin")
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        assert local_whisper.transcribe_mlx(audio, tmp_path / "mlx") == [{"start": 1.0, "end": 2.0, "text": "word"}]

    def test_model_label_is_short(self):
        assert local_whisper.model_label() == "large-v3-turbo"


class TestRepairLoopsMlx:
    def test_looped_window_is_resliced_and_spliced(self, tmp_path: Path, monkeypatch):
        """The fake CLI returns the fix for the window; the loop must be gone
        and the replacement shifted into source time."""
        _fake_mlx(tmp_path / "bin", [{"start": 1.0, "end": 3.0, "text": "real speech"}])
        _use_bin(monkeypatch, tmp_path / "bin")
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 40.0)

        segs = [_seg(0, 5, "intro")] + [_seg(10 + i, 11 + i, "Thank you.") for i in range(4)] + [_seg(30, 35, "outro")]
        loops = verify.find_repetition_runs(segs)
        assert len(loops) == 1

        out = local_whisper.repair_loops(segs, loops, audio, tmp_path / "mlx", duration_seconds=40.0)

        assert verify.find_repetition_runs(out) == []
        assert [s["text"] for s in out] == ["intro", "real speech", "outro"]
        # window lo = 10 - 5 = 5 → replacement 1.0..3.0 lands at 6.0..8.0 source time
        assert out[1]["start"] == 6.0 and out[1]["end"] == 8.0
        assert (tmp_path / "mlx" / "loop_00.wav").exists()

    def test_empty_retry_keeps_original_window(self, tmp_path: Path, monkeypatch):
        _fake_mlx(tmp_path / "bin", [])  # retry returns nothing
        _use_bin(monkeypatch, tmp_path / "bin")
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 40.0)
        segs = [_seg(0, 5, "intro")] + [_seg(10 + i, 11 + i, "Thank you.") for i in range(4)]
        loops = verify.find_repetition_runs(segs)
        out = local_whisper.repair_loops(segs, loops, audio, tmp_path / "mlx", duration_seconds=40.0)
        assert out == segs  # nothing silently deleted
        assert verify.find_repetition_runs(out) == loops  # re-verify still reports it


# --- discovery on a fresh machine ------------------------------------------------

class TestMlxPath:
    def test_conftest_hides_developer_mlx_by_default(self, monkeypatch):
        """Guard for every other test file: conftest replaces the lookup so a
        developer's real mlx_whisper never changes `auto`. (This module restores
        the real lookup, so re-apply conftest's patch to show the default.)"""
        monkeypatch.setattr(local_whisper, "mlx_path", lambda: None)
        assert local_whisper.mlx_available() is False

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
        _use_bin(monkeypatch, tmp_path / "bin")
        local_whisper.warm(tmp_path / "warm")
        argv = json.loads(record.read_text())
        assert argv[0].endswith("silence.wav")
        assert (tmp_path / "warm" / "silence.wav").stat().st_size > 0


class TestTranscribeVideoMlx:
    def test_full_local_path_returns_quality(self, tmp_path: Path, monkeypatch):
        """transcribe_video(backend="mlx"): wav extraction → CLI → verify; the
        (segments, backend) shape is unchanged and quality rides on segments."""
        import whisper
        record = tmp_path / "argv.json"
        _fake_mlx(tmp_path / "bin", [{"start": 0.0, "end": 1.5, "text": "hello"}], record)
        _use_bin(monkeypatch, tmp_path / "bin")
        video = tmp_path / "v.mp4"
        _make_video(video, 2)
        segs, backend = whisper.transcribe_video(str(video), tmp_path / "audio.mp3", backend="mlx")
        assert backend == "mlx"
        assert segs == [{"start": 0.0, "end": 1.5, "text": "hello"}]
        assert segs.quality == verify.CLEAN
        assert (tmp_path / "audio.wav").exists() and not (tmp_path / "audio.mp3").exists()
        assert json.loads(record.read_text())[0].endswith("audio.wav")

    def test_mlx_rejects_api_key(self, tmp_path: Path):
        import whisper
        with pytest.raises(SystemExit, match="API key"):
            whisper.transcribe_video("v.mp4", tmp_path / "audio.mp3", backend="mlx", api_key="sk")

    def test_auto_falls_back_to_api_key_when_mlx_fails(self, tmp_path: Path, monkeypatch):
        import whisper
        bin_dir = tmp_path / "bin"; bin_dir.mkdir()
        bad = bin_dir / "mlx_whisper"
        bad.write_text("#!/bin/sh\necho 'ModuleNotFoundError: mlx' >&2\nexit 1\n"); bad.chmod(0o755)
        _use_bin(monkeypatch, bin_dir)
        monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
        monkeypatch.setattr(whisper, "extract_audio", lambda video, out: out)
        monkeypatch.setattr(whisper, "audio_duration", lambda p: 3.0)
        calls = []
        monkeypatch.setattr(whisper, "_transcribe_file", lambda b, k, p: calls.append((b, k)) or [_seg(0, 1, "api")])
        (tmp_path / "audio.mp3").write_bytes(b"x")
        (tmp_path / "audio.wav").write_bytes(b"x")

        segs, backend = whisper.transcribe_video("v.mp4", tmp_path / "audio.mp3", backend="mlx", api_fallback=True)
        assert backend == "groq" and calls == [("groq", "sk-groq")]
        assert segs.quality["ok"] is True  # API results are verified too

        with pytest.raises(SystemExit, match="mlx_whisper failed"):  # pinned: no fallback
            whisper.transcribe_video("v.mp4", tmp_path / "audio.mp3", backend="mlx")


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

    def test_unpadded_window_is_the_segment_itself(self):
        segs = [_seg(0, 30, "long"), _seg(30, 34, "next")]
        assert local_whisper.retry_window(segs, segs[0], duration_seconds=40.0, pad=0.0) == (0.0, 30.0)

    def test_repair_keeps_words_before_pad(self, tmp_path: Path, monkeypatch):
        """End to end with a fake CLI: the CLI must receive the widened window
        and the pre-pad words must survive (re-emitted by the retry, not lost)."""
        record = tmp_path / "argv.json"
        # The fake returns 0-based segments for whatever window it is given;
        # lo will be 3.0, so 0-4 → 3-7 source time reproduces "A B C D".
        _fake_mlx(tmp_path / "bin", [{"start": 0.0, "end": 4.0, "text": "A B C D"},
                                     {"start": 7.0, "end": 10.0, "text": "real speech"}], record)
        _use_bin(monkeypatch, tmp_path / "bin")
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 40.0)
        segs = [_seg(3, 7, "A B C D")] + [_seg(10 + i, 11 + i, "Thank you.") for i in range(3)] + [_seg(30, 35, "outro")]
        loops = verify.find_repetition_runs(segs)

        out = local_whisper.repair_loops(segs, loops, audio, tmp_path / "mlx", duration_seconds=40.0)

        assert [s["text"] for s in out] == ["A B C D", "real speech", "outro"]
        assert out[0]["start"] == 3.0 and out[0]["end"] == 7.0
        # ffmpeg sliced exactly the widened window (3.0 → 18.0)
        assert abs(_duration(tmp_path / "mlx" / "loop_00.wav") - 15.0) < 0.1


# --- truncated windows: the per-window token cap ---------------------------------

TRUNC_TEXT = "पहला हिस्सा ठीक है लेकिन फिर�"


class TestSlicePlan:
    def test_thirty_seconds_is_three_ten_second_slices(self):
        assert local_whisper.slice_plan(0.0, 30.0) == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]

    def test_uneven_span_gets_equal_slices_under_the_limit(self):
        plan = local_whisper.slice_plan(5.0, 27.0)
        assert len(plan) == 3 and plan[0][0] == 5.0 and plan[-1][1] == 27.0
        assert all(end - start <= local_whisper.TRUNCATED_SLICE_SECONDS for start, end in plan)
        assert all(a[1] == b[0] for a, b in zip(plan, plan[1:]))

    def test_empty_span(self):
        assert local_whisper.slice_plan(3.0, 3.0) == []


class TestRepairTruncated:
    def test_truncated_window_is_resliced_with_overlap_and_words_kept_once(self, tmp_path: Path, monkeypatch):
        """A 0-30s segment ending in U+FFFD is re-decoded as three overlapping
        slices with word timestamps. Each slice also hears 0.75 s past each cut,
        so the word at a cut is half-heard by one slice and whole in the other;
        only the slice owning the word's midpoint keeps it: nothing is garbled
        and nothing is doubled."""
        def seg(words):
            return {"start": words[0][0], "end": words[-1][1], "text": "".join(w[2] for w in words),
                    "words": [{"start": a, "end": b, "word": t} for a, b, t in words]}
        record = tmp_path / "argv.json"
        _fake_mlx(tmp_path / "bin", [], record, by_name={
            # window 0.0-10.75: "ती" is the first half of the word at the 10 s cut
            "trunc_00_00": [seg([(0.5, 1.0, " एक"), (9.0, 9.6, " दो"), (10.2, 10.7, " ती")])],
            # window 9.25-20.75: hears "तीन" whole (source 9.85-10.65, midpoint 10.25)
            "trunc_00_01": [seg([(0.6, 1.4, " तीन"), (5.0, 6.0, " चार"), (10.9, 11.4, " पाँ")])],
            # window 19.25-30.75
            "trunc_00_02": [seg([(0.6, 1.5, " पाँच"), (5.0, 6.0, " छह")])],
        })
        _use_bin(monkeypatch, tmp_path / "bin")
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 36.0)  # a continuous tone: no pauses to snap to
        segs = [_seg(0.0, 30.0, TRUNC_TEXT), _seg(30.0, 35.0, "अंत")]
        truncated = verify.find_truncated(segs)

        out = local_whisper.repair_truncated(segs, truncated, audio, tmp_path / "mlx", duration_seconds=36.0)

        assert [s["text"] for s in out] == ["एक दो", "तीन चार", "पाँच छह", "अंत"]
        assert [(s["start"], s["end"]) for s in out[:3]] == [(0.5, 9.6), (9.85, 15.25), (19.85, 25.25)]
        assert verify.verify_segments(out, 36.0, check_window_length=True)["ok"] is True
        calls = [json.loads(line) for line in Path(str(record) + ".all").read_text().splitlines()]
        assert [Path(c[0]).name for c in calls] == ["trunc_00_00.wav", "trunc_00_01.wav", "trunc_00_02.wav"]
        assert all(c[c.index("--word-timestamps") + 1] == "True" for c in calls)
        assert all(c[c.index("--condition-on-previous-text") + 1] == "False" for c in calls)
        assert abs(_duration(tmp_path / "mlx" / "trunc_00_01.wav") - 11.5) < 0.1  # 10 s + 2 x 0.75 s
        assert abs(_duration(tmp_path / "mlx" / "trunc_00_00.wav") - 10.75) < 0.1  # clamped at 0 s
        assert abs(_duration(tmp_path / "mlx" / "trunc_00_02.wav") - 11.5) < 0.1  # past the outer 30 s edge too

    def test_cut_at_a_pause_needs_no_overlap(self, tmp_path: Path, monkeypatch):
        """Windows start and end exactly at a snapped pause (never mid-word)."""
        _fake_mlx(tmp_path / "bin", [{"start": 0.5, "end": 3.0, "text": "speech"}])
        _use_bin(monkeypatch, tmp_path / "bin")
        audio = tmp_path / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-t", "9.5", "-i", "sine=frequency=440:sample_rate=16000",
             "-f", "lavfi", "-t", "1", "-i", "anullsrc=r=16000:cl=mono",
             "-f", "lavfi", "-t", "9.5", "-i", "sine=frequency=440:sample_rate=16000",
             "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]", "-map", "[a]",
             "-acodec", "pcm_s16le", str(audio)], check=True)
        segs = [_seg(0.0, 20.0, TRUNC_TEXT)]
        out = local_whisper.repair_truncated(segs, segs, audio, tmp_path / "mlx", 20.0)
        assert [s["text"] for s in out] == ["speech", "speech"]
        assert abs(_duration(tmp_path / "mlx" / "trunc_00_00.wav") - 10.0) < 0.15
        assert abs(_duration(tmp_path / "mlx" / "trunc_00_01.wav") - 10.0) < 0.15

    def test_empty_slices_keep_original_so_warning_stays(self, tmp_path: Path, monkeypatch):
        _fake_mlx(tmp_path / "bin", [])
        _use_bin(monkeypatch, tmp_path / "bin")
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 31.0)
        segs = [_seg(0.0, 30.0, TRUNC_TEXT)]
        out = local_whisper.repair_truncated(segs, verify.find_truncated(segs), audio, tmp_path / "mlx", 31.0)
        assert out == segs
        assert verify.verify_segments(out, 31.0)["truncated"]

    def test_failed_slice_keeps_original(self, tmp_path: Path, monkeypatch):
        bin_dir = tmp_path / "bin"; bin_dir.mkdir()
        bad = bin_dir / "mlx_whisper"
        bad.write_text("#!/bin/sh\necho 'metal OOM' >&2\nexit 3\n"); bad.chmod(0o755)
        _use_bin(monkeypatch, bin_dir)
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 31.0)
        segs = [_seg(0.0, 30.0, TRUNC_TEXT)]
        assert local_whisper.repair_truncated(segs, verify.find_truncated(segs), audio, tmp_path / "mlx", 31.0) == segs


class TestTranscribeVideoTruncated:
    def test_mlx_path_repairs_truncation_end_to_end(self, tmp_path: Path, monkeypatch):
        import whisper
        _fake_mlx(tmp_path / "bin", [], by_name={
            "audio": [{"start": 0.0, "end": 30.0, "text": TRUNC_TEXT}, {"start": 30.0, "end": 34.0, "text": "अंत"}],
            "trunc_00_00": [{"start": 0.0, "end": 9.0, "text": "एक"}],
            "trunc_00_01": [{"start": 0.0, "end": 9.0, "text": "दो"}],
            "trunc_00_02": [{"start": 0.0, "end": 9.0, "text": "तीन"}],
        })
        _use_bin(monkeypatch, tmp_path / "bin")
        video = tmp_path / "v.mp4"
        _make_video(video, 35.0)
        segs, backend = whisper.transcribe_video(str(video), tmp_path / "audio.mp3", backend="mlx")
        assert backend == "mlx"
        assert [s["text"] for s in segs] == ["एक", "दो", "तीन", "अंत"]
        assert not any(s["text"].endswith("�") for s in segs)
        assert segs.quality["ok"] is True and segs.quality["truncated"] == []

    def test_unrepaired_truncation_is_reported(self, tmp_path: Path, monkeypatch, capsys):
        import whisper
        # Every decode, including the slices, comes back cut off.
        _fake_mlx(tmp_path / "bin", [{"start": 0.0, "end": 9.0, "text": "फिर�"}], by_name={
            "audio": [{"start": 0.0, "end": 30.0, "text": TRUNC_TEXT}]})
        _use_bin(monkeypatch, tmp_path / "bin")
        video = tmp_path / "v.mp4"
        _make_video(video, 31.0)
        segs, _ = whisper.transcribe_video(str(video), tmp_path / "audio.mp3", backend="mlx")
        assert segs.quality["ok"] is False and segs.quality["truncated"]
        assert "looks truncated" in capsys.readouterr().err


class TestSliceMerge:
    def test_keep_owned_by_word_midpoint(self):
        segs = [{"start": 8.0, "end": 10.7, "text": "a b c", "words": [
            {"start": 8.0, "end": 8.5, "text": " a"}, {"start": 9.0, "end": 9.9, "text": " b"},
            {"start": 9.9, "end": 10.7, "text": " c"}]}]
        assert local_whisper.keep_owned(segs, 0.0, 10.0, last=False) == [{"start": 8.0, "end": 9.9, "text": "a b"}]
        assert local_whisper.keep_owned(segs, 10.0, 20.0, last=True) == [{"start": 9.9, "end": 10.7, "text": "c"}]

    def test_keep_owned_without_words_uses_segment_midpoint(self):
        segs = [_seg(9.0, 10.6, "x"), _seg(10.6, 12.0, "y")]
        assert [s["text"] for s in local_whisper.keep_owned(segs, 0.0, 10.0, last=False)] == ["x"]  # mid 9.8
        assert [s["text"] for s in local_whisper.keep_owned(segs, 10.0, 20.0, last=False)] == ["y"]

    def test_last_slice_owns_its_end(self):
        assert local_whisper.keep_owned([_seg(29.0, 31.0, "z")], 20.0, 30.0, last=True)[0]["text"] == "z"
        assert local_whisper.keep_owned([_seg(29.0, 31.0, "z")], 20.0, 30.0, last=False) == []

    def test_snap_cuts_moves_to_nearest_pause(self):
        plan = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
        out = local_whisper.snap_cuts(plan, [(8.6, 9.2), (14.0, 15.0), (19.9, 20.5)])
        assert out == [(0.0, 8.9), (8.9, 20.2), (20.2, 30.0)]
        assert local_whisper.snap_cuts(plan, [(3.0, 4.0)]) == plan  # too far: keep nominal

    def test_snap_never_makes_a_slice_longer_than_the_cap(self):
        """The live DewiA failure: cuts 9.75 and 22.38 made a 12.6 s middle slice."""
        plan = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
        out = local_whisper.snap_cuts(plan, [(9.6, 9.9), (22.26, 22.5)])
        assert out == [(0.0, 9.75), (9.75, 20.0), (20.0, 30.0)]
        assert all(b - a <= local_whisper.MAX_SLICE_SECONDS for a, b in out)

    def test_detect_silences_on_real_audio(self, tmp_path: Path):
        audio = tmp_path / "gap.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-t", "4", "-i", "sine=frequency=440:sample_rate=16000",
             "-f", "lavfi", "-t", "1", "-i", "anullsrc=r=16000:cl=mono",
             "-f", "lavfi", "-t", "4", "-i", "sine=frequency=440:sample_rate=16000",
             "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]", "-map", "[a]",
             "-acodec", "pcm_s16le", str(audio)], check=True)
        silences = local_whisper.detect_silences(audio, 2.0, 9.0)
        assert len(silences) == 1
        assert abs(silences[0][0] - 4.0) < 0.15 and abs(silences[0][1] - 5.0) < 0.15

    def test_words_flag_parses_word_timestamps(self, tmp_path: Path, monkeypatch):
        _fake_mlx(tmp_path / "bin", [{"start": 0, "end": 1, "text": " hi there",
                                      "words": [{"start": 0, "end": 0.4, "word": " hi"},
                                                {"start": 0.5, "end": 1, "word": " there"}]}])
        _use_bin(monkeypatch, tmp_path / "bin")
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFF")
        seg = local_whisper.transcribe_mlx(audio, tmp_path / "mlx", words=True)[0]
        assert seg["words"] == [{"start": 0.0, "end": 0.4, "text": " hi"}, {"start": 0.5, "end": 1.0, "text": " there"}]
        assert "words" not in local_whisper.transcribe_mlx(audio, tmp_path / "mlx")[0]


class TestDecodeWindow:
    def test_edges_stay_put_inside_a_pause(self):
        assert local_whisper.decode_window(9.75, 20.0, [(9.6, 9.9), (19.8, 20.3)]) == (9.75, 20.05)

    def test_edges_move_out_to_a_nearby_pause(self):
        assert local_whisper.decode_window(10.0, 20.0, [(8.8, 9.2), (21.0, 21.4)]) == (9.0, 21.2)

    def test_no_pause_nearby_means_fixed_overlap(self):
        assert local_whisper.decode_window(10.0, 20.0, [(5.0, 6.0)]) == (9.25, 20.75)

    def test_clamped_to_audio(self):
        assert local_whisper.decode_window(0.0, 30.0, [], duration_seconds=30.2) == (0.0, 30.2)


class TestRepairCap:
    def test_truncated_windows_beyond_the_cap_stay_flagged(self, tmp_path: Path, monkeypatch, capsys):
        record = tmp_path / "argv.json"
        _fake_mlx(tmp_path / "bin", [{"start": 0.0, "end": 2.0, "text": "fixed"}], record)
        _use_bin(monkeypatch, tmp_path / "bin")
        monkeypatch.setattr(local_whisper, "MAX_REPAIR_WINDOWS", 2)
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 40.0)
        segs = [_seg(10.0 * i, 10.0 * i + 5.0, f"cut {i}\ufffd") for i in range(4)]
        out = local_whisper.repair_truncated(segs, verify.find_truncated(segs), audio, tmp_path / "mlx", 40.0)
        left = verify.verify_segments(out, 40.0)["truncated"]
        assert [t["start"] for t in left] == [20.0, 30.0]
        assert len(Path(str(record) + ".all").read_text().splitlines()) == 2  # 2 windows x 1 slice
        assert "2 more truncated segment window(s) left unrepaired" in capsys.readouterr().err

    def test_loop_repairs_are_capped_too(self, tmp_path: Path, monkeypatch, capsys):
        _fake_mlx(tmp_path / "bin", [{"start": 0.0, "end": 1.0, "text": "real"}])
        _use_bin(monkeypatch, tmp_path / "bin")
        monkeypatch.setattr(local_whisper, "MAX_REPAIR_WINDOWS", 1)
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 60.0)
        segs = ([_seg(i, i + 1, "x") for i in range(3)] + [_seg(10, 11, "gap")]
                + [_seg(30 + i, 31 + i, "y") for i in range(3)])
        loops = verify.find_repetition_runs(segs)
        out = local_whisper.repair_loops(segs, loops, audio, tmp_path / "mlx", 60.0)
        assert [l["text"] for l in verify.find_repetition_runs(out)] == ["y"]
        assert "1 more repetition loop window(s) left unrepaired" in capsys.readouterr().err
