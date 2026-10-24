"""Transcript verification: repetition loops, truncated windows, early stop, splicing."""
from __future__ import annotations

import verify


# --- verification: the decoder's silent failure modes -------------------------

def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


class TestFindRepetitionRuns:
    def test_clean_transcript_has_no_runs(self):
        segs = [_seg(0, 1, "a"), _seg(1, 2, "b"), _seg(2, 3, "c")]
        assert verify.find_repetition_runs(segs) == []

    def test_two_repeats_is_not_a_loop(self):
        segs = [_seg(0, 1, "ok"), _seg(1, 2, "ok"), _seg(2, 3, "next")]
        assert verify.find_repetition_runs(segs) == []

    def test_three_repeats_is_a_loop_with_span(self):
        segs = [_seg(0, 1, "hi"), *(_seg(10 + i, 11 + i, "Thank you.") for i in range(5)), _seg(20, 21, "bye")]
        runs = verify.find_repetition_runs(segs)
        assert runs == [{"start": 10.0, "end": 15.0, "count": 5, "text": "Thank you."}]

    def test_loop_at_end_of_transcript_is_found(self):
        segs = [_seg(0, 1, "hi"), *(_seg(1 + i, 2 + i, "x") for i in range(3))]
        assert len(verify.find_repetition_runs(segs)) == 1

    def test_match_ignores_case_and_whitespace(self):
        segs = [_seg(0, 1, "Thank you."), _seg(1, 2, "thank  you."), _seg(2, 3, " THANK YOU. ")]
        assert len(verify.find_repetition_runs(segs)) == 1

    def test_empty(self):
        assert verify.find_repetition_runs([]) == []


class TestVerifySegments:
    def test_clean_full_coverage_is_ok(self):
        segs = [_seg(0, 5, "a"), _seg(5, 10, "b")]
        q = verify.verify_segments(segs, duration_seconds=12.0)
        assert q == {"ok": True, "loops": [], "truncated": [], "ends_early_seconds": 0.0}
        assert q == verify.CLEAN

    def test_loop_fails(self):
        segs = [_seg(0, 1, "a")] + [_seg(1 + i, 2 + i, "z") for i in range(3)]
        q = verify.verify_segments(segs, duration_seconds=4.0)
        assert q["ok"] is False
        assert len(q["loops"]) == 1

    def test_ends_early_is_advisory_not_failure(self):
        """Whisper emits nothing for outro music/silence, so a short tail is a
        note for the reader, not a verification failure."""
        segs = [_seg(0, 5, "a")]
        q = verify.verify_segments(segs, duration_seconds=100.0)
        assert q["ok"] is True
        assert q["ends_early_seconds"] == 95.0

    def test_text_only_fallback_segment_skips_tail_check(self):
        q = verify.verify_segments([_seg(0, 0, "whole text")], duration_seconds=100.0)
        assert q["ends_early_seconds"] == 0.0

    def test_tail_limit_is_respected(self):
        segs = [_seg(0, 5, "a")]
        assert verify.verify_segments(segs, duration_seconds=30.0, tail_limit=30.0)["ok"] is True

    def test_unknown_duration_skips_tail_check(self):
        segs = [_seg(0, 5, "a")]
        assert verify.verify_segments(segs, duration_seconds=0.0)["ok"] is True


class TestFindTruncated:
    def test_replacement_char_ending_is_truncated(self):
        """The observed Hindi case: the cap cut a Devanagari character in half."""
        segs = [_seg(0.0, 12.0, "नमस्ते दोस्तों आज हम बात कर\ufffd"), _seg(30.0, 34.0, "अगला")]
        assert verify.find_truncated(segs) == [{"start": 0.0, "end": 12.0, "text": segs[0]["text"]}]

    def test_whole_window_segment_is_truncated(self):
        segs = [_seg(0.0, 30.0, "one long cue with no closing timestamp"), _seg(30.0, 33.0, "next")]
        assert [t["start"] for t in verify.find_truncated(segs)] == [0.0]

    def test_threshold_is_named_and_inclusive(self):
        limit = verify.TRUNCATED_MIN_SECONDS
        assert 25.0 < limit < 30.0
        assert verify.find_truncated([_seg(10.0, 10.0 + limit, "x")])
        assert verify.find_truncated([_seg(10.0, 10.0 + limit - 0.5, "x")]) == []

    def test_normal_speech_is_not_truncated(self):
        segs = [_seg(0, 6, "hello there"), _seg(6, 14.5, "a longer sentence, still fine"), _seg(15, 16, "ok.")]
        assert verify.find_truncated(segs) == []

    def test_window_length_rule_is_mlx_only(self):
        """WhisperX (--no_align) reports merged VAD chunks of up to 30 s as
        segments; those are normal, so only mlx gets the duration rule."""
        whisperx_like = [_seg(0.5, 29.8, "a long merged chunk"), _seg(30.1, 59.6, "another one")]
        assert verify.verify_segments(whisperx_like, 60.0)["ok"] is True
        cut = [_seg(0.5, 29.8, "a long merged chunk\ufffd"), _seg(30.1, 59.6, "another one")]
        q = verify.verify_segments(cut, 60.0)
        assert q["ok"] is False and [t["start"] for t in q["truncated"]] == [0.5]
        mlx = [_seg(0.0, 30.0, "no closing timestamp"), _seg(30.0, 33.0, "next")]
        assert verify.verify_segments(mlx, 35.0)["ok"] is True
        assert [t["start"] for t in verify.verify_segments(mlx, 35.0, check_window_length=True)["truncated"]] == [0.0]

    def test_truncated_fails_verification(self):
        segs = [_seg(0.0, 30.0, "cut off \ufffd"), _seg(30.0, 35.0, "rest")]
        q = verify.verify_segments(segs, duration_seconds=35.0)
        assert q["ok"] is False and q["loops"] == []
        assert q["truncated"] == [{"start": 0.0, "end": 30.0, "text": "cut off \ufffd"}]


class TestSpliceSegments:
    def test_replaces_window_and_keeps_order(self):
        segs = [_seg(0, 1, "a"), _seg(10, 11, "bad"), _seg(12, 13, "bad"), _seg(20, 21, "z")]
        fixed = [_seg(10.5, 12.5, "good")]
        out = verify.splice_segments(segs, 10.0, 13.0, fixed)
        assert [s["text"] for s in out] == ["a", "good", "z"]

    def test_drops_segments_overlapping_the_window(self):
        """A segment that merely touches the boundary (ends at lo / starts at hi) is kept."""
        segs = [_seg(0, 5, "a"), _seg(5, 6, "b"), _seg(9, 10, "c")]
        out = verify.splice_segments(segs, 5.0, 9.0, [])
        assert [s["text"] for s in out] == ["a", "c"]

    def test_straddling_segment_is_replaced_whole(self):
        """A segment overlapping the window is dropped in full — the caller
        (local_whisper.retry_window) guarantees the retry covers its whole span."""
        segs = [_seg(3, 7, "before loop"), _seg(10, 11, "x"), _seg(11, 12, "x"), _seg(12, 13, "x")]
        redo = [_seg(3, 7, "before loop again"), _seg(10, 13, "real speech")]
        out = verify.splice_segments(segs, 3.0, 18.0, redo)
        assert [s["text"] for s in out] == ["before loop again", "real speech"]


class TestFilterQualityRange:
    def test_drops_truncated_outside_focus_window(self):
        q = {"ok": False, "loops": [], "truncated": [{"start": 0.0, "end": 30.0, "text": "x\ufffd"}],
             "ends_early_seconds": 0.0}
        assert verify.filter_quality_range(q, 10.0, 20.0)["truncated"] == q["truncated"]
        assert verify.filter_quality_range(q, 10.0, 20.0)["ok"] is False
        out = verify.filter_quality_range(q, 40.0, 50.0)
        assert out["ok"] is True and out["truncated"] == []

    def test_drops_loops_outside_focus_window(self):
        q = {"ok": False, "loops": [{"start": 10, "end": 20, "count": 3, "text": "x"},
                                    {"start": 100, "end": 110, "count": 4, "text": "y"}],
             "truncated": [], "ends_early_seconds": 0.0}
        out = verify.filter_quality_range(q, 90.0, 120.0)
        assert [l["start"] for l in out["loops"]] == [100]
        assert out["ok"] is False
        assert verify.filter_quality_range(q, 30.0, 60.0)["ok"] is True
        assert verify.filter_quality_range(q, None, None) is q


class TestQualityWarningLines:
    def test_clean_is_empty(self):
        assert verify.quality_warning_lines(verify.CLEAN, str) == []

    def test_loop_block_and_tail_note(self):
        q = {"ok": False, "loops": [{"start": 60, "end": 90, "count": 5, "text": "Thank you."}],
             "ends_early_seconds": 40.0}
        lines = verify.quality_warning_lines(q, lambda s: f"{int(s)}s")
        text = "\n".join(lines)
        assert "verification failed" in text and "60s → 90s" in text and "×5" in text
        assert "no speech was recognised in the last 40s" in text

    def test_truncated_segment_is_listed(self):
        q = {"ok": False, "loops": [], "truncated": [{"start": 0.0, "end": 30.0, "text": "x\ufffd"}],
             "ends_early_seconds": 0.0}
        text = "\n".join(verify.quality_warning_lines(q, lambda s: f"{int(s)}s"))
        assert "verification failed" in text and "truncated segment 0s → 30s" in text


class TestReportQuality:
    def test_truncated_is_reported_on_stderr(self, capsys):
        verify.report_quality({"ok": False, "loops": [], "truncated": [{"start": 0.0, "end": 30.0, "text": "x"}],
                               "ends_early_seconds": 0.0}, "mlx")
        err = capsys.readouterr().err
        assert "WARNING" in err and "0s-30s looks truncated" in err


class TestClampEnds:
    def test_end_past_next_start_is_clamped(self):
        segs = [_seg(10.0, 20.05, "a"), _seg(19.99, 25.0, "b"), _seg(25.0, 26.0, "c")]
        out = verify.clamp_ends(segs)
        assert [(s["start"], s["end"]) for s in out] == [(10.0, 19.99), (19.99, 25.0), (25.0, 26.0)]
        assert segs[0]["end"] == 20.05  # input untouched

    def test_never_ends_before_it_starts(self):
        out = verify.clamp_ends([_seg(5.0, 9.0, "a"), _seg(5.0, 6.0, "b")])
        assert out[0]["end"] == 5.0

    def test_splice_clamps_the_seam(self):
        out = verify.splice_segments([_seg(0, 10, "x\ufffd"), _seg(20.0, 22.0, "next")], 0, 10,
                                     [_seg(1.0, 9.6, "fixed"), _seg(9.5, 20.05, "tail")])
        assert [(s["start"], s["end"]) for s in out] == [(1.0, 9.5), (9.5, 20.0), (20.0, 22.0)]
