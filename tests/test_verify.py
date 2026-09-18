"""Transcript verification: repetition loops, early stop, window splicing."""
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
        assert q == {"ok": True, "loops": [], "ends_early_seconds": 0.0}

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


class TestSpliceSegments:
    def test_replaces_window_and_keeps_order(self):
        segs = [_seg(0, 1, "a"), _seg(10, 11, "bad"), _seg(12, 13, "bad"), _seg(20, 21, "z")]
        fixed = [_seg(10.5, 12.5, "good")]
        out = verify.splice_segments(segs, 10.0, 13.0, fixed)
        assert [s["text"] for s in out] == ["a", "good", "z"]

    def test_window_bounds_are_inclusive_on_start(self):
        segs = [_seg(0, 1, "a"), _seg(5, 6, "b"), _seg(9, 10, "c")]
        out = verify.splice_segments(segs, 5.0, 9.0, [])
        assert [s["text"] for s in out] == ["a"]


class TestFilterQualityRange:
    def test_drops_loops_outside_focus_window(self):
        q = {"ok": False, "loops": [{"start": 10, "end": 20, "count": 3, "text": "x"},
                                    {"start": 100, "end": 110, "count": 4, "text": "y"}],
             "ends_early_seconds": 0.0}
        out = verify.filter_quality_range(q, 90.0, 120.0)
        assert [l["start"] for l in out["loops"]] == [100]
        assert out["ok"] is False
        assert verify.filter_quality_range(q, 30.0, 60.0)["ok"] is True
        assert verify.filter_quality_range(q, None, None) is q


class TestQualityWarningLines:
    def test_clean_is_empty(self):
        assert verify.quality_warning_lines({"ok": True, "loops": [], "ends_early_seconds": 0.0}, str) == []

    def test_loop_block_and_tail_note(self):
        q = {"ok": False, "loops": [{"start": 60, "end": 90, "count": 5, "text": "Thank you."}],
             "ends_early_seconds": 40.0}
        lines = verify.quality_warning_lines(q, lambda s: f"{int(s)}s")
        text = "\n".join(lines)
        assert "verification failed" in text and "60s → 90s" in text and "×5" in text
        assert "no speech was recognised in the last 40s" in text
