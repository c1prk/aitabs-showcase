"""Tests for false-positive categorization."""

from __future__ import annotations

from aitabs.eval.error_analysis import categorize_false_positives
from aitabs.eval.types import EvalNote


def _n(start: float, pitch: int, *, dur: float = 0.4) -> EvalNote:
    return EvalNote(start=start, end=start + dur, pitch_midi=pitch, string=0, fret=0, confidence=1.0)


def test_octave_false_positive():
    ref = [_n(1.0, 40)]
    pred = [_n(1.0, 40), _n(1.0, 52)]  # match + octave ghost
    bd = categorize_false_positives(ref, pred, estimate_offset=False)
    assert bd.n_matched == 1 and bd.n_false == 1
    assert bd.octave == 1


def test_fifth_and_duplicate_and_isolated():
    ref = [_n(1.0, 40), _n(2.0, 50)]
    pred = [
        _n(1.0, 40),   # match
        _n(1.0, 47),   # +7 fifth over 40
        _n(2.0, 50),   # match
        _n(2.3, 50),   # same pitch, mistimed -> duplicate/sustain (concurrent via overlap)
        _n(5.0, 60),   # nothing sounds here -> isolated
    ]
    bd = categorize_false_positives(ref, pred, estimate_offset=False, concurrency_sec=0.07)
    assert bd.n_matched == 2
    assert bd.fifth_or_twelfth == 1
    assert bd.duplicate_sustain == 1
    assert bd.isolated == 1


def test_other_concurrent_wrong_note():
    ref = [_n(1.0, 40)]
    pred = [_n(1.0, 40), _n(1.0, 43)]  # +3 minor third = not harmonic
    bd = categorize_false_positives(ref, pred, estimate_offset=False)
    assert bd.other_concurrent == 1
    assert bd.octave == 0


def test_no_false_positives():
    ref = [_n(1.0, 40), _n(2.0, 50)]
    bd = categorize_false_positives(ref, list(ref), estimate_offset=False)
    assert bd.n_false == 0
