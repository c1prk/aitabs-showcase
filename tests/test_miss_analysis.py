"""Tests for false-negative (miss) categorization."""

from __future__ import annotations

from aitabs.eval.error_analysis import categorize_false_negatives
from aitabs.eval.types import EvalNote


def _n(start: float, pitch: int, *, dur: float = 0.4) -> EvalNote:
    return EvalNote(start=start, end=start + dur, pitch_midi=pitch, string=0, fret=0, confidence=1.0)


def test_chord_inner_and_isolated():
    ref = [_n(1, 60), _n(1, 64), _n(1, 67), _n(2, 72)]
    pred = [_n(1, 60)]  # caught one chord tone; missed 64,67 (inner) and 72 (isolated)
    bd = categorize_false_negatives(ref, pred, estimate_offset=False)
    assert bd.n_missed == 3
    assert bd.chord_inner == 2
    assert bd.isolated == 1


def test_masked_by_sustain():
    ref = [_n(0.0, 40, dur=2.0), _n(1.0, 67, dur=0.2)]  # bass rings under missed melody note
    pred = [_n(0.0, 40, dur=2.0)]
    bd = categorize_false_negatives(ref, pred, estimate_offset=False)
    assert bd.masked_by_sustain == 1


def test_low_register_miss():
    # isolated missed bass note, nothing concurrent/sustaining
    ref = [_n(0.0, 45, dur=0.2), _n(3.0, 64, dur=0.2)]
    pred = [_n(3.0, 64, dur=0.2)]
    bd = categorize_false_negatives(ref, pred, estimate_offset=False)
    assert bd.low_register == 1


def test_no_misses():
    ref = [_n(1, 60), _n(2, 64)]
    bd = categorize_false_negatives(ref, list(ref), estimate_offset=False)
    assert bd.n_missed == 0
