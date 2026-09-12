"""Tests for onset rhythm-structure diagnostics (aitabs.eval.rhythm_structure)."""

from __future__ import annotations

from aitabs.eval.rhythm_structure import (
    classify_ioi,
    collapse_onsets,
    diagnose,
    onset_iois_in_beats,
    profile_onsets,
)


def _notes(starts: list[float], dur: float = 0.1) -> list[dict]:
    return [
        {"start": s, "end": s + dur, "pitch_midi": 60, "string": 0,
         "fret": 0, "confidence": 1.0}
        for s in starts
    ]


def test_classify_ioi_buckets():
    assert classify_ioi(1.0) == "quarter"
    assert classify_ioi(0.5) == "eighth"
    assert classify_ioi(0.25) == "sixteenth"
    assert classify_ioi(0.75) == "dotted-eighth"
    assert classify_ioi(1 / 3) == "eighth-triplet"
    assert classify_ioi(0.12) == "finer"


def test_collapse_onsets_merges_chord():
    # Three near-simultaneous attacks (a chord) collapse to one onset.
    onsets = collapse_onsets(_notes([1.00, 1.01, 1.02, 1.60]), chord_window_sec=0.05)
    assert onsets == [1.00, 1.60]


def test_iois_in_beats_at_known_tempo():
    # 120 BPM -> 0.5 s = 1 beat. Eighth notes (0.25 s) -> 0.5 beat IOIs.
    iois = onset_iois_in_beats(_notes([0.0, 0.25, 0.5, 0.75]), bpm=120.0)
    assert all(abs(x - 0.5) < 1e-9 for x in iois)


def test_profile_flags_sub_eighth():
    # 120 BPM: 0.125 s spacing = 0.25 beat = sixteenths -> 100% sub-eighth.
    p = profile_onsets(_notes([i * 0.125 for i in range(8)]), bpm=120.0)
    assert p.sub_eighth_fraction == 1.0
    assert p.counts.get("sixteenth", 0) >= 6


def test_diagnose_detector_recall_when_pred_lacks_sixteenths():
    bpm = 120.0
    ref = _notes([i * 0.125 for i in range(16)])          # all sixteenths
    pred = _notes([i * 0.25 for i in range(8)])           # only eighths detected
    verdict, p, r = diagnose(pred, ref, bpm)
    assert "DETECTOR/RECALL" in verdict
    assert r.sub_eighth_fraction > 0.5 and p.sub_eighth_fraction < 0.1


def test_diagnose_notation_flatten_when_pred_has_sixteenths():
    bpm = 120.0
    ref = _notes([i * 0.125 for i in range(16)])          # sixteenths
    pred = _notes([i * 0.125 + (0.01 if i % 2 else 0) for i in range(16)])  # also sixteenths
    verdict, p, r = diagnose(pred, ref, bpm)
    assert "NOTATION-FLATTEN" in verdict
    assert p.sub_eighth_fraction > 0.5


def test_diagnose_reference_is_all_eighths():
    bpm = 120.0
    ref = _notes([i * 0.25 for i in range(8)])
    pred = _notes([i * 0.25 for i in range(8)])
    verdict, _, _ = diagnose(pred, ref, bpm)
    assert "rhythmically faithful" in verdict


def test_diagnose_no_reference():
    bpm = 120.0
    pred = _notes([i * 0.25 for i in range(8)])
    verdict, p, r = diagnose(pred, None, bpm)
    assert r is None
    assert "recall problem upstream" in verdict
