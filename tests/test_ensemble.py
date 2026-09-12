"""Tests for multi-detector note reconciliation (ensemble)."""

from __future__ import annotations

from aitabs.pipeline.audio.pitch import NoteEvent
from aitabs.pipeline.audio.ensemble import reconcile_notes


def _n(start: float, pitch: int, conf: float, dur: float = 0.4) -> NoteEvent:
    return NoteEvent(start=start, end=start + dur, pitch_midi=pitch, confidence=conf)


def _pitches(notes):
    return sorted(n["pitch_midi"] for n in notes)


def test_agreement_is_matched_once():
    # Same pitch within tolerance → one merged note, not two.
    kong = [_n(1.0, 40, 0.8)]
    bp = [_n(1.02, 40, 0.6)]
    out = reconcile_notes(kong, bp, mode="union")
    assert len(out) == 1
    assert out[0]["source"] == "both"


def test_agreement_keeps_primary_timing_and_boosts_conf():
    kong = [_n(1.0, 40, 0.8)]
    bp = [_n(1.03, 40, 0.6)]
    out = reconcile_notes(kong, bp, mode="union", agreement_bonus=0.15)
    assert out[0]["start"] == 1.0  # primary (Kong) timing kept
    assert out[0]["confidence"] > 0.8  # agreement raised confidence


def test_intersection_keeps_only_agreed():
    kong = [_n(1.0, 40, 0.8), _n(2.0, 45, 0.9)]
    bp = [_n(1.02, 40, 0.6)]  # only agrees on pitch 40
    out = reconcile_notes(kong, bp, mode="intersection")
    assert _pitches(out) == [40]


def test_union_keeps_all():
    kong = [_n(1.0, 40, 0.8)]
    bp = [_n(2.0, 45, 0.6)]  # disjoint
    out = reconcile_notes(kong, bp, mode="union")
    assert _pitches(out) == [40, 45]


def test_gated_drops_weak_singletons_keeps_agreed():
    kong = [
        _n(1.0, 40, 0.8),   # agreed → kept regardless
        _n(2.0, 45, 0.3),   # weak singleton → dropped at min_conf 0.5
        _n(3.0, 50, 0.9),   # strong singleton → kept
    ]
    bp = [_n(1.02, 40, 0.7)]
    out = reconcile_notes(kong, bp, mode="gated", min_singleton_conf=0.5)
    assert _pitches(out) == [40, 50]


def test_augment_keeps_all_primary_gates_secondary():
    # augment: keep ALL Kong (even weak), add BP only if >= min_singleton_conf.
    kong = [
        _n(1.0, 40, 0.8),   # agreed
        _n(2.0, 45, 0.3),   # weak Kong singleton -> KEPT (never drop primary)
        _n(3.0, 50, 0.9),
    ]
    bp = [
        _n(1.02, 40, 0.7),  # agreed
        _n(4.0, 60, 0.9),   # strong BP singleton -> added
        _n(5.0, 62, 0.3),   # weak BP singleton -> dropped
    ]
    out = reconcile_notes(kong, bp, mode="augment", min_singleton_conf=0.5)
    assert _pitches(out) == [40, 45, 50, 60]


def test_gated_keeps_strong_secondary_singleton():
    # A note only BasicPitch found, but confident → recovered (the recall win).
    kong: list[NoteEvent] = []
    bp = [_n(1.0, 62, 0.9)]
    out = reconcile_notes(kong, bp, mode="gated", min_singleton_conf=0.5)
    assert _pitches(out) == [62]
    assert out[0]["source"] == "secondary"


def test_different_pitch_same_time_not_merged():
    kong = [_n(1.0, 40, 0.8)]
    bp = [_n(1.0, 47, 0.6)]  # simultaneous but different pitch (chord)
    out = reconcile_notes(kong, bp, mode="union")
    assert _pitches(out) == [40, 47]


def test_onset_outside_tolerance_not_merged():
    kong = [_n(1.0, 40, 0.8)]
    bp = [_n(1.2, 40, 0.6)]  # same pitch, 200 ms later → separate note
    out = reconcile_notes(kong, bp, mode="union", onset_tolerance_sec=0.05)
    assert len(out) == 2


def test_empty_inputs():
    assert reconcile_notes([], [], mode="union") == []
    assert reconcile_notes([], [], mode="gated") == []
