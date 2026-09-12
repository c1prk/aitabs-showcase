"""Tests for octave/harmonic ghost suppression."""

from __future__ import annotations

from aitabs.pipeline.audio.pitch import NoteEvent, suppress_octave_harmonics


def _n(start: float, pitch: int, conf: float, dur: float = 0.4) -> NoteEvent:
    return NoteEvent(start=start, end=start + dur, pitch_midi=pitch, confidence=conf)


def _pitches(notes):
    return sorted(n["pitch_midi"] for n in notes)


def test_coincident_weaker_octave_is_dropped():
    # E2 (40) fundamental + a weaker octave ghost E3 (52) at the same onset.
    notes = [_n(1.0, 40, 0.9), _n(1.01, 52, 0.6)]
    out = suppress_octave_harmonics(notes)
    assert _pitches(out) == [40]


def test_octave_fifth_and_two_octaves_dropped():
    notes = [
        _n(0.0, 40, 0.9),
        _n(0.0, 59, 0.5),  # +19 (octave + fifth)
        _n(0.0, 64, 0.5),  # +24 (two octaves)
    ]
    out = suppress_octave_harmonics(notes)
    assert _pitches(out) == [40]


def test_separately_played_octave_is_kept():
    # Same pitches but the upper note has its own, later onset → real note.
    notes = [_n(1.0, 40, 0.9), _n(1.6, 52, 0.6)]
    out = suppress_octave_harmonics(notes)
    assert _pitches(out) == [40, 52]


def test_stronger_upper_note_is_kept():
    # If the "harmonic" is stronger than the candidate fundamental, keep it.
    notes = [_n(1.0, 40, 0.4), _n(1.0, 52, 0.9)]
    out = suppress_octave_harmonics(notes)
    assert _pitches(out) == [40, 52]


def test_non_harmonic_interval_kept():
    # A major third (40, 44) is not a harmonic interval → both kept.
    notes = [_n(1.0, 40, 0.9), _n(1.0, 44, 0.6)]
    out = suppress_octave_harmonics(notes)
    assert _pitches(out) == [40, 44]


def test_conf_ratio_protects_real_octave_double():
    # An octave double of comparable strength survives a stricter ratio.
    notes = [_n(1.0, 40, 0.9), _n(1.0, 52, 0.85)]
    strict = suppress_octave_harmonics(notes, max_conf_ratio=0.7)
    assert _pitches(strict) == [40, 52]
    loose = suppress_octave_harmonics(notes, max_conf_ratio=1.0)
    assert _pitches(loose) == [40]


def test_empty_input():
    assert suppress_octave_harmonics([]) == []
