"""Tests for DP string/fret mapping."""

from __future__ import annotations

from aitabs.pipeline.mapping.fingering import (
    _chord_group_epsilon_sec,
    map_sequence,
    transition_cost,
)
from aitabs.pipeline.mapping.guitar import fret_to_midi


def test_transition_cost_same_position():
    assert transition_cost((2, 5), (2, 5)) == 0.0


def test_transition_cost_same_string_different_fret():
    assert transition_cost((2, 5), (2, 7)) == 0.5


def test_transition_cost_string_change():
    assert transition_cost((2, 5), (3, 5)) == 1.0
    assert transition_cost((2, 5), (4, 5)) == 2.0


def test_map_sequence_empty():
    assert map_sequence([]) == []


def test_chord_group_epsilon_sec_sixteenth_at_120():
    assert _chord_group_epsilon_sec(120.0) == 0.125


def test_map_sequence_accepts_bpm():
    notes = [{"start": 0.0, "end": 0.5, "pitch_midi": 57, "confidence": 0.9}]
    tab = map_sequence(notes, bpm=120.0)
    assert len(tab) == 1


def test_map_sequence_reproduces_pitch():
    notes = [
        {"start": 0.0, "end": 0.5, "pitch_midi": 57, "confidence": 0.9},
        {"start": 1.0, "end": 1.5, "pitch_midi": 59, "confidence": 0.9},
        {"start": 2.0, "end": 2.5, "pitch_midi": 62, "confidence": 0.9},
    ]
    tab = map_sequence(notes)
    assert len(tab) == 3
    for t in tab:
        assert fret_to_midi(t["string"], t["fret"]) == t["pitch_midi"]
