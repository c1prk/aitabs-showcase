

from __future__ import annotations

import numpy as np

from aitabs.pipeline.audio.pitch import NoteEvent, clean_notes
from aitabs.pipeline.audio.tempo import (
    TempoAnalysis,
    classify_same_pitch_pair,
    clean_notes_with_tempo,
    onset_to_tick,
)


def _analysis(bpm: float = 120.0, origin: float = 0.0) -> TempoAnalysis:
    beat_period = 60.0 / bpm
    beats = origin + np.arange(0, 10) * beat_period
    return TempoAnalysis(
        bpm=bpm,
        beat_times=beats,
        beat_period=beat_period,
        origin=origin,
        ticks_per_beat=4,
        confidence=0.9,
    )


def test_classify_duplicate_within_merge_window():
    ta = _analysis()
    assert classify_same_pitch_pair(1.0, 1.1, ta) == "duplicate"


def test_classify_intentional_repeat_far_apart():
    ta = _analysis()
    assert classify_same_pitch_pair(2.0, 6.15, ta) == "intentional"


def test_classify_a4_double_at_120bpm():
    ta = _analysis(120.0, 0.0)
    # User clip: 5.74 and 5.91 — should merge as duplicate
    assert classify_same_pitch_pair(5.74, 5.91, ta, pitch_merge_sec=0.15) == "duplicate"


def test_clean_notes_tempo_merges_opening_doubles():
    notes = [
        NoteEvent(start=0.05, end=0.5, pitch_midi=52, confidence=0.9),
        NoteEvent(start=0.14, end=0.5, pitch_midi=52, confidence=0.7),
        NoteEvent(start=6.15, end=6.5, pitch_midi=45, confidence=0.9),
        NoteEvent(start=2.01, end=2.5, pitch_midi=45, confidence=0.9),
    ]
    ta = _analysis()
    cleaned = clean_notes_with_tempo(notes, ta)
    assert len(cleaned) == 3
    assert clean_notes(notes, pitch_merge_sec=0.15)  # still runs


def test_onset_to_tick_increases_with_time():
    ta = _analysis(120.0)
    assert onset_to_tick(0.0, ta) < onset_to_tick(1.0, ta)
