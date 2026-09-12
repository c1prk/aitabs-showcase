"""Tests for rhythm grid quantization."""

from __future__ import annotations

import numpy as np

from aitabs.eval.metrics import rhythm_tick_f1
from aitabs.eval.types import EvalNote
from aitabs.pipeline.audio.pitch import NoteEvent
from aitabs.pipeline.audio.rhythm import (
    infer_beat_phase_template,
    quantize_notes_adaptive,
    quantize_notes_to_grid,
)
from aitabs.pipeline.audio.tempo import TempoAnalysis, onset_to_tick, snap_time_to_grid


def _analysis(bpm: float = 91.0, tpb: int = 2) -> TempoAnalysis:
    period = 60.0 / bpm
    return TempoAnalysis(
        bpm=bpm,
        beat_times=np.array([0.0, period, 2 * period]),
        beat_period=period,
        origin=0.0,
        ticks_per_beat=tpb,
    )


def test_quantize_collapses_jitter_to_one_tick():
    a = _analysis()
    eighth = a.tick_duration
    notes = [
        NoteEvent(start=0.02, end=0.2, pitch_midi=60, confidence=0.9),
        NoteEvent(start=0.05, end=0.2, pitch_midi=64, confidence=0.8),
        NoteEvent(start=eighth + 0.01, end=eighth + 0.3, pitch_midi=62, confidence=0.85),
    ]
    out = quantize_notes_to_grid(notes, a)
    ticks = {onset_to_tick(n["start"], a) for n in out}
    assert len(ticks) == 2
    assert len(out) == 3  # two pitches on first 8th, one on second


def test_max_one_note_per_tick():
    a = _analysis()
    notes = [
        NoteEvent(start=0.02, end=0.2, pitch_midi=60, confidence=0.9),
        NoteEvent(start=0.05, end=0.2, pitch_midi=64, confidence=0.5),
    ]
    out = quantize_notes_to_grid(notes, a, max_notes_per_tick=1)
    assert len(out) == 1
    assert out[0]["pitch_midi"] == 60


def test_adaptive_discovers_eighth_grid():
    a = _analysis(bpm=120, tpb=4)
    eighth = a.beat_period / 2
    onsets = np.array([0.0, eighth, a.beat_period, a.beat_period + eighth])
    template = infer_beat_phase_template(onsets, a)
    assert any(abs(p) < 0.05 or abs(p - 1.0) < 0.05 for p in template)
    assert any(abs(p - 0.5) < 0.08 for p in template)


def test_adaptive_discovers_syncopated_sixteenths():
    a = _analysis(bpm=91, tpb=4)
    beat = a.beat_period
    # 8th + 3 sixteenths within one beat (quarter = 1.0 beat fraction)
    onsets = np.array(
        [
            0.0,
            0.5 * beat,
            0.625 * beat,
            0.75 * beat,
            beat,
            beat + 0.5 * beat,
        ]
    )
    template = infer_beat_phase_template(onsets, a)
    assert any(abs(p - 0.5) < 0.06 for p in template)
    assert any(abs(p - 0.625) < 0.06 for p in template)


def test_quantize_adaptive_snaps_to_template():
    a = _analysis(bpm=120, tpb=4)
    beat = a.beat_period
    notes = [
        NoteEvent(start=0.02, end=0.2, pitch_midi=60, confidence=0.9),
        NoteEvent(start=0.5 * beat + 0.01, end=0.6 * beat, pitch_midi=62, confidence=0.9),
    ]
    out, adaptive = quantize_notes_adaptive(notes, a)
    assert adaptive.phase_template
    assert snap_time_to_grid(0.02, adaptive) == out[0]["start"]
    assert len(out) == 2


def test_rhythm_tick_f1_perfect():
    a = _analysis()
    ref = [
        EvalNote(start=0.0, end=0.3, pitch_midi=60, string=2, fret=0, confidence=1.0),
        EvalNote(start=a.tick_duration, end=a.tick_duration + 0.3, pitch_midi=62, string=2, fret=2, confidence=1.0),
    ]
    pred = list(ref)
    _, _, f1 = rhythm_tick_f1(ref, pred, a)
    assert f1 == 1.0
