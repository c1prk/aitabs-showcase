"""Tests for GP5 import and eval metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from aitabs.eval.gp5_import import load_gp5_notes, load_guitar_setup
from aitabs.eval.metrics import compare_note_lists, estimate_time_offset
from aitabs.eval.types import EvalNote


GP5_SAMPLE = Path("audio_recording/output/test_audio_3_offset5s_10s.gp5")
# Committed reference with a mid-song tempo change (120 → 190 BPM at measure 25).
TEMPO_CHANGE_GP5 = Path("data/eval/reference/test1.gp5")


def _note(
    start: float,
    end: float,
    pitch: int,
    string: int,
    fret: int,
) -> EvalNote:
    return EvalNote(
        start=start,
        end=end,
        pitch_midi=pitch,
        string=string,
        fret=fret,
        confidence=1.0,
    )


@pytest.mark.skipif(not GP5_SAMPLE.is_file(), reason="sample gp5 missing")
def test_load_gp5_notes_roundtrip_pitch():
    notes, bpm, setup = load_gp5_notes(GP5_SAMPLE)
    assert bpm > 0
    assert len(notes) > 0
    for n in notes:
        assert setup.written_to_concert_midi(n["string"], n["fret"]) == n["pitch_midi"]


@pytest.mark.skipif(not TEMPO_CHANGE_GP5.is_file(), reason="reference gp5 missing")
def test_load_gp5_honours_tempo_change():
    """Notes after a tempo speed-up must be timed faster, not at the header tempo.

    Regression for the constant-tempo bug: test1.gp5 jumps 120→190 BPM at
    measure 25, so the true timeline is much shorter than a flat-120 reading.
    """
    from guitarpro import Duration
    import guitarpro

    notes, header_bpm, _ = load_gp5_notes(TEMPO_CHANGE_GP5)
    assert header_bpm == 120.0
    assert notes, "expected notes"

    # Flat-120 duration (the old, buggy reading) from total ticks.
    song = guitarpro.parse(str(TEMPO_CHANGE_GP5))
    total_ticks = sum(
        beat.duration.time
        for measure in song.tracks[0].measures
        for beat in measure.voices[0].beats
    )
    flat_120_sec = total_ticks * 60.0 / (120.0 * Duration.quarterTime)

    # The tempo speeds up, so the real end must be clearly shorter than flat-120.
    assert notes[-1]["end"] < 0.85 * flat_120_sec


def test_compare_perfect_match():
    ref = [_note(0.0, 0.5, 64, 5, 0), _note(0.5, 1.0, 67, 5, 3)]
    pred = list(ref)
    scores = compare_note_lists(
        ref,
        pred,
        clip_id="test",
        estimate_offset=False,
        time_offset_sec=0.0,
    )
    assert scores.pitch_f1 == 1.0
    assert scores.tab_f1 == 1.0


def test_estimate_time_offset():
    ref = [
        _note(1.0, 1.5, 60, 2, 0),
        _note(2.0, 2.5, 62, 2, 2),
    ]
    pred = [
        _note(0.0, 0.5, 60, 2, 0),
        _note(1.0, 1.5, 62, 2, 2),
    ]
    off = estimate_time_offset(ref, pred, max_shift_sec=2.0, step_sec=0.01)
    assert abs(off - 1.0) < 0.03
