"""Step 1 rhythm fix: meter + tempo-map aware notation."""

from __future__ import annotations

from pathlib import Path

import pytest
from guitarpro.models import NoteType

from aitabs.eval.gp5_import import (
    load_gp5_notes,
    load_gp5_tempo_map,
    load_gp5_time_signature,
)
from aitabs.export.gp5_layout import Gp5ExportOptions, validate_gp5_song
from aitabs.export.guitarpro_types import ExportNote
from aitabs.export.rhythm_notation import build_song_from_note_values
from aitabs.pipeline.audio.tempo import TempoMap

# test1.gp5 = "One Summer's Day": 3/4, tempo 120 → 190 at measure 25.
TEST1 = Path("data/eval/reference/test1.gp5")


def test_tempo_map_conversion():
    tm = TempoMap(anchors=((0.0, 0.0, 120.0), (24.0, 12.0, 190.0)))
    assert abs(tm.seconds_to_beats(0.0)) < 1e-9
    assert abs(tm.seconds_to_beats(12.0) - 24.0) < 1e-9
    # one beat at 190 bpm past the change
    assert abs(tm.seconds_to_beats(12.0 + 60 / 190) - 25.0) < 1e-6
    assert abs(tm.beats_to_seconds(25.0) - (12.0 + 60 / 190)) < 1e-6


def test_tempo_map_constant():
    tm = TempoMap.constant(120.0)
    assert abs(tm.seconds_to_beats(1.0) - 2.0) < 1e-9  # 120 bpm → 2 beats/sec


@pytest.mark.skipif(not TEST1.is_file(), reason="reference gp5 missing")
def test_reference_meter_and_tempo_map():
    assert load_gp5_time_signature(TEST1) == (3, 4)
    tm = load_gp5_tempo_map(TEST1)
    assert any(abs(a[2] - 190.0) < 1e-6 for a in tm.anchors)


def _tie_fraction(song) -> float:
    total = tied = 0
    for measure in song.tracks[0].measures:
        for beat in measure.voices[0].beats:
            for note in beat.notes:
                total += 1
                if note.type == NoteType.tie:
                    tied += 1
    return tied / total if total else 0.0


@pytest.mark.skipif(not TEST1.is_file(), reason="reference gp5 missing")
def test_meter_and_tempo_map_fix_notation():
    notes, bpm, _setup = load_gp5_notes(TEST1)
    tm = load_gp5_tempo_map(TEST1)
    export_notes = [
        ExportNote(
            start=n["start"], end=n["end"], pitch_midi=n["pitch_midi"],
            string=n["string"], fret=n["fret"], confidence=1.0,
        )
        for n in notes
    ]

    def build(ts, tempo_map):
        opts = Gp5ExportOptions(
            tempo=int(bpm), time_signature=ts, duration_mode="note_value", tempo_map=tempo_map
        )
        return build_song_from_note_values(export_notes, options=opts)

    fixed = build((3, 4), tm)
    old = build((4, 4), None)
    validate_gp5_song(fixed)

    # Correct meter restored.
    assert fixed.measureHeaders[0].timeSignature.numerator == 3
    # Tempo-map quantization removes the tie/dot mush from the constant-tempo path.
    assert _tie_fraction(fixed) < 0.05
    assert _tie_fraction(fixed) < _tie_fraction(old)


def test_coarse_grid_absorbs_jitter_and_ghosts():
    from aitabs.export.rhythm_notation import TICKS_PER_QUARTER, quantize_onset_ticks

    tpq = TICKS_PER_QUARTER

    # Even eighths with ±jitter plus an off-beat ghost should stay on the 8th grid.
    jittered = [0 + 20, int(0.5 * tpq) - 25, tpq + 15, int(1.5 * tpq) + 30, int(0.15 * tpq)]
    out = quantize_onset_ticks(jittered, enable_triplets=False)
    fracs = {round((t % tpq) / tpq, 3) for t in out}
    assert fracs <= {0.0, 0.5}  # no spurious 16th/syncopation

    # Genuine sixteenths must still be notated as sixteenths.
    sixteenths = [0, int(0.25 * tpq), int(0.5 * tpq), int(0.75 * tpq)]
    out16 = quantize_onset_ticks(sixteenths, enable_triplets=False)
    fr16 = {round((t % tpq) / tpq, 3) for t in out16}
    assert 0.25 in fr16 and 0.75 in fr16
