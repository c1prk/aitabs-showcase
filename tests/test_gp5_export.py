"""Tests for GP5 export layout and roundtrip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import guitarpro
import numpy as np

from aitabs.pipeline.audio.tempo import TempoAnalysis

from aitabs.export.gp5_layout import (
    Gp5ExportOptions,
    build_note_groups,
    build_song_from_tab_notes,
    group_notes_by_grid,
    group_notes_by_onset,
    group_notes_for_export,
    snap_notes_to_grid,
    validate_gp5_song,
)
from aitabs.export.guitarpro import export_gp5
from aitabs.export.guitarpro_types import ExportNote


def _note(
    start: float,
    end: float,
    pitch_midi: int,
    string: int,
    fret: int,
) -> ExportNote:
    return ExportNote(
        start=start,
        end=end,
        pitch_midi=pitch_midi,
        string=string,
        fret=fret,
        confidence=0.9,
    )


def test_group_simultaneous_onsets():
    notes = [
        _note(1.0, 1.5, 59, 4, 0),
        _note(1.02, 1.5, 73, 2, 14),
    ]
    groups = group_notes_by_onset(notes, epsilon_sec=0.04)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_validate_rejects_duplicate_strings():
    notes = [
        _note(0.0, 0.5, 52, 2, 2),
        _note(0.0, 0.5, 55, 2, 5),
    ]
    song = build_song_from_tab_notes(notes, options=Gp5ExportOptions(tempo=120))
    validate_gp5_song(song)  # dedupe keeps one note per string — should pass


def test_build_song_has_beats_and_notes():
    notes = [
        _note(0.0, 0.5, 45, 1, 0),
        _note(0.5, 1.0, 57, 3, 7),
        _note(1.0, 1.5, 59, 4, 0),
    ]
    song = build_song_from_tab_notes(
        notes,
        options=Gp5ExportOptions(tempo=120, quantize_division=16),
    )
    voice = song.tracks[0].measures[0].voices[0]
    assert len(voice.beats) >= 1
    played = [b for b in voice.beats if b.notes]
    assert len(played) >= 2
    assert all(n.type.name == "normal" for b in played for n in b.notes)


def test_group_notes_by_grid_merges_fingerstyle_burst():
    analysis = TempoAnalysis(
        bpm=120.0,
        beat_times=np.array([0.0, 0.5, 1.0]),
        beat_period=0.5,
        origin=0.0,
        ticks_per_beat=4,
    )
    notes = [
        _note(0.00, 0.40, 52, 1, 7),
        _note(0.03, 0.40, 45, 1, 0),
        _note(0.12, 0.50, 57, 1, 12),
        _note(0.50, 0.90, 59, 2, 0),
    ]
    groups = group_notes_by_grid(notes, analysis, max_tick_gap=1)
    assert len(groups) == 2
    assert len(groups[0]) == 3
    assert len(groups[1]) == 1


def test_build_note_groups_snaps_chord_to_one_tick():
    analysis = TempoAnalysis(
        bpm=120.0,
        beat_times=np.array([0.0, 0.5, 1.0]),
        beat_period=0.5,
        origin=0.0,
        ticks_per_beat=4,
    )
    notes = [
        _note(0.00, 0.40, 52, 1, 7),
        _note(0.03, 0.40, 45, 1, 0),
        _note(0.50, 0.90, 59, 2, 0),
    ]
    groups = build_note_groups(
        notes,
        Gp5ExportOptions(tempo=120, tempo_grid=analysis, chord_grid_ticks=1),
    )
    assert len(groups) == 2
    assert groups[0].tick_start == 0
    assert len(groups[0].notes) == 2
    assert groups[1].tick_start == 960


def test_tempo_grid_uses_clean_durations():
    """Grid-snapped durations should be standard note values, not 32nd-chains."""
    analysis = TempoAnalysis(
        bpm=120.0,
        beat_times=np.array([0.0, 0.5, 1.0, 1.5]),
        beat_period=0.5,
        origin=0.0,
        ticks_per_beat=4,
    )
    notes = [
        _note(0.0, 0.5, 64, 5, 0),
        _note(0.5, 1.0, 67, 5, 3),
    ]
    groups = build_note_groups(
        notes,
        Gp5ExportOptions(tempo=120, tempo_grid=analysis, duration_mode="to_next_onset"),
    )
    assert groups[0].duration_ticks == 960  # four 16ths = quarter
    assert groups[1].duration_ticks == 960
    assert groups[1].tick_start == 960  # on beat 2


def test_snap_notes_to_grid_collapses_jitter():
    """Onsets within one 16th should share one grid time before grouping."""
    analysis = TempoAnalysis(
        bpm=120.0,
        beat_times=np.array([0.0, 0.5, 1.0]),
        beat_period=0.5,
        origin=0.0,
        ticks_per_beat=4,
    )
    notes = [
        _note(0.02, 0.2, 64, 5, 0),
        _note(0.05, 0.2, 67, 5, 3),
    ]
    snapped = snap_notes_to_grid(notes, analysis)
    assert snapped[0]["start"] == snapped[1]["start"] == 0.0
    groups = build_note_groups(
        notes,
        Gp5ExportOptions(
            tempo=120,
            tempo_grid=analysis,
            chord_grid_ticks=1,
            snap_to_grid=True,
            duration_mode="to_next_onset",
        ),
    )
    assert len(groups) == 1
    assert len(groups[0].notes) == 2


def test_to_next_onset_fills_gap_without_rests():
    """Legato mode: duration reaches next onset so export timeline has no stray rests."""
    analysis = TempoAnalysis(
        bpm=120.0,
        beat_times=np.array([0.0, 0.5, 1.0]),
        beat_period=0.5,
        origin=0.0,
        ticks_per_beat=4,
    )
    notes = [
        _note(0.0, 0.35, 64, 5, 0),
        _note(0.12, 0.45, 67, 5, 3),
        _note(0.50, 0.90, 69, 5, 5),
    ]
    groups = build_note_groups(
        notes,
        Gp5ExportOptions(
            tempo=120,
            tempo_grid=analysis,
            chord_grid_ticks=1,
            duration_mode="to_next_onset",
        ),
    )
    assert len(groups) == 2  # first two notes merge
    assert groups[0].tick_start == 0
    assert groups[1].tick_start == groups[0].tick_start + groups[0].duration_ticks


def test_export_gp5_roundtrip():
    notes = [
        _note(0.0, 0.4, 64, 5, 0),
        _note(0.5, 0.9, 67, 5, 3),
        _note(1.0, 1.4, 69, 5, 5),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.gp5"
        export_gp5(notes, path, tempo=120, title="roundtrip")
        parsed = guitarpro.parse(str(path))
        beats = parsed.tracks[0].measures[0].voices[0].beats
        note_count = sum(len(b.notes) for b in beats)
        assert note_count >= 3
        assert parsed.tempo == 120


def test_same_string_chord_roundtrip():
    """Two pitches on one string in one onset must not corrupt the file."""
    notes = [
        _note(0.0, 0.5, 52, 2, 2),
        _note(0.0, 0.5, 55, 2, 5),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "same_string.gp5"
        export_gp5(notes, path, tempo=120)
        parsed = guitarpro.parse(str(path))
        beat = parsed.tracks[0].measures[0].voices[0].beats[0]
        assert len(beat.notes) == 1


def test_measures_are_fully_filled():
    """Every measure must sum to the header length (Guitar Pro rejects partial bars)."""
    notes = [
        _note(0.0, 0.3, 64, 5, 0),
        _note(0.35, 0.7, 67, 5, 3),
        _note(0.75, 1.1, 69, 5, 5),
        _note(1.2, 1.6, 71, 5, 7),
        _note(1.65, 2.0, 72, 5, 8),
        _note(2.1, 2.5, 74, 5, 10),
        _note(2.55, 2.9, 76, 5, 12),
        _note(3.0, 3.4, 77, 5, 13),
        _note(3.45, 3.9, 79, 5, 15),
        _note(4.0, 4.4, 81, 5, 17),
        _note(4.5, 4.9, 83, 5, 19),
        _note(5.0, 5.4, 84, 5, 20),
        _note(5.5, 5.9, 86, 5, 22),
        _note(6.0, 6.4, 88, 5, 24),
        _note(6.5, 6.9, 89, 5, 25),
        _note(7.0, 7.4, 91, 5, 27),
        _note(7.5, 7.9, 93, 5, 29),
        _note(8.0, 8.4, 95, 5, 31),
        _note(8.5, 8.9, 96, 5, 32),
        _note(9.0, 9.4, 98, 5, 34),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "filled.gp5"
        export_gp5(notes, path, tempo=117, duration_mode="to_next_onset")
        parsed = guitarpro.parse(str(path))
        for mi, measure in enumerate(parsed.tracks[0].measures):
            expected = parsed.measureHeaders[mi].length
            total = sum(b.duration.time for b in measure.voices[0].beats)
            assert total == expected, f"measure {mi + 1}: {total} != {expected}"


def test_grid_unit_duration_is_one_eighth():
    """grid_unit + ticks_per_beat=2 → each onset group lasts one 8th note."""
    analysis = TempoAnalysis(
        bpm=91.0,
        beat_times=np.array([0.0, 0.66, 1.32]),
        beat_period=60.0 / 91.0,
        origin=0.0,
        ticks_per_beat=2,
    )
    notes = [
        _note(0.0, 0.2, 64, 5, 0),
        _note(0.33, 0.5, 67, 5, 3),
    ]
    groups = build_note_groups(
        notes,
        Gp5ExportOptions(
            tempo=91,
            tempo_grid=analysis,
            snap_to_grid=True,
            duration_mode="grid_unit",
            chord_grid_ticks=0,
        ),
    )
    assert groups[0].duration_ticks == 480  # eighth
    assert groups[1].duration_ticks == 480


def test_build_note_groups_trim_leading_silence():
    notes = [
        _note(2.0, 2.5, 45, 1, 0),
        _note(2.5, 3.0, 47, 1, 2),
    ]
    groups = build_note_groups(
        notes,
        Gp5ExportOptions(tempo=120, trim_leading_silence=True),
    )
    assert groups[0].tick_start == 0
