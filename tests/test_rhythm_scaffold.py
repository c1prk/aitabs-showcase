"""Tests for reference rhythm scaffold export."""

from __future__ import annotations

import tempfile
from pathlib import Path

import guitarpro
import pytest

from aitabs.export.gp5_layout import Gp5ExportOptions
from guitarpro.models import BeatStatus

from aitabs.export.rhythm_scaffold import (
    assign_notes_to_scaffold,
    build_song_from_scaffold,
    export_gp5_with_reference_rhythm,
    load_rhythm_scaffold,
)
from aitabs.export.guitarpro_types import ExportNote

REF = Path("data/eval/reference/test2.gp5")


def _beat_durs(path: str, n_meas: int = 1) -> list[int]:
    song = guitarpro.parse(path)
    durs = []
    for beat in song.tracks[0].measures[0].voices[0].beats:
        durs.append(int(beat.duration.value))
    return durs[:20]


@pytest.mark.skipif(not REF.is_file(), reason="test2.gp5 missing")
def test_scaffold_matches_reference_measure1_durations():
    slots, _, _ = load_rhythm_scaffold(REF)
    m1 = [s for s in slots if s.measure_index == 0]
    durs = [s.duration_value for s in m1[:13]]
    assert durs[0] == 8
    assert durs[1:4] == [16, 16, 16]


@pytest.mark.skipif(not REF.is_file(), reason="test2.gp5 missing")
def test_export_with_scaffold_preserves_rhythm_pattern():
    slots, bpm, ts = load_rhythm_scaffold(REF)
    fake_notes = [
        ExportNote(start=s.start_sec, end=s.start_sec + 0.1, pitch_midi=60, string=2, fret=0, confidence=0.9)
        for s in slots
        if not s.is_rest
    ][:20]
    assignments = assign_notes_to_scaffold(fake_notes, slots, max_snap_sec=0.2)
    with tempfile.TemporaryDirectory() as tmp:
        song = build_song_from_scaffold(
            slots,
            assignments,
            options=Gp5ExportOptions(tempo=int(round(bpm)), time_signature=ts),
        )
        out = Path(tmp) / "scaffold.gp5"
        song.versionTuple = (5, 1, 0)
        guitarpro.write(song, str(out), version=(5, 1, 0))
        assert _beat_durs(str(out)) == _beat_durs(str(REF))


def _rest_count(path: str, n_meas: int = 2) -> int:
    song = guitarpro.parse(path)
    total = 0
    for mi in range(n_meas):
        for beat in song.tracks[0].measures[mi].voices[0].beats:
            if beat.status == BeatStatus.rest or not beat.notes:
                total += 1
    return total


@pytest.mark.skipif(not REF.is_file(), reason="test2.gp5 missing")
def test_assign_never_substitutes_or_duplicates_pitches():
    """Each detection may appear at most once; never borrow another note for a slot."""
    from collections import Counter

    slots, _, _ = load_rhythm_scaffold(REF)
    tab_key = lambda n: (int(n["pitch_midi"]), int(n["string"]), int(n["fret"]))
    fake = [
        ExportNote(
            start=s.start_sec + 0.01,
            end=s.start_sec + 0.1,
            pitch_midi=60 + (i % 5),
            string=i % 6,
            fret=i % 5,
            confidence=0.9,
        )
        for i, s in enumerate(slots)
        if not s.is_rest
    ]
    tab_counts = Counter(tab_key(n) for n in fake)
    assignments = assign_notes_to_scaffold(fake, slots, max_snap_sec=0.12)
    assigned = [n for grp in assignments for n in grp]
    assigned_counts = Counter(tab_key(n) for n in assigned)
    for key, n in assigned_counts.items():
        assert n <= tab_counts.get(key, 0)
    assert len(assigned) == len(fake)


@pytest.mark.skipif(not REF.is_file(), reason="test2.gp5 missing")
def test_scaffold_export_no_spurious_rests_when_notes_aligned():
    slots, _, _ = load_rhythm_scaffold(REF)
    aligned = [
        ExportNote(
            start=s.start_sec,
            end=s.start_sec + 0.1,
            pitch_midi=60 + (i % 12),
            string=2,
            fret=0,
            confidence=0.9,
        )
        for i, s in enumerate(slots)
        if not s.is_rest
    ]
    assignments = assign_notes_to_scaffold(aligned, slots, max_snap_sec=0.12)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "aligned.gp5"
        song = build_song_from_scaffold(
            slots,
            assignments,
            options=Gp5ExportOptions(tempo=91, time_signature=(4, 4)),
        )
        song.versionTuple = (5, 1, 0)
        guitarpro.write(song, str(out), version=(5, 1, 0))
        assert _rest_count(str(out)) == _rest_count(str(REF))
