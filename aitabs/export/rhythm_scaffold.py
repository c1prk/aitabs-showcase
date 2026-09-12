"""Build export rhythm from a reference GP5 beat schedule (rests + durations)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import guitarpro
from guitarpro.models import BeatStatus, Duration

from .gp5_layout import (
    Gp5ExportOptions,
    _NoteGroup,
    _TrackCursor,
    _emit_rest_gap,
    _emit_ticks,
    _pad_measure,
    _time_signature,
    dedupe_chord_notes,
    validate_gp5_song,
)
from .guitarpro_types import ExportNote
from guitarpro.models import GuitarString, Song

_TICKS_PER_QUARTER = Duration.quarterTime


@dataclass(frozen=True)
class RhythmBeatSlot:
    """One GP5 beat position (note or rest)."""

    start_sec: float
    duration_ticks: int
    duration_value: int
    is_rest: bool
    measure_index: int
    tick_in_song: int


def load_rhythm_scaffold(
    gp5_path: str | Path,
    *,
    track_index: int = 0,
    voice_index: int = 0,
) -> tuple[list[RhythmBeatSlot], float, tuple[int, int]]:
    """Parse reference GP5 into an ordered beat schedule including rests."""
    path = Path(gp5_path)
    song = guitarpro.parse(str(path))
    bpm = float(song.tempo) if song.tempo else 120.0
    track = song.tracks[track_index]
    ts = song.measureHeaders[0].timeSignature
    num = ts.numerator.value if hasattr(ts.numerator, "value") else ts.numerator
    denom = ts.denominator.value if hasattr(ts.denominator, "value") else ts.denominator
    time_sig = (int(num), int(denom))

    slots: list[RhythmBeatSlot] = []
    song_tick = 0

    for mi, measure in enumerate(track.measures):
        if voice_index >= len(measure.voices):
            continue
        voice = measure.voices[voice_index]
        for beat in voice.beats:
            is_rest = beat.status == BeatStatus.rest or not beat.notes
            start_sec = song_tick * 60.0 / (bpm * _TICKS_PER_QUARTER)
            slots.append(
                RhythmBeatSlot(
                    start_sec=start_sec,
                    duration_ticks=int(beat.duration.time),
                    duration_value=int(beat.duration.value),
                    is_rest=is_rest,
                    measure_index=mi,
                    tick_in_song=song_tick,
                )
            )
            song_tick += int(beat.duration.time)

    return slots, bpm, time_sig


def _note_key(note: ExportNote) -> tuple[float, int, int]:
    return (round(float(note["start"]), 6), int(note["pitch_midi"]), int(note["string"]))


def estimate_scaffold_time_offset(
    tab_notes: list[ExportNote],
    slots: list[RhythmBeatSlot],
    *,
    max_snap_sec: float = 0.12,
    max_shift_sec: float = 3.0,
    step_sec: float = 0.01,
) -> float:
    """Find a constant shift on predictions that maximizes slot coverage."""
    import numpy as np

    playable = [s for s in slots if not s.is_rest]
    if not playable or not tab_notes:
        return 0.0

    pred_t = np.array([float(n["start"]) for n in tab_notes], dtype=float)
    slot_t = np.array([s.start_sec for s in playable], dtype=float)

    best_offset = 0.0
    best_filled = -1
    for offset in np.arange(-max_shift_sec, max_shift_sec + step_sec * 0.5, step_sec):
        dist = np.abs(pred_t + float(offset) - slot_t[:, np.newaxis])
        filled = int(np.sum(np.min(dist, axis=1) <= max_snap_sec))
        if filled > best_filled:
            best_filled = filled
            best_offset = float(offset)
    return best_offset


def _shift_export_notes(notes: list[ExportNote], offset_sec: float) -> list[ExportNote]:
    if offset_sec == 0.0:
        return notes
    return [
        ExportNote(
            start=float(n["start"]) + offset_sec,
            end=float(n["end"]) + offset_sec,
            pitch_midi=int(n["pitch_midi"]),
            string=int(n["string"]),
            fret=int(n["fret"]),
            confidence=float(n.get("confidence", 1.0)),
        )
        for n in notes
    ]


def _export_note_at_slot(note: ExportNote, slot: RhythmBeatSlot) -> ExportNote:
    """Place note on the reference beat time; keep pitch/string/fret unchanged."""
    end = max(float(note["end"]), slot.start_sec + 0.02)
    return ExportNote(
        start=slot.start_sec,
        end=end,
        pitch_midi=int(note["pitch_midi"]),
        string=int(note["string"]),
        fret=int(note["fret"]),
        confidence=float(note.get("confidence", 1.0)),
    )


def assign_notes_to_scaffold(
    tab_notes: list[ExportNote],
    slots: list[RhythmBeatSlot],
    *,
    max_snap_sec: float = 0.12,
) -> list[list[ExportNote]]:
    """Map each predicted note to at most one slot; never invent or reuse pitches."""
    ordered = sorted(tab_notes, key=lambda n: n["start"])
    assignments: list[list[ExportNote]] = [[] for _ in slots]
    placed: set[tuple[float, int, int]] = set()

    def _nearest_playable_slot(note: ExportNote) -> tuple[int | None, float]:
        best_i: int | None = None
        best_dt = float("inf")
        for i, slot in enumerate(slots):
            if slot.is_rest:
                continue
            dt = abs(float(note["start"]) - slot.start_sec)
            if dt < best_dt:
                best_dt = dt
                best_i = i
        return best_i, best_dt

    for note in ordered:
        key = _note_key(note)
        if key in placed:
            continue
        best_i, best_dt = _nearest_playable_slot(note)
        if best_i is not None and best_dt <= max_snap_sec:
            slot = slots[best_i]
            assignments[best_i].append(_export_note_at_slot(note, slot))
            placed.add(key)

    # Looser snap for notes not yet placed (still one note → one slot).
    for note in ordered:
        key = _note_key(note)
        if key in placed:
            continue
        best_i, best_dt = _nearest_playable_slot(note)
        if best_i is not None and best_dt <= max_snap_sec * 2:
            slot = slots[best_i]
            assignments[best_i].append(_export_note_at_slot(note, slot))
            placed.add(key)

    return assignments


def build_note_groups_from_scaffold(
    slots: list[RhythmBeatSlot],
    assignments: list[list[ExportNote]],
    *,
    trim_leading_silence: bool = True,
) -> list[_NoteGroup]:
    """Build export groups with exact reference durations and rest gaps."""
    if len(slots) != len(assignments):
        raise ValueError("slots and assignments length mismatch")

    base_tick = slots[0].tick_in_song if trim_leading_silence and slots else 0
    groups: list[_NoteGroup] = []

    for slot, notes in zip(slots, assignments):
        tick_start = slot.tick_in_song - base_tick
        if slot.is_rest:
            groups.append(
                _NoteGroup(
                    tick_start=tick_start,
                    duration_ticks=slot.duration_ticks,
                    notes=[],
                )
            )
        elif not notes:
            # Unfilled playable slot: keep timing; export may show a rest beat.
            groups.append(
                _NoteGroup(
                    tick_start=tick_start,
                    duration_ticks=slot.duration_ticks,
                    notes=[],
                )
            )
        else:
            groups.append(
                _NoteGroup(
                    tick_start=tick_start,
                    duration_ticks=slot.duration_ticks,
                    notes=dedupe_chord_notes(notes),
                )
            )

    groups.sort(key=lambda g: g.tick_start)
    return groups


def build_song_from_scaffold(
    slots: list[RhythmBeatSlot],
    assignments: list[list[ExportNote]],
    *,
    title: str = "AITabs Transcription",
    options: Gp5ExportOptions | None = None,
) -> Song:
    """Write GP5 using reference rhythm; predicted notes fill each slot."""
    opts = options or Gp5ExportOptions()
    song = Song()
    song.title = title
    song.tempo = opts.tempo

    track = song.tracks[0]
    track.name = "Guitar"
    track.strings = [
        GuitarString(number=i, value=v)
        for i, v in enumerate([64, 59, 55, 50, 45, 40], start=1)
    ]

    num, denom = opts.time_signature
    song.measureHeaders[0].timeSignature = _time_signature(num, denom)
    measure_length = song.measureHeaders[0].length

    groups = build_note_groups_from_scaffold(
        slots,
        assignments,
        trim_leading_silence=opts.trim_leading_silence,
    )

    cursor = _TrackCursor()
    timeline = 0
    for group in groups:
        gap = group.tick_start - timeline
        if gap > 0:
            _emit_rest_gap(song, track, cursor, measure_length, gap)
            timeline += gap
        if not group.notes:
            _emit_rest_gap(song, track, cursor, measure_length, group.duration_ticks)
        else:
            _emit_ticks(song, track, cursor, measure_length, group.duration_ticks, group.notes)
        timeline += group.duration_ticks

    _pad_measure(cursor, track, measure_length)
    validate_gp5_song(song)
    return song


def export_gp5_with_reference_rhythm(
    tab_notes: list[ExportNote],
    reference_gp5: str | Path,
    output_path: str | Path,
    *,
    title: str = "AITabs Transcription",
    slot_snap_sec: float = 0.12,
) -> Path:
    """Export prediction using reference GP5 beat/rest layout."""
    import guitarpro as gp

    slots, bpm, time_sig = load_rhythm_scaffold(reference_gp5)
    offset = estimate_scaffold_time_offset(tab_notes, slots, max_snap_sec=slot_snap_sec)
    aligned = _shift_export_notes(tab_notes, offset)
    assignments = assign_notes_to_scaffold(aligned, slots, max_snap_sec=slot_snap_sec)
    song = build_song_from_scaffold(
        slots,
        assignments,
        title=title,
        options=Gp5ExportOptions(
            tempo=int(round(bpm)),
            time_signature=time_sig,
            trim_leading_silence=True,
        ),
    )
    song.versionTuple = (5, 1, 0)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".gp5.tmp")
    gp.write(song, str(tmp), version=(5, 1, 0))
    gp.parse(str(tmp))
    tmp.replace(path)
    return path
