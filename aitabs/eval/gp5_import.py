"""Import note events from Guitar Pro (.gp5) files."""

from __future__ import annotations

from pathlib import Path

import guitarpro
from guitarpro.models import BeatStatus, Duration

from aitabs.pipeline.mapping.guitar import GuitarSetup, read_guitar_setup_from_track

from .types import EvalNote

_TICKS_PER_QUARTER = Duration.quarterTime


def gp_string_to_aitabs(gp_string: int) -> int:
    """Guitar Pro string number (1=high e) → AITabs index (0=low E)."""
    return 6 - int(gp_string)


def _ticks_to_seconds(tick: int, bpm: float) -> float:
    if bpm <= 0:
        bpm = 120.0
    return float(tick) * 60.0 / (float(bpm) * float(_TICKS_PER_QUARTER))


def load_guitar_setup(
    gp5_path: str | Path,
    *,
    track_index: int = 0,
) -> GuitarSetup:
    """Read capo and string tuning from a GP5 track."""
    song = guitarpro.parse(str(Path(gp5_path)))
    return read_guitar_setup_from_track(song.tracks[track_index])


def load_gp5_time_signature(gp5_path: str | Path) -> tuple[int, int]:
    """Read the first measure's time signature as ``(numerator, denominator)``."""
    song = guitarpro.parse(str(Path(gp5_path)))
    ts = song.measureHeaders[0].timeSignature
    num = int(ts.numerator)
    # denominator note value: whole(3840)/dur.time → 4 for quarter, 8 for eighth, …
    denom = int(round(3840 / ts.denominator.time)) if ts.denominator.time else 4
    return num, denom


def load_gp5_tempo_map(gp5_path: str | Path, *, track_index: int = 0) -> "TempoMap":
    """Build a :class:`TempoMap` honouring mid-song ``mixTableChange.tempo`` events."""
    from aitabs.pipeline.audio.tempo import TempoMap

    song = guitarpro.parse(str(Path(gp5_path)))
    header_bpm = float(song.tempo) if song.tempo else 120.0
    track = song.tracks[track_index]

    anchors: list[tuple[float, float, float]] = [(0.0, 0.0, header_bpm)]
    elapsed_beats = 0.0
    elapsed_sec = 0.0
    current_bpm = header_bpm

    for measure in track.measures:
        if not measure.voices:
            continue
        for beat in measure.voices[0].beats:
            mix_change = getattr(getattr(beat, "effect", None), "mixTableChange", None)
            if mix_change is not None and mix_change.tempo is not None:
                new_bpm = float(mix_change.tempo.value)
                if new_bpm > 0 and new_bpm != current_bpm:
                    current_bpm = new_bpm
                    anchors.append((elapsed_beats, elapsed_sec, current_bpm))
            quarters = beat.duration.time / _TICKS_PER_QUARTER
            elapsed_beats += quarters
            elapsed_sec += quarters * 60.0 / current_bpm

    return TempoMap(anchors=tuple(anchors))


def load_gp5_notes(
    gp5_path: str | Path,
    *,
    track_index: int = 0,
    voice_index: int = 0,
) -> tuple[list[EvalNote], float, GuitarSetup]:
    """Parse a GP5 file into timed notes with string/fret.

    Honours mid-song tempo changes (``mixTableChange.tempo``): note times are
    accumulated beat-by-beat using the tempo in force at each beat, rather than a
    single constant tempo. A constant tempo silently drifts every note after a
    tempo change out of alignment — fatal for onset-tolerance matching.

    ``pitch_midi`` is the *concert* (sounding) pitch, accounting for capo and
    alternate tunings stored in the file.

    Returns:
        (notes sorted by start, header_tempo_bpm from the file, guitar setup)
    """
    path = Path(gp5_path)
    song = guitarpro.parse(str(path))
    header_bpm = float(song.tempo) if song.tempo else 120.0
    track = song.tracks[track_index]
    setup = read_guitar_setup_from_track(track)

    notes: list[EvalNote] = []
    elapsed_sec = 0.0
    current_bpm = header_bpm

    for measure in track.measures:
        if voice_index >= len(measure.voices):
            continue
        voice = measure.voices[voice_index]
        for beat in voice.beats:
            # A tempo change on this beat applies from this beat onward.
            mix_change = getattr(getattr(beat, "effect", None), "mixTableChange", None)
            if mix_change is not None and mix_change.tempo is not None:
                current_bpm = float(mix_change.tempo.value)

            dur_sec = _ticks_to_seconds(beat.duration.time, current_bpm)

            if beat.status == BeatStatus.rest or not beat.notes:
                elapsed_sec += dur_sec
                continue

            start_sec = elapsed_sec
            end_sec = elapsed_sec + dur_sec

            for gp_note in beat.notes:
                gp_str = int(gp_note.string)
                fret = int(gp_note.value)
                aitabs_str = gp_string_to_aitabs(gp_str)
                notes.append(
                    EvalNote(
                        start=start_sec,
                        end=end_sec,
                        pitch_midi=setup.written_to_concert_midi(aitabs_str, fret),
                        string=aitabs_str,
                        fret=fret,
                        confidence=1.0,
                    )
                )
            elapsed_sec += dur_sec

    notes.sort(key=lambda n: (n["start"], n["pitch_midi"], n["string"]))
    return notes, header_bpm, setup


def eval_notes_to_export(notes: list[EvalNote]) -> list[dict]:
    """Convert EvalNote list to export-compatible dicts."""
    return [
        {
            "start": n["start"],
            "end": n["end"],
            "pitch_midi": n["pitch_midi"],
            "string": n["string"],
            "fret": n["fret"],
            "confidence": n["confidence"],
        }
        for n in notes
    ]
