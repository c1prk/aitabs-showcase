"""MIDI export from tab note sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pretty_midi

from .guitarpro_types import ExportNote


def export_midi(
    notes: Sequence[ExportNote],
    output_path: str | Path,
    *,
    program: int = 25,
) -> Path:
    """Write a .mid file from tab notes using MIDI pitch and timing.

    Args:
        notes: Notes with pitch_midi and timing in seconds.
        output_path: Destination .mid path.
        program: General MIDI program (25 = acoustic guitar).

    Returns:
        Path to the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, name="Guitar")

    for note in sorted(notes, key=lambda n: n["start"]):
        duration = max(note["end"] - note["start"], 0.05)
        instrument.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=note["pitch_midi"],
                start=note["start"],
                end=note["start"] + duration,
            )
        )

    midi.instruments.append(instrument)
    midi.write(str(path))
    return path
