"""Guitar Pro (.gp5) export from tab note sequences."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import guitarpro

from .gp5_layout import Gp5ExportOptions, build_song_from_tab_notes, validate_gp5_song
from .guitarpro_types import ExportNote, aitabs_string_to_gp

if TYPE_CHECKING:
    from aitabs.pipeline.audio.tempo import TempoAnalysis

__all__ = [
    "ExportNote",
    "Gp5ExportOptions",
    "aitabs_string_to_gp",
    "export_gp5",
    "validate_gp5_song",
]

_GP5_VERSION = (5, 1, 0)
DurationMode = Literal["to_next_onset", "detected", "grid_unit", "note_value"]


def export_gp5(
    notes: Sequence[ExportNote],
    output_path: str | Path,
    *,
    title: str = "AITabs Transcription",
    tempo: int = 120,
    time_signature: tuple[int, int] = (4, 4),
    simultaneous_onset_sec: float | None = None,
    chord_grid_ticks: int = 0,
    trim_leading_silence: bool = True,
    quantize_division: int = 16,
    duration_mode: DurationMode = "note_value",
    tempo_grid: TempoAnalysis | None = None,
    snap_to_grid: bool = False,
    enable_triplets: bool = False,
    tuning: list[int] | None = None,
    capo: int = 0,
    tempo_map=None,
    lead_in_beats: int = 0,
    snap_dominant_grid: bool = False,
    dominant_grid_max_mae: float = 0.12,
    legato: bool = False,
    optimize_grid: bool = False,
    auto_rhythm: bool = False,
    base_division: int = 4,
) -> Path:
    """Write a .gp5 file from resolved tab notes."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    options = Gp5ExportOptions(
        tempo=tempo,
        time_signature=time_signature,
        simultaneous_onset_sec=simultaneous_onset_sec,
        chord_grid_ticks=chord_grid_ticks,
        trim_leading_silence=trim_leading_silence,
        quantize_division=quantize_division,
        duration_mode=duration_mode,
        tempo_grid=tempo_grid,
        snap_to_grid=snap_to_grid,
        enable_triplets=enable_triplets,
        tuning=tuple(tuning) if tuning is not None else None,
        capo=capo,
        tempo_map=tempo_map,
        lead_in_beats=lead_in_beats,
        snap_dominant_grid=snap_dominant_grid,
        dominant_grid_max_mae=dominant_grid_max_mae,
        legato=legato,
        optimize_grid=optimize_grid,
        auto_rhythm=auto_rhythm,
        base_division=base_division,
    )
    song = build_song_from_tab_notes(notes, title=title, options=options)
    song.versionTuple = _GP5_VERSION
    validate_gp5_song(song)

    tmp = path.with_suffix(".gp5.tmp")
    try:
        guitarpro.write(song, str(tmp), version=_GP5_VERSION)
        guitarpro.parse(str(tmp))
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return path
