"""Rhythm quantization helpers (adaptive template, grid snap, onset collapsing)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from aitabs.pipeline.audio.pitch import NoteEvent, _as_note_event
from aitabs.pipeline.audio.tempo import (
    TempoAnalysis,
    _phase_distance,
    onset_to_tick,
    snap_time_to_grid,
    template_slot_to_time,
    tick_to_time,
    time_to_template_slot,
)


def quantize_notes_to_grid(
    notes: list[NoteEvent] | list[tuple],
    analysis: TempoAnalysis,
    *,
    max_notes_per_tick: int | None = None,
) -> list[NoteEvent]:
    """Snap onsets to the tempo grid and merge notes that share a grid tick.

    Args:
        notes: Detected note events.
        analysis: Tempo grid (``ticks_per_beat=2`` → 8ths, ``4`` → 16ths).
        max_notes_per_tick: If 1, keep only highest-confidence note per tick.
            If None, keep all pitches at that tick (chord / arpeggio bucket).
    """
    normalized = [_as_note_event(n) for n in notes]
    if not normalized:
        return []

    by_tick: dict[int, list[NoteEvent]] = {}
    for note in normalized:
        tick = onset_to_tick(float(note["start"]), analysis)
        by_tick.setdefault(tick, []).append(note)

    kept: list[NoteEvent] = []
    for tick in sorted(by_tick):
        bucket = sorted(by_tick[tick], key=lambda n: -n["confidence"])
        if max_notes_per_tick == 1:
            bucket = bucket[:1]
        grid_start = tick_to_time(tick, analysis)
        for note in bucket:
            end_tick = max(onset_to_tick(float(note["end"]), analysis), tick + 1)
            kept.append(
                NoteEvent(
                    start=grid_start,
                    end=tick_to_time(end_tick, analysis),
                    pitch_midi=int(note["pitch_midi"]),
                    confidence=float(note["confidence"]),
                )
            )

    return sorted(kept, key=lambda n: (n["start"], n["pitch_midi"]))


# test2-style cell: eighth at 0 + three sixteenths at 40%, 60%, 80% of cell
_CELL_FRAC = (0.0, 0.4, 0.6, 0.8)
_CELL_DURATIONS = ("eighth", "sixteenth", "sixteenth", "sixteenth")


def quantize_notes_fingerstyle_cell(
    notes: list[NoteEvent] | list[tuple],
    analysis: TempoAnalysis,
    *,
    cell_ticks: int = 1200,
) -> list[NoteEvent]:
    """Snap onsets to repeating [8th + 3x16th] cell (GP tick space at 960 ticks/quarter)."""
    normalized = [_as_note_event(n) for n in notes]
    if not normalized or analysis.bpm <= 0:
        return []

    tick_sec = 60.0 / (analysis.bpm * 960.0)
    cell_sec = cell_ticks * tick_sec
    origin = analysis.origin

    by_slot: dict[tuple[int, int], list[NoteEvent]] = {}
    for note in normalized:
        t = float(note["start"]) - origin
        if t < 0:
            continue
        cell_i = int(t // cell_sec)
        pos_in = (t % cell_sec) / cell_sec
        slot_i = min(range(len(_CELL_FRAC)), key=lambda i: abs(pos_in - _CELL_FRAC[i]))
        by_slot.setdefault((cell_i, slot_i), []).append(note)

    kept: list[NoteEvent] = []
    for (cell_i, slot_i), bucket in sorted(by_slot.items()):
        bucket.sort(key=lambda n: -n["confidence"])
        grid_start = origin + cell_i * cell_sec + _CELL_FRAC[slot_i] * cell_sec
        for note in bucket:
            kept.append(
                NoteEvent(
                    start=grid_start,
                    end=grid_start + analysis.tick_duration,
                    pitch_midi=int(note["pitch_midi"]),
                    confidence=float(note["confidence"]),
                )
            )
    return sorted(kept, key=lambda n: (n["start"], n["pitch_midi"]))


def _collect_onset_phases(
    onset_times: np.ndarray,
    analysis: TempoAnalysis,
) -> np.ndarray:
    """Beat-normalized attack phases in [0, 1) for each onset."""
    beat_period = analysis.beat_period
    if beat_period <= 0 or len(onset_times) == 0:
        return np.array([], dtype=float)

    phases: list[float] = []
    for t in np.asarray(onset_times, dtype=float):
        rel = float(t) - analysis.origin
        if rel < -beat_period * 0.25:
            continue
        beat_i = int(np.floor(rel / beat_period))
        phase = (rel - beat_i * beat_period) / beat_period
        phases.append(float(phase % 1.0))
    return np.asarray(phases, dtype=float)


def _merge_phase_peaks(phases: np.ndarray, *, merge_frac: float) -> list[float]:
    """Merge nearby phase samples into cluster centers (circular)."""
    if len(phases) == 0:
        return []
    if len(phases) == 1:
        return [float(phases[0])]

    doubled = np.concatenate([phases, phases + 1.0])
    doubled.sort()
    clusters: list[list[float]] = [[float(doubled[0])]]
    for p in doubled[1:]:
        if p - clusters[-1][-1] <= merge_frac:
            clusters[-1].append(float(p))
        else:
            clusters.append([float(p)])

    centers: list[float] = []
    for cluster in clusters:
        raw = float(np.median(cluster))
        centers.append(raw % 1.0)

    centers.sort()
    merged: list[float] = []
    for c in centers:
        if merged and _phase_distance(c, merged[-1]) <= merge_frac:
            merged[-1] = (merged[-1] + c) / 2.0
        else:
            merged.append(c)
    return merged


def infer_beat_phase_template(
    onset_times: np.ndarray,
    analysis: TempoAnalysis,
    *,
    merge_frac: float = 0.04,
    peak_threshold_ratio: float = 0.35,
    min_slots: int = 2,
    max_slots: int = 16,
) -> tuple[float, ...]:
    """Discover recurring attack phases within a quarter-note beat from audio onsets."""
    phases = _collect_onset_phases(onset_times, analysis)
    if len(phases) < 4:
        n = max(analysis.ticks_per_beat, 4)
        return tuple(i / n for i in range(n))

    bins = max(32, max_slots * 4)
    hist, edges = np.histogram(phases, bins=bins, range=(0.0, 1.0))
    if float(hist.max()) <= 0:
        n = max(analysis.ticks_per_beat, 4)
        return tuple(i / n for i in range(n))

    threshold = float(hist.max()) * peak_threshold_ratio
    peak_centers: list[float] = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= threshold and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]:
            peak_centers.append(float((edges[i] + edges[i + 1]) * 0.5))

    if not peak_centers:
        peak_centers = _merge_phase_peaks(phases, merge_frac=merge_frac)
    else:
        peak_centers = _merge_phase_peaks(np.asarray(peak_centers, dtype=float), merge_frac=merge_frac)

    if len(peak_centers) < min_slots:
        for fallback in (0.0, 0.25, 0.5, 0.75):
            if all(_phase_distance(fallback, p) > merge_frac for p in peak_centers):
                if any(_phase_distance(fallback, ph) <= merge_frac * 1.5 for ph in phases):
                    peak_centers.append(fallback)
        peak_centers.sort()

    peak_centers = peak_centers[:max_slots]
    if 0.0 not in peak_centers and any(ph < merge_frac * 2 for ph in phases):
        peak_centers = [0.0] + peak_centers

    return tuple(sorted(set(round(p, 4) for p in peak_centers)))


def refine_grid_origin(
    onset_times: np.ndarray,
    analysis: TempoAnalysis,
    template: tuple[float, ...],
    *,
    search_sec: float | None = None,
    step_sec: float = 0.01,
) -> float:
    """Shift grid origin so onsets land closest to the inferred template."""
    if len(onset_times) == 0 or not template:
        return analysis.origin

    search = search_sec if search_sec is not None else analysis.beat_period
    best_origin = analysis.origin
    best_cost = float("inf")
    for delta in np.arange(-search, search + step_sec * 0.5, step_sec):
        origin = analysis.origin + float(delta)
        trial = replace(analysis, origin=origin, phase_template=template)
        cost = 0.0
        for t in onset_times:
            beat_i, slot_i = time_to_template_slot(float(t), trial)
            snapped = template_slot_to_time(beat_i, slot_i, trial)
            cost += abs(float(t) - snapped)
        if cost < best_cost:
            best_cost = cost
            best_origin = origin
    return best_origin


def build_adaptive_tempo_analysis(
    notes: list[NoteEvent] | list[tuple],
    analysis: TempoAnalysis,
    *,
    merge_frac: float = 0.04,
) -> TempoAnalysis:
    """Attach an audio-inferred phase template and refined origin to tempo analysis."""
    normalized = [_as_note_event(n) for n in notes]
    if not normalized:
        return analysis

    onsets = np.array([float(n["start"]) for n in normalized], dtype=float)
    template = infer_beat_phase_template(onsets, analysis, merge_frac=merge_frac)
    origin = refine_grid_origin(onsets, analysis, template)
    return replace(
        analysis,
        origin=origin,
        phase_template=template,
        ticks_per_beat=len(template),
    )


def quantize_notes_adaptive(
    notes: list[NoteEvent] | list[tuple],
    analysis: TempoAnalysis,
    *,
    merge_frac: float = 0.04,
) -> tuple[list[NoteEvent], TempoAnalysis]:
    """Snap onsets to an audio-inferred rhythmic template (no reference score).

    Discovers which subdivisions are actually played from the onset distribution,
    then quantizes each note to the nearest template slot while preserving pitch.
    """
    normalized = [_as_note_event(n) for n in notes]
    if not normalized:
        return [], analysis

    adaptive = build_adaptive_tempo_analysis(normalized, analysis, merge_frac=merge_frac)

    by_slot: dict[tuple[int, int], list[NoteEvent]] = {}
    for note in normalized:
        beat_i, slot_i = time_to_template_slot(float(note["start"]), adaptive)
        by_slot.setdefault((beat_i, slot_i), []).append(note)

    kept: list[NoteEvent] = []
    for (beat_i, slot_i), bucket in sorted(by_slot.items()):
        grid_start = template_slot_to_time(beat_i, slot_i, adaptive)
        bucket.sort(key=lambda n: -n["confidence"])
        for note in bucket:
            end_tick = max(
                onset_to_tick(float(note["end"]), adaptive),
                onset_to_tick(grid_start, adaptive) + 1,
            )
            kept.append(
                NoteEvent(
                    start=grid_start,
                    end=tick_to_time(end_tick, adaptive),
                    pitch_midi=int(note["pitch_midi"]),
                    confidence=float(note["confidence"]),
                )
            )

    return sorted(kept, key=lambda n: (n["start"], n["pitch_midi"])), adaptive
