"""Convert timed tab notes into a PyGuitarPro Song (measures, beats, rests)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Sequence

from guitarpro.models import (
    Beat,
    BeatStatus,
    Duration,
    GuitarString,
    Note,
    NoteType,
    Song,
    TimeSignature,
    Track,
    Voice,
)

from aitabs.pipeline.audio.tempo import TempoAnalysis, TempoMap, onset_to_tick, tick_to_time
from aitabs.pipeline.mapping.guitar import STANDARD_TUNING, GuitarSetup, fret_to_midi

from .guitarpro_types import ExportNote, aitabs_string_to_gp

_STANDARD_DURATION_VALUES = (
    Duration.whole,
    Duration.half,
    Duration.quarter,
    Duration.eighth,
    Duration.sixteenth,
    Duration.thirtySecond,
    Duration.sixtyFourth,
)

TICKS_PER_QUARTER = Duration.quarterTime
MIN_NOTE_TICKS = TICKS_PER_QUARTER // 16


@dataclass(frozen=True)
class Gp5ExportOptions:
    """Layout options for seconds → Guitar Pro notation."""

    tempo: int = 120
    time_signature: tuple[int, int] = (4, 4)
    simultaneous_onset_sec: float | None = None
    chord_grid_ticks: int = 0
    trim_leading_silence: bool = True
    quantize_division: int = 16
    min_note_ticks: int = MIN_NOTE_TICKS
    duration_mode: Literal["to_next_onset", "detected", "grid_unit", "note_value"] = "note_value"
    tempo_grid: TempoAnalysis | None = None
    snap_to_grid: bool = False
    enable_triplets: bool = False
    tuning: tuple[int, ...] | None = None  # open-string MIDI low→high; None = standard
    capo: int = 0
    tempo_map: "TempoMap | None" = None  # variable tempo for note_value quantization
    lead_in_beats: int = 0  # pickup beats before the first downbeat (bar anchoring)
    # Live-recording notation (note_value mode): snap onsets to the coarsest
    # uniform grid that fits (quarter→eighth→sixteenth) so jittered human timing
    # reads as clean note values, and fill each note to the next onset (legato)
    # instead of inserting rests from unreliable audio note-offsets.
    snap_dominant_grid: bool = False
    dominant_grid_max_mae: float = 0.12  # beats; max mean snap error to accept a grid
    legato: bool = False  # ignore note-offsets: note length = inter-onset interval
    # Joint tempo-spacing + phase grid alignment (note_value mode): rescale the
    # onset stream within a small tempo band so a few-% BPM estimation error stops
    # turning steady eighths into 16ths + false syncopation. No-op when the grid is
    # already correct (e.g. MIDI renders), so tight takes are not regressed.
    optimize_grid: bool = False
    # Self-configuring rhythm: infer legato vs rests, tempo-spacing correction, and
    # mixed-resolution grid from the onsets instead of fixed flags. Overrides
    # snap_dominant_grid / legato / optimize_grid. Recommended for real recordings.
    auto_rhythm: bool = False
    # Finest onset grid for note_value mode: 4 = sixteenth (default), 8 = thirty-
    # second (preserves more distinct attacks from close performance timing).
    base_division: int = 4


def apply_guitar_setup(track: Track, setup: GuitarSetup | None = None, *, options: Gp5ExportOptions | None = None) -> None:
    """Write GP5 string tuning and capo onto a track."""
    if setup is not None:
        tuning = setup.tuning
        capo = setup.capo
    elif options is not None:
        tuning = options.tuning if options.tuning is not None else tuple(STANDARD_TUNING)
        capo = options.capo
    else:
        tuning = tuple(STANDARD_TUNING)
        capo = 0
    track.strings = [
        GuitarString(number=gp_num, value=int(tuning[6 - gp_num]))
        for gp_num in range(1, 7)
    ]
    track.offset = int(capo)


@dataclass
class _NoteGroup:
    tick_start: int
    duration_ticks: int
    notes: list[ExportNote]


@dataclass
class _TrackCursor:
    """Current measure index and tick position within that measure."""

    measure_index: int = 0
    tick: int = 0

    def voice(self, track: Track) -> Voice:
        return track.measures[self.measure_index].voices[0]


def seconds_to_ticks(seconds: float, bpm: int) -> int:
    return int(round(seconds * bpm / 60.0 * TICKS_PER_QUARTER))


def gp_ticks_per_grid_unit(analysis: TempoAnalysis) -> int:
    return TICKS_PER_QUARTER // analysis.ticks_per_beat


def quantize_ticks(ticks: int, division: int) -> int:
    if division <= 0:
        return ticks
    grid = max(TICKS_PER_QUARTER // division, 1)
    return int(round(ticks / grid) * grid)


def group_notes_by_onset(
    notes: Sequence[ExportNote],
    *,
    epsilon_sec: float,
) -> list[list[ExportNote]]:
    """Cluster notes with nearly the same start time."""
    return _cluster_notes(
        sorted(notes, key=lambda n: (n["start"], n["pitch_midi"])),
        lambda prev, note: abs(note["start"] - prev["start"]) <= epsilon_sec,
    )


def group_notes_by_grid(
    notes: Sequence[ExportNote],
    analysis: TempoAnalysis,
    *,
    max_tick_gap: int,
) -> list[list[ExportNote]]:
    """Cluster notes within ``max_tick_gap`` steps on the tempo grid."""
    max_tick_gap = max(max_tick_gap, 0)
    ordered = sorted(notes, key=lambda n: (n["start"], n["pitch_midi"]))

    def within_gap(prev: ExportNote, note: ExportNote) -> bool:
        anchor = onset_to_tick(prev["start"], analysis)
        return onset_to_tick(note["start"], analysis) - anchor <= max_tick_gap

    return _cluster_notes(ordered, within_gap)


def _cluster_notes(
    ordered: Sequence[ExportNote],
    merge_with_last,
) -> list[list[ExportNote]]:
    if not ordered:
        return []
    groups: list[list[ExportNote]] = [[ordered[0]]]
    for note in ordered[1:]:
        if merge_with_last(groups[-1][0], note):
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def snap_notes_to_grid(
    notes: Sequence[ExportNote],
    analysis: TempoAnalysis,
) -> list[ExportNote]:
    """Snap each note start (and end) to the tempo grid before rhythm layout."""
    snapped: list[ExportNote] = []
    for note in notes:
        start = tick_to_time(onset_to_tick(float(note["start"]), analysis), analysis)
        end = tick_to_time(onset_to_tick(float(note["end"]), analysis), analysis)
        if end < start:
            end = start
        snapped.append(
            ExportNote(
                start=start,
                end=end,
                pitch_midi=int(note["pitch_midi"]),
                string=int(note["string"]),
                fret=int(note["fret"]),
                confidence=float(note.get("confidence", 1.0)),
            )
        )
    return snapped


def group_notes_for_export(
    notes: Sequence[ExportNote],
    options: Gp5ExportOptions,
) -> list[list[ExportNote]]:
    analysis = options.tempo_grid
    if analysis is not None and not analysis.uses_adaptive_template:
        return group_notes_by_grid(
            notes,
            analysis,
            max_tick_gap=options.chord_grid_ticks,
        )
    if analysis is not None and analysis.uses_adaptive_template:
        epsilon = max(analysis.tick_duration * 0.35, 0.02)
        return group_notes_by_onset(notes, epsilon_sec=epsilon)
    epsilon = options.simultaneous_onset_sec if options.simultaneous_onset_sec is not None else 0.04
    return group_notes_by_onset(notes, epsilon_sec=epsilon)


def _apply_duration_mode(
    groups: list[_NoteGroup],
    options: Gp5ExportOptions,
    *,
    gp_unit: int,
) -> None:
    min_dur = max(gp_unit, options.min_note_ticks)
    for i, group in enumerate(groups):
        span = (
            groups[i + 1].tick_start - group.tick_start
            if i + 1 < len(groups)
            else group.duration_ticks
        )
        span = max(span, min_dur)
        if options.duration_mode == "grid_unit":
            group.duration_ticks = gp_unit
        elif options.duration_mode == "to_next_onset":
            group.duration_ticks = span
        else:
            group.duration_ticks = max(min(group.duration_ticks, span), min_dur)


@dataclass
class _TimingMapper:
    """Map note chunks to tick_start and raw duration_ticks."""

    gp_unit: int
    start_ticks: Callable[[Sequence[ExportNote]], int]
    duration_ticks: Callable[[Sequence[ExportNote]], int]

    def start(self, chunk: Sequence[ExportNote]) -> int:
        return self.start_ticks(chunk)

    def duration(self, chunk: Sequence[ExportNote]) -> int:
        return self.duration_ticks(chunk)


def _grid_timing(options: Gp5ExportOptions, first_start: float) -> _TimingMapper:
    analysis = options.tempo_grid
    assert analysis is not None
    gp_unit = gp_ticks_per_grid_unit(analysis)
    base_grid = onset_to_tick(first_start, analysis) if options.trim_leading_silence else 0

    def start_ticks(chunk: Sequence[ExportNote]) -> int:
        start_grid = min(onset_to_tick(n["start"], analysis) for n in chunk) - base_grid
        return max(0, start_grid) * gp_unit

    def duration_ticks(chunk: Sequence[ExportNote]) -> int:
        start_grid = min(onset_to_tick(n["start"], analysis) for n in chunk) - base_grid
        start_grid = max(0, start_grid)
        end_grid = max(
            onset_to_tick(max(n["end"] for n in chunk), analysis) - base_grid,
            start_grid + 1,
        )
        return (end_grid - start_grid) * gp_unit

    return _TimingMapper(gp_unit=gp_unit, start_ticks=start_ticks, duration_ticks=duration_ticks)


def _bpm_timing(options: Gp5ExportOptions, first_start: float) -> _TimingMapper:
    t0 = first_start if options.trim_leading_silence else 0.0
    division = options.quantize_division
    gp_unit = TICKS_PER_QUARTER // max(division, 1)

    def start_ticks(chunk: Sequence[ExportNote]) -> int:
        rel = min(n["start"] for n in chunk) - t0
        return quantize_ticks(seconds_to_ticks(rel, options.tempo), division)

    def duration_ticks(chunk: Sequence[ExportNote]) -> int:
        rel_start = min(n["start"] for n in chunk) - t0
        rel_end = max(n["end"] for n in chunk) - t0
        tick_start = quantize_ticks(seconds_to_ticks(rel_start, options.tempo), division)
        tick_end = quantize_ticks(
            seconds_to_ticks(max(rel_end, rel_start), options.tempo),
            division,
        )
        raw = max(
            tick_end - tick_start,
            options.min_note_ticks,
            quantize_ticks(options.min_note_ticks, division),
        )
        return sum(d.time for d in decompose_duration_ticks(raw))

    return _TimingMapper(gp_unit=gp_unit, start_ticks=start_ticks, duration_ticks=duration_ticks)


def build_note_groups(
    notes: Sequence[ExportNote],
    options: Gp5ExportOptions,
) -> list[_NoteGroup]:
    if not notes:
        return []

    ordered = sorted(notes, key=lambda n: n["start"])
    analysis = options.tempo_grid
    use_wall_clock = analysis is None or analysis.uses_adaptive_template
    if analysis is not None and options.snap_to_grid and not analysis.uses_adaptive_template:
        ordered = snap_notes_to_grid(ordered, analysis)
    timing = (
        _bpm_timing(options, ordered[0]["start"])
        if use_wall_clock
        else _grid_timing(options, ordered[0]["start"])
    )

    groups = [
        _NoteGroup(
            tick_start=timing.start(chunk),
            duration_ticks=timing.duration(chunk),
            notes=list(chunk),
        )
        for chunk in group_notes_for_export(ordered, options)
    ]
    groups = _coalesce_groups_at_same_tick(groups)
    _apply_duration_mode(groups, options, gp_unit=timing.gp_unit)
    return groups


def _coalesce_groups_at_same_tick(groups: list[_NoteGroup]) -> list[_NoteGroup]:
    """Merge export groups that share a tick start (chord / duplicate snap)."""
    if not groups:
        return []
    merged: list[_NoteGroup] = [groups[0]]
    for group in groups[1:]:
        if group.tick_start == merged[-1].tick_start:
            merged[-1].notes.extend(group.notes)
            merged[-1].notes = dedupe_chord_notes(merged[-1].notes)
        else:
            merged.append(group)
    return merged


def decompose_duration_ticks(ticks: int) -> list[Duration]:
    if ticks <= 0:
        return []
    smallest = Duration(value=Duration.sixtyFourth).time
    remaining = max(int(round(ticks / smallest)) * smallest, smallest)
    durations: list[Duration] = []
    while remaining > 0:
        picked = next(
            (
                Duration(value=value)
                for value in _STANDARD_DURATION_VALUES
                if Duration(value=value).time <= remaining
            ),
            Duration(value=Duration.sixtyFourth),
        )
        durations.append(picked)
        remaining -= picked.time
    return durations


def durations_for_note_span(ticks: int) -> list[Duration]:
    if ticks <= 0:
        return [Duration(value=Duration.sixteenth)]
    try:
        return [Duration.fromTime(ticks)]
    except ValueError:
        return decompose_duration_ticks(ticks)


def dedupe_chord_notes(notes: Sequence[ExportNote]) -> list[ExportNote]:
    by_string: dict[int, ExportNote] = {}
    for note in notes:
        string_idx = int(note["string"])
        if string_idx not in by_string:
            by_string[string_idx] = note
            continue
        prev = by_string[string_idx]
        prev_ok = fret_to_midi(string_idx, prev["fret"]) == prev["pitch_midi"]
        curr_ok = fret_to_midi(string_idx, note["fret"]) == note["pitch_midi"]
        if curr_ok and not prev_ok:
            by_string[string_idx] = note
        elif note["confidence"] > prev["confidence"]:
            by_string[string_idx] = note
    return list(by_string.values())


def _time_signature(numerator: int, denominator: int) -> TimeSignature:
    denom_map = {
        1: Duration.whole,
        2: Duration.half,
        4: Duration.quarter,
        8: Duration.eighth,
        16: Duration.sixteenth,
    }
    return TimeSignature(
        numerator=numerator,
        denominator=Duration(value=denom_map.get(denominator, Duration.quarter)),
    )


def _append_rest_beat(voice: Voice, duration: Duration) -> None:
    voice.beats.append(Beat(voice=voice, duration=duration, status=BeatStatus.rest))


def _append_note_beat(
    voice: Voice,
    duration: Duration,
    notes: Sequence[ExportNote],
    *,
    tie: bool = False,
) -> None:
    chord = dedupe_chord_notes(notes)
    if not chord:
        return
    beat = Beat(voice=voice, duration=duration, status=BeatStatus.normal)
    note_type = NoteType.tie if tie else NoteType.normal
    for note in chord:
        gp_string = aitabs_string_to_gp(note["string"])
        if not 1 <= gp_string <= 6:
            raise ValueError(f"Invalid string index {note['string']}")
        beat.notes.append(
            Note(
                beat=beat,
                string=gp_string,
                value=int(note["fret"]),
                velocity=80,
                type=note_type,
            )
        )
    voice.beats.append(beat)


def _pad_measure(cursor: _TrackCursor, track: Track, measure_length: int) -> None:
    gap = measure_length - cursor.tick
    if gap <= 0:
        return
    voice = cursor.voice(track)
    for duration in decompose_duration_ticks(gap):
        _append_rest_beat(voice, duration)
    cursor.tick = measure_length


def _roll_measure(
    song: Song,
    track: Track,
    cursor: _TrackCursor,
    measure_length: int,
) -> None:
    _pad_measure(cursor, track, measure_length)
    cursor.measure_index += 1
    song.newMeasure()
    song.measureHeaders[-1].timeSignature = song.measureHeaders[0].timeSignature
    cursor.tick = 0


def _emit_ticks(
    song: Song,
    track: Track,
    cursor: _TrackCursor,
    measure_length: int,
    ticks: int,
    notes: Sequence[ExportNote],
) -> None:
    remaining = ticks
    tie_continue = False
    while remaining > 0:
        space = measure_length - cursor.tick
        if space <= 0:
            _roll_measure(song, track, cursor, measure_length)
            continue

        chunk = min(remaining, space)
        voice = cursor.voice(track)
        parts = durations_for_note_span(chunk)
        for i, duration in enumerate(parts):
            _append_note_beat(voice, duration, notes, tie=tie_continue or i > 0)
        tie_continue = remaining > chunk
        cursor.tick += chunk
        remaining -= chunk
        if cursor.tick >= measure_length:
            _roll_measure(song, track, cursor, measure_length)


def _emit_rest_gap(
    song: Song,
    track: Track,
    cursor: _TrackCursor,
    measure_length: int,
    ticks: int,
) -> None:
    remaining = ticks
    while remaining > 0:
        space = measure_length - cursor.tick
        if space <= 0:
            _roll_measure(song, track, cursor, measure_length)
            continue
        chunk = min(remaining, space)
        voice = cursor.voice(track)
        for duration in decompose_duration_ticks(chunk):
            _append_rest_beat(voice, duration)
        cursor.tick += chunk
        remaining -= chunk
        if cursor.tick >= measure_length:
            _roll_measure(song, track, cursor, measure_length)


def validate_gp5_song(song: Song) -> None:
    track = song.tracks[0]
    for mi, measure in enumerate(track.measures):
        header = song.measureHeaders[mi]
        beats = measure.voices[0].beats
        total = sum(b.duration.time for b in beats)
        if total != header.length:
            raise ValueError(
                f"Measure {mi + 1}: beats sum to {total} ticks, expected {header.length}"
            )
        for bi, beat in enumerate(beats):
            if beat.status == BeatStatus.rest and beat.notes:
                raise ValueError(f"Measure {mi + 1} beat {bi + 1}: rest beat has notes")
            strings = [n.string for n in beat.notes]
            if len(strings) != len(set(strings)):
                raise ValueError(
                    f"Measure {mi + 1} beat {bi + 1}: duplicate string(s) {strings}"
                )
            for note in beat.notes:
                if not 1 <= note.string <= 6:
                    raise ValueError(
                        f"Measure {mi + 1} beat {bi + 1}: invalid string {note.string}"
                    )
                if not 0 <= note.value <= 99:
                    raise ValueError(
                        f"Measure {mi + 1} beat {bi + 1}: invalid fret {note.value}"
                    )


def build_song_from_tab_notes(
    notes: Sequence[ExportNote],
    *,
    title: str = "AITabs Transcription",
    options: Gp5ExportOptions | None = None,
) -> Song:
    opts = options or Gp5ExportOptions()
    if opts.duration_mode == "note_value":
        from .rhythm_notation import build_song_from_note_values

        return build_song_from_note_values(
            list(notes),
            title=title,
            options=opts,
            enable_triplets=opts.enable_triplets,
        )

    song = Song()
    song.title = title
    song.tempo = opts.tempo

    track = song.tracks[0]
    track.name = "Guitar"
    apply_guitar_setup(track, options=opts)

    num, denom = opts.time_signature
    song.measureHeaders[0].timeSignature = _time_signature(num, denom)
    measure_length = song.measureHeaders[0].length

    groups = build_note_groups(notes, opts)
    if not groups:
        _append_rest_beat(track.measures[0].voices[0], Duration(value=Duration.quarter))
        return song

    cursor = _TrackCursor()
    timeline = 0
    pending_notes: list[ExportNote] = []
    for group in groups:
        gap = group.tick_start - timeline
        if gap < 0:
            pending_notes.extend(group.notes)
            continue
        if pending_notes:
            group = _NoteGroup(
                tick_start=group.tick_start,
                duration_ticks=group.duration_ticks,
                notes=dedupe_chord_notes([*pending_notes, *group.notes]),
            )
            pending_notes = []
        if gap > 0:
            _emit_rest_gap(song, track, cursor, measure_length, gap)
            timeline += gap
        _emit_ticks(song, track, cursor, measure_length, group.duration_ticks, group.notes)
        timeline += group.duration_ticks

    _pad_measure(cursor, track, measure_length)
    return song
