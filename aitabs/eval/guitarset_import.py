"""Import note events from GuitarSet JAMS annotations.

GuitarSet (https://zenodo.org/records/3371780, MIT licensed) ships per-string
note annotations as JAMS files (JSON Annotated Music Specification). JAMS files
are plain JSON, so we parse them with the stdlib — no `jams`/`mirdata`
dependency required.

Each excerpt has six ``note_midi`` annotations, one per string. GuitarSet's
string index 0 = low E (E2) … 5 = high e (E4), which matches the aitabs
convention exactly (see :data:`aitabs.pipeline.mapping.guitar.STANDARD_TUNING`),
so string/fret map straight across.

Why load JAMS directly instead of converting to .gp5 first:
GuitarSet captures *real performance* onset times. Converting to a notated .gp5
and back through :func:`load_gp5_notes` would re-quantise those onsets to a tempo
grid, destroying the very timing we want to measure note accuracy against. The
JAMS observations carry the true performed times, so we keep them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aitabs.pipeline.mapping.guitar import MAX_FRETS, STANDARD_TUNING, midi_to_positions

from .types import EvalNote

# GuitarSet string index → open-string MIDI (low E = 0 … high e = 5).
_STRING_MIDI = STANDARD_TUNING


def _iter_observations(data: object) -> list[dict]:
    """Yield observation dicts from a JAMS annotation ``data`` field.

    Handles both the row form (list of ``{time, duration, value, confidence}``)
    and the rarer columnar form (dict of parallel lists).
    """
    if isinstance(data, list):
        return [obs for obs in data if isinstance(obs, dict)]
    if isinstance(data, dict) and "value" in data:
        times = data.get("time", [])
        durations = data.get("duration", [])
        values = data.get("value", [])
        confidences = data.get("confidence", [None] * len(values))
        out: list[dict] = []
        for i, value in enumerate(values):
            out.append(
                {
                    "time": times[i] if i < len(times) else 0.0,
                    "duration": durations[i] if i < len(durations) else 0.0,
                    "value": value,
                    "confidence": confidences[i] if i < len(confidences) else None,
                }
            )
        return out
    return []


def _string_index(annotation: dict, fallback: int) -> int:
    """Extract the 0–5 string index for a ``note_midi`` annotation."""
    meta = annotation.get("annotation_metadata") or {}
    sandbox = annotation.get("sandbox") or {}
    for source in (
        meta.get("data_source"),
        sandbox.get("string_index"),
        sandbox.get("string"),
    ):
        if source is None:
            continue
        try:
            return int(str(source).strip().split()[-1])
        except (ValueError, IndexError):
            continue
    return fallback


def _fret_for(pitch_midi: int, string_idx: int) -> tuple[int, int]:
    """Return (string, fret) for a pitch, trusting the annotated string if valid.

    Falls back to the lowest valid position for the pitch when the annotated
    string would imply an out-of-range fret (defensive; pitch always wins).
    """
    if 0 <= string_idx < len(_STRING_MIDI):
        fret = pitch_midi - _STRING_MIDI[string_idx]
        if 0 <= fret <= MAX_FRETS:
            return string_idx, fret

    positions = midi_to_positions(pitch_midi)
    if positions:
        return positions[0]
    return max(0, min(string_idx, len(_STRING_MIDI) - 1)), 0


def _read_tempo(annotations: list[dict]) -> float:
    for annotation in annotations:
        if annotation.get("namespace") != "tempo":
            continue
        for obs in _iter_observations(annotation.get("data")):
            try:
                bpm = float(obs["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if bpm > 0:
                return bpm
    return 120.0


def load_guitarset_notes(jams_path: str | Path) -> tuple[list[EvalNote], float]:
    """Parse a GuitarSet JAMS file into timed notes with string/fret.

    Args:
        jams_path: Path to a GuitarSet ``*.jams`` annotation file.

    Returns:
        ``(notes sorted by start, tempo_bpm)`` — same shape as
        :func:`aitabs.eval.gp5_import.load_gp5_notes`, so it drops straight into
        the eval harness.
    """
    path = Path(jams_path)
    jam = json.loads(path.read_text(encoding="utf-8"))
    annotations = jam.get("annotations", [])

    bpm = _read_tempo(annotations)

    notes: list[EvalNote] = []
    string_counter = 0
    for annotation in annotations:
        if annotation.get("namespace") != "note_midi":
            continue
        string_idx = _string_index(annotation, fallback=string_counter)
        string_counter += 1

        for obs in _iter_observations(annotation.get("data")):
            try:
                start = float(obs["time"])
                duration = float(obs.get("duration") or 0.0)
                pitch_midi = int(round(float(obs["value"])))
            except (KeyError, TypeError, ValueError):
                continue
            string, fret = _fret_for(pitch_midi, string_idx)
            notes.append(
                EvalNote(
                    start=start,
                    end=start + duration,
                    pitch_midi=pitch_midi,
                    string=string,
                    fret=fret,
                    confidence=1.0,
                )
            )

    notes.sort(key=lambda n: (n["start"], n["pitch_midi"], n["string"]))
    return notes, bpm


@dataclass(frozen=True)
class BeatAnnotation:
    """Ground-truth metrical grid read from a GuitarSet ``beat_position`` layer.

    Times are the performed beat times in seconds. ``downbeat_times`` are the
    subset carrying ``position == 1`` (bar starts). ``numerator`` /
    ``denominator`` come from the JAMS ``num_beats`` / ``beat_units`` fields
    (constant across GuitarSet excerpts). ``tempo_bpm`` is the annotated tempo.
    """

    beat_times: tuple[float, ...]
    downbeat_times: tuple[float, ...]
    numerator: int
    denominator: int
    tempo_bpm: float


def load_guitarset_beats(jams_path: str | Path) -> BeatAnnotation:
    """Parse the metrical grid (beats, downbeats, meter, tempo) from a JAMS file.

    GuitarSet's ``beat_position`` annotation stores one observation per beat with
    ``value = {position, measure, num_beats, beat_units}``. ``position == 1``
    marks a downbeat. These are *performed* times (not a constant grid), so they
    are the honest ground truth for beat-tracking accuracy on real audio — the
    thing note-quantisation is built on.

    Returns:
        :class:`BeatAnnotation`. Beat/downbeat times are sorted ascending.
    """
    path = Path(jams_path)
    jam = json.loads(path.read_text(encoding="utf-8"))
    annotations = jam.get("annotations", [])
    tempo_bpm = _read_tempo(annotations)

    beat_times: list[float] = []
    downbeat_times: list[float] = []
    numerator = 4
    denominator = 4
    for annotation in annotations:
        if annotation.get("namespace") != "beat_position":
            continue
        for obs in _iter_observations(annotation.get("data")):
            try:
                t = float(obs["time"])
            except (KeyError, TypeError, ValueError):
                continue
            value = obs.get("value") or {}
            beat_times.append(t)
            try:
                if int(value.get("position")) == 1:
                    downbeat_times.append(t)
            except (TypeError, ValueError):
                pass
            # num_beats / beat_units are constant per excerpt; last one wins.
            try:
                numerator = int(value.get("num_beats") or numerator)
                denominator = int(value.get("beat_units") or denominator)
            except (TypeError, ValueError):
                pass

    beat_times.sort()
    downbeat_times.sort()
    return BeatAnnotation(
        beat_times=tuple(beat_times),
        downbeat_times=tuple(downbeat_times),
        numerator=numerator,
        denominator=denominator,
        tempo_bpm=tempo_bpm,
    )


def load_guitarset_training_pairs(data_dir: str | Path) -> list[dict]:
    """Return training pairs for all solo GuitarSet clips under data_dir.

    Only includes clips with ``_solo`` in their name (excludes ``_comp`` chord takes).
    Audio must be the ``*_mic.wav`` variant (clean acoustic, already isolated).

    Args:
        data_dir: Root of a prepared GuitarSet eval folder, e.g.
            ``data/eval/guitarset/`` produced by ``scripts/prepare_guitarset.py``.
            Expected layout::

                data_dir/
                  annotation/<clip_id>.jams
                  audio/<clip_id>_mic.wav

    Returns:
        List of dicts with keys:
          - ``clip_id``: stem of the JAMS file (e.g. ``"00_BN1-129-Eb_solo"``)
          - ``audio_path``: absolute path string to the mic WAV
          - ``bpm``: tempo from the JAMS tempo annotation
          - ``notes``: list of NoteEvent dicts (start, end, pitch_midi, confidence=1.0)
            sorted by (start, pitch_midi).  No string/fret — those are not needed
            for detector training; use :func:`load_guitarset_notes` for eval.
    """
    from aitabs.pipeline.audio.pitch import NoteEvent

    data_dir = Path(data_dir)
    anno_dir = data_dir / "annotation"
    audio_dir = data_dir / "audio"

    pairs: list[dict] = []
    for jams_path in sorted(anno_dir.glob("*.jams")):
        clip_id = jams_path.stem
        if "_solo" not in clip_id:
            continue

        audio_path = audio_dir / f"{clip_id}_mic.wav"
        if not audio_path.is_file():
            continue

        jam = json.loads(jams_path.read_text(encoding="utf-8"))
        annotations = jam.get("annotations", [])
        bpm = _read_tempo(annotations)

        notes: list[NoteEvent] = []
        for annotation in annotations:
            if annotation.get("namespace") != "note_midi":
                continue
            for obs in _iter_observations(annotation.get("data")):
                try:
                    start = float(obs["time"])
                    duration = float(obs.get("duration") or 0.0)
                    pitch_midi = int(round(float(obs["value"])))
                except (KeyError, TypeError, ValueError):
                    continue
                notes.append(NoteEvent(
                    start=start,
                    end=start + duration,
                    pitch_midi=pitch_midi,
                    confidence=1.0,
                ))

        notes.sort(key=lambda n: (n["start"], n["pitch_midi"]))
        pairs.append({
            "clip_id": clip_id,
            "audio_path": str(audio_path.resolve()),
            "bpm": bpm,
            "notes": [dict(n) for n in notes],
        })

    return pairs
