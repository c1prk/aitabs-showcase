"""Fusion layer — combine audio note events with vision finger positions.

This module solves the core problem that audio-only tools cannot:
  Audio tells you WHAT pitch was played and WHEN.
  Vision tells you WHERE on the fretboard the fingers were.

Together they give you string + fret — the information needed for tabs.

The fusion strategy:
  1. For each audio-detected note (timestamp + MIDI pitch), find the
     video frame(s) that correspond to that timestamp.
  2. From those frames, get the finger positions (from MediaPipe).
  3. Filter the candidate (string, fret) positions for that pitch to only
     those a finger is actually touching.
  4. If exactly one position remains: assign it. If zero or multiple: fall
     back to the DP-based fingering optimizer from aitabs.mapping.fingering.
"""

from __future__ import annotations

from typing import TypedDict


class FusedNote(TypedDict):
    start: float
    end: float
    pitch_midi: int
    string: int       # 0 = low E … 5 = high e
    fret: int
    confidence: float
    source: str       # "vision" | "audio_fallback" | "dp_fallback"


def _cell(c) -> "tuple[int, int]":
    """Normalize a vision cell to (string, fret) — accepts a tuple or a dict."""
    if isinstance(c, dict):
        return int(c["string"]), int(c["fret"])
    return int(c[0]), int(c[1])


def _vision_resolve(note: dict, candidates, frames, times, window_sec):
    """Pick the candidate (string, fret) a finger was most-often on near the onset.

    Returns (string, fret) or None if no candidate was seen on any finger in the
    onset window. ``candidates`` are the physically-playable positions for the note's
    pitch; only those a finger actually touched can win — that's the whole point.
    """
    import bisect
    from collections import Counter

    onset = note["start"]
    lo = bisect.bisect_left(times, onset - window_sec)
    hi = bisect.bisect_right(times, max(onset + window_sec, note["end"]))
    if hi <= lo:
        return None
    counts: "Counter[tuple[int,int]]" = Counter()
    cand_set = set(candidates)
    for f in frames[lo:hi]:
        for c in f.get("cells", []):
            cell = _cell(c)
            if cell in cand_set:
                counts[cell] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def fuse(
    audio_notes: list[dict],
    vision_frames: list[dict],
    *,
    onset_window_sec: float = 0.066,
    setup=None,
) -> list[FusedNote]:
    """Merge audio note events with per-frame vision detections into tab notes.

    Audio owns *pitch* + *timing* (reliable, ~0.92 F1); vision resolves *which
    (string, fret)* among the physically-playable positions for that pitch — the
    axis audio is capped on (~0.65). Each note is resolved by vision when a finger
    was seen on a valid position near its onset, else by a deterministic
    lowest-fret fallback. The ``source`` field lets you measure the vision
    contribution (dev rule: only ship if the vision-resolved fraction lifts tab F1).

    Args:
        audio_notes: ``[{start, end, pitch_midi, confidence}]`` from the detector.
        vision_frames: time-ordered ``[{timestamp, cells}]`` where ``cells`` is the
            list of ``(string, fret)`` (or ``{"string","fret"}``) a fingertip was on
            that frame — produced upstream by MediaPipe + the fretboard homography
            (``fretboard_geometry.nearest_cell``).
        onset_window_sec: ± window around each onset to look for the finger (covers
            audio ~20 ms jitter + video ~33 ms/frame at 30 fps).
        setup: optional ``GuitarSetup`` (tuning/capo) passed to ``midi_to_positions``.

    Returns:
        One :class:`FusedNote` per input note, with ``source`` in
        ``{"vision", "audio_fallback"}``.
    """
    from aitabs.pipeline.mapping.guitar import midi_to_positions

    frames = sorted(vision_frames, key=lambda f: f.get("timestamp", 0.0))
    times = [f.get("timestamp", 0.0) for f in frames]

    out: list[FusedNote] = []
    for note in audio_notes:
        cands = (midi_to_positions(note["pitch_midi"], setup=setup) if setup is not None
                 else midi_to_positions(note["pitch_midi"]))
        if not cands:
            continue  # pitch outside the guitar range (shouldn't survive the detector)
        resolved = _vision_resolve(note, cands, frames, times, onset_window_sec)
        if resolved is not None:
            string, fret = resolved
            source = "vision"
        else:
            string, fret = min(cands, key=lambda c: c[1])  # lowest playable fret
            source = "audio_fallback"
        out.append(FusedNote(
            start=float(note["start"]), end=float(note["end"]),
            pitch_midi=int(note["pitch_midi"]), string=int(string), fret=int(fret),
            confidence=float(note.get("confidence", 1.0)), source=source,
        ))
    return out


def source_distribution(fused: list[FusedNote]) -> dict[str, float]:
    """Fraction of notes resolved by each source — the vision-contribution metric."""
    if not fused:
        return {}
    from collections import Counter
    c = Counter(n["source"] for n in fused)
    return {k: v / len(fused) for k, v in c.items()}
