"""Render a GP5 reference to a clean synthetic WAV (Track 0 diagnostic).

Purpose: produce audio whose timing is *mechanically identical* to the reference
note times (from :func:`load_gp5_notes`, which is tempo- and capo-correct). Running
the pipeline on this synthetic render measures the **detection ceiling** with the
score-vs-performance timing-mismatch variable removed. Comparing the synthetic
score to the real-recording score isolates, per clip, whether accuracy loss comes
from timbre/detection or from expressive timing.

The synth is a lightweight additive plucked-string model (numpy only): a few
decaying harmonics per note. It is intentionally simple and clean — not a guitar
emulation — so it stresses pitch/onset detection without inventing the harmonic
chaos of real recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .gp5_import import load_gp5_notes

# Relative amplitudes of the first few harmonics of a plucked string.
_HARMONICS: tuple[float, ...] = (1.0, 0.5, 0.33, 0.25)


def midi_to_hz(midi: int) -> float:
    return 440.0 * 2.0 ** ((float(midi) - 69.0) / 12.0)


def _pluck(freq: float, dur_sec: float, sr: int, decay: float = 5.0) -> np.ndarray:
    """One plucked-string note: decaying sum of harmonics."""
    n = max(int(dur_sec * sr), 1)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-decay * t)
    sig = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate(_HARMONICS, start=1):
        partial = freq * k
        if partial >= sr / 2:  # skip above Nyquist
            break
        sig += amp * np.sin(2.0 * np.pi * partial * t)
    return sig * env


def render_gp5_to_wav(
    gp5_path: str | Path,
    out_wav: str | Path,
    *,
    sr: int = 22050,
    ring_sec: float = 0.25,
    tail_sec: float = 0.5,
    track_index: int = 0,
) -> str:
    """Synthesize a clean WAV from a GP5 reference's note events.

    Args:
        gp5_path: Reference Guitar Pro file.
        out_wav: Destination WAV path.
        sr: Sample rate.
        ring_sec: Extra ring time appended to each note's duration (sustain).
        tail_sec: Silence appended after the last note.
        track_index: Which GP5 track to render.

    Returns:
        The output WAV path (as str).
    """
    notes, _bpm, _setup = load_gp5_notes(gp5_path, track_index=track_index)
    out_wav = str(out_wav)
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

    if not notes:
        sf.write(out_wav, np.zeros(int(sr * tail_sec), dtype=np.float32), sr)
        return out_wav

    total_sec = max(n["end"] for n in notes) + ring_sec + tail_sec
    buf = np.zeros(int(total_sec * sr) + 1, dtype=np.float64)

    for note in notes:
        freq = midi_to_hz(note["pitch_midi"])
        dur = max(note["end"] - note["start"], 0.08) + ring_sec
        seg = _pluck(freq, dur, sr)
        start = int(note["start"] * sr)
        end = min(start + len(seg), len(buf))
        buf[start:end] += seg[: end - start]

    peak = float(np.max(np.abs(buf)))
    if peak > 0:
        buf = 0.9 * buf / peak

    sf.write(out_wav, buf.astype(np.float32), sr)
    return out_wav
