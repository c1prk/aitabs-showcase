"""Librosa onset detection — precise attack timestamps for each note."""

from __future__ import annotations

import numpy as np


def detect_onsets(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
    backtrack: bool = True,
    delta: float = 0.07,
) -> np.ndarray:
    """Detect note onset times using librosa's onset envelope + peak-picking.

    Why use this alongside BasicPitch?
    BasicPitch gives you pitch + timing, but its onset times can drift by
    20–50 ms because it operates on short overlapping frames. Librosa's
    onset detector looks directly at energy transients, giving sharper
    attack alignment. The typical workflow is:
      1. Get pitches from BasicPitch
      2. Snap each pitch's start time to the nearest librosa onset

    Args:
        audio_path: Path to audio file (WAV or MP3).
        sr: Sample rate to load at. 22050 is librosa's default.
        hop_length: STFT hop in samples. 512 → ~23 ms resolution at 22050 Hz.
        backtrack: If True, slide each onset back to the nearest energy trough
            (gives the true attack rather than the spectral peak).
        delta: Minimum height of a peak in the onset strength envelope.
            Higher = fewer, more confident onsets. Tune this per dataset.

    Returns:
        1-D numpy array of onset times in seconds, sorted ascending.

    TODO (you):
        - Compare onset times against your ground-truth annotations to find
          the delta that minimizes mean absolute error on your dataset.
        - For fingerpicked playing, energy transients are softer — you may
          need to lower delta and use `wait` parameter to avoid double-triggers.
        - For strummed chords, the onset detector may fire once per strum even
          though multiple strings are hit simultaneously. This is correct behavior
          for tab purposes, but worth verifying against your ground truth.
    """
    import librosa

    y, _sr = librosa.load(audio_path, sr=sr)

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack,
        delta=delta,
    )

    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
