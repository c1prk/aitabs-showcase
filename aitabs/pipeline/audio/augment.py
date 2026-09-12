"""Audio augmentation for training data (pure numpy/scipy — no non-commercial deps).

All functions accept a float32 or float64 waveform array and return the same dtype.
Augmentation is designed to simulate pickup type, recording environment, and gain
variation without altering note pitches or onset times.
"""

from __future__ import annotations

import numpy as np


def gain_jitter(
    y: np.ndarray,
    *,
    db_range: float = 6.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a random gain shift uniformly drawn from [-db_range/2, +db_range/2] dB."""
    rng = rng or np.random.default_rng()
    db = float(rng.uniform(-db_range / 2.0, db_range / 2.0))
    return (y * (10.0 ** (db / 20.0))).astype(y.dtype)


def add_noise(
    y: np.ndarray,
    *,
    snr_db: float = 40.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add white Gaussian noise at the given signal-to-noise ratio (dB).

    At snr_db=40 the noise is inaudible but provides distributional coverage
    for detector training. Lower values (e.g. 20) simulate noisier recordings.
    """
    rng = rng or np.random.default_rng()
    signal_power = float(np.mean(y ** 2))
    if signal_power <= 0.0:
        return y
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(len(y)) * float(np.sqrt(noise_power))
    return (y + noise).astype(y.dtype)


def highpass(
    y: np.ndarray,
    sr: int,
    *,
    cutoff_hz: float = 80.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth high-pass filter to strip low-frequency rumble.

    The lowest guitar note (open low E) is ~82 Hz, so a cutoff at 80 Hz
    removes floor rumble and DC without touching guitar content.
    """
    from scipy.signal import butter, sosfilt

    nyq = sr / 2.0
    sos = butter(order, cutoff_hz / nyq, btype="high", output="sos")
    return sosfilt(sos, y).astype(y.dtype)


def pitch_shift(
    y: np.ndarray,
    sr: int,
    *,
    n_semitones: int,
) -> np.ndarray:
    """Shift pitch by n_semitones using librosa (positive = up, negative = down).

    The caller is responsible for shifting MIDI note labels by the same amount.
    Typically used during Kong fine-tuning augmentation (±2 semitones).
    """
    import librosa
    return librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=n_semitones)


def soft_clip(
    y: np.ndarray,
    *,
    drive: float = 3.0,
) -> np.ndarray:
    """Soft-clipping distortion to simulate electric guitar processing.

    Uses tanh saturation, which adds odd harmonics like a tube amp.
    drive=1.0 is clean, drive=5.0 is light overdrive, drive=10.0 is heavy.
    """
    return np.tanh(drive * y) / np.tanh(np.float32(drive))


def augment(
    y: np.ndarray,
    sr: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the full augmentation chain: gain jitter -> noise -> high-pass.

    Args:
        y: Input waveform (float32 or float64).
        sr: Sample rate in Hz.
        rng: Optional seeded generator for reproducibility.

    Returns:
        Augmented waveform with the same shape and dtype as ``y``.
    """
    rng = rng or np.random.default_rng()
    y = gain_jitter(y, rng=rng)
    y = add_noise(y, rng=rng)
    y = highpass(y, sr)
    return y


def augment_kong(
    y: np.ndarray,
    sr: int,
    notes: list[dict],
    *,
    rng: np.random.Generator | None = None,
    pitch_shift_range: tuple[int, int] = (-2, 2),
    distort_prob: float = 0.3,
    distort_drive_range: tuple[float, float] = (1.5, 4.0),
) -> tuple[np.ndarray, list[dict]]:
    """Guitar-specific augmentation chain for Kong fine-tuning.

    Applies gain jitter, optional pitch shift, optional distortion, noise, and
    high-pass filter. Returns augmented audio AND updated note labels.

    Args:
        y: Input waveform at sr Hz.
        sr: Sample rate.
        notes: List of {start, end, pitch_midi, ...} dicts.
        rng: RNG for reproducibility.
        pitch_shift_range: (min, max) semitone shift, inclusive.
        distort_prob: Probability of adding soft-clip distortion.
        distort_drive_range: (min, max) drive for soft_clip.

    Returns:
        (augmented_audio, shifted_notes)
    """
    rng = rng or np.random.default_rng()

    y = gain_jitter(y, rng=rng, db_range=6.0)

    # Pitch shift (most important augmentation for guitar generalization)
    n_st = int(rng.integers(pitch_shift_range[0], pitch_shift_range[1] + 1))
    if n_st != 0:
        y = pitch_shift(y, sr, n_semitones=n_st)
        notes = [
            {**n, "pitch_midi": int(n["pitch_midi"]) + n_st}
            for n in notes
            if 21 <= int(n["pitch_midi"]) + n_st <= 108
        ]

    # Distortion (simulates electric guitar processing)
    if rng.random() < distort_prob:
        drive = float(rng.uniform(*distort_drive_range))
        y = soft_clip(y, drive=drive)

    y = add_noise(y, snr_db=float(rng.uniform(25.0, 45.0)), rng=rng)
    y = highpass(y, sr)
    return y.astype(np.float32), notes
