"""Audio front-end curation — make real-world input resemble the (clean, dry,
level-consistent) training distribution *before* the detector sees it.

Guiding principle: a stage helps only if it moves messy audio *toward* the
GuitarSet distribution the detector was trained on — not if it colors the sound.
So this is deliberately Tier-1 only (the safe, distribution-narrowing stages):

  * DC-offset removal
  * high-pass filter at ~78 Hz (below low-E's 82 Hz → pure rumble/handling noise)
  * loudness normalization (LUFS via pyloudnorm if installed, else RMS)

Aggressive gates, creative EQ, and heavy compression are intentionally *not*
here — in testing they push audio off-distribution and hurt recall. Every stage
is toggleable so the degraded-audio A/B bench (``scripts/eval_preprocess.py``)
can attribute the gain per stage. Pure numpy/scipy — no ML dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """Toggles + parameters for :func:`preprocess_audio`."""

    remove_dc: bool = True
    highpass: bool = True
    highpass_hz: float = 78.0
    highpass_order: int = 4
    dereverb: bool = False           # Tier-3, opt-in — measure before enabling by default
    dereverb_decay: float = 0.6
    dereverb_beta: float = 0.7
    dereverb_floor: float = 0.1
    normalize: bool = True
    target_lufs: float = -18.0
    target_rms: float = 0.05  # RMS fallback when pyloudnorm is unavailable


def remove_dc_offset(y: np.ndarray) -> np.ndarray:
    """Subtract the mean (removes a DC bias that skews onset energy)."""
    return y - float(np.mean(y)) if y.size else y


def highpass_filter(
    y: np.ndarray, sr: int, *, cutoff_hz: float = 78.0, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth high-pass to strip sub-guitar rumble.

    Cutoff defaults just below low-E (82 Hz), so real note fundamentals pass
    while HVAC/handling/stand rumble — a common false-onset source — is removed.
    Uses ``filtfilt`` for zero phase shift (preserves onset timing).
    """
    from scipy.signal import butter, filtfilt

    if y.size < 3 * (order + 1):
        return y
    nyq = 0.5 * sr
    wn = min(max(cutoff_hz / nyq, 1e-4), 0.99)
    b, a = butter(order, wn, btype="highpass")
    return filtfilt(b, a, y).astype(np.float32, copy=False)


def dereverb_spectral(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = 1024,
    hop: int = 256,
    decay: float = 0.6,
    beta: float = 0.7,
    floor: float = 0.1,
) -> np.ndarray:
    """Lightweight single-channel dereverb by spectral subtraction of the tail.

    Reverb smears onset energy forward in time (sustained decay), which is a
    direct onset-recall killer. Per frequency bin, the late-reverberation
    magnitude is modeled as an exponentially-decaying trace of *past* magnitude;
    ``beta`` of it is subtracted, with a spectral ``floor`` to avoid musical
    noise. New onsets — whose magnitude rises above the decayed past — survive,
    so onsets sharpen while sustained reverb tails are attenuated. Original phase
    is retained (ISTFT), preserving timing.

    Opt-in (Tier-3): validate on the degraded bench before enabling by default.
    """
    import librosa

    if y.size < n_fft:
        return y
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    rev = np.zeros_like(mag)
    for t in range(1, mag.shape[1]):
        rev[:, t] = decay * np.maximum(rev[:, t - 1], mag[:, t - 1])
    clean = np.maximum(mag - beta * rev, floor * mag)
    out = librosa.istft(clean * np.exp(1j * phase), hop_length=hop, length=len(y))
    return out.astype(np.float32, copy=False)


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0


def normalize_loudness(
    y: np.ndarray,
    sr: int,
    *,
    target_lufs: float = -18.0,
    target_rms: float = 0.05,
    peak_ceiling: float = 0.98,
) -> np.ndarray:
    """Normalize to a consistent loudness so inputs sit where training sat.

    Uses ITU-R BS.1770 integrated loudness (pyloudnorm) when available; otherwise
    falls back to RMS normalization. Either way a peak ceiling prevents clipping.
    CRNN detectors are level-sensitive, so this recovers clips the model was
    under-responding to — a recall lever on quiet input.
    """
    if not y.size:
        return y
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if np.isfinite(loudness):
            out = pyln.normalize.loudness(y, loudness, target_lufs)
        else:  # silent / undefined loudness
            out = y
    except Exception:
        cur = _rms(y)
        out = y * (target_rms / cur) if cur > 1e-9 else y

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > peak_ceiling:
        out = out * (peak_ceiling / peak)
    return out.astype(np.float32, copy=False)


def preprocess_audio(
    y: np.ndarray, sr: int, config: PreprocessConfig | None = None
) -> np.ndarray:
    """Apply the Tier-1 front-end chain (DC → high-pass → normalize).

    Args:
        y: mono float32 waveform.
        sr: sample rate (Hz).
        config: stage toggles; defaults to the full Tier-1 chain.

    Returns:
        Processed mono float32 waveform (same length).
    """
    cfg = config or PreprocessConfig()
    out = np.asarray(y, dtype=np.float32)
    if out.ndim > 1:
        out = out.mean(axis=1).astype(np.float32)
    if cfg.remove_dc:
        out = remove_dc_offset(out)
    if cfg.highpass:
        out = highpass_filter(out, sr, cutoff_hz=cfg.highpass_hz, order=cfg.highpass_order)
    if cfg.dereverb:
        out = dereverb_spectral(out, sr, decay=cfg.dereverb_decay,
                                beta=cfg.dereverb_beta, floor=cfg.dereverb_floor)
    if cfg.normalize:
        out = normalize_loudness(out, sr, target_lufs=cfg.target_lufs, target_rms=cfg.target_rms)
    return out
