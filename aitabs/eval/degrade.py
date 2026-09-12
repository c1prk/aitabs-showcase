"""Synthesize realistic degradations of clean GuitarSet audio for a two-sided
preprocessing bench.

GuitarSet is clean/dry, so preprocessing (HPF, normalize) shows ~neutral on it —
there's nothing to fix. To measure whether a front-end *recovers* accuracy we
need degraded audio that resembles real-world conditions (reverb, noise, level
shift, sub-bass rumble). This module generates those from clean clips, so the
bench can check: (a) preprocessing doesn't regress the clean set, and (b) it
recovers recall on the degraded set.

License-clean: synthetic (numpy-generated) reverb and noise — no external IR/
sample assets. Deterministic given ``seed``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DegradeConfig:
    """Degradation parameters. ``None`` disables that stage."""

    reverb_decay_sec: float | None = 0.45  # exponential-decay synthetic room
    reverb_wet: float = 0.35
    snr_db: float | None = 18.0            # additive white noise at this SNR
    gain_db: float | None = -12.0          # quieter input (level-shift)
    rumble_hz: float | None = 45.0         # low-frequency rumble the HPF should kill
    rumble_amp: float = 0.02
    seed: int = 0


def _synth_reverb_ir(sr: int, decay_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Exponentially-decaying white-noise impulse response (a simple room)."""
    n = max(1, int(decay_sec * sr))
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-t / (decay_sec / 3.0))
    ir[0] += 1.0  # direct path
    return (ir / np.sum(np.abs(ir))).astype(np.float32)


def apply_reverb(y: np.ndarray, sr: int, decay_sec: float, wet: float,
                 rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import fftconvolve

    ir = _synth_reverb_ir(sr, decay_sec, rng)
    wetsig = fftconvolve(y, ir)[: len(y)]
    return ((1.0 - wet) * y + wet * wetsig).astype(np.float32)


def add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_p = float(np.mean(np.square(y))) or 1e-12
    noise_p = sig_p / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(len(y)) * np.sqrt(noise_p)
    return (y + noise).astype(np.float32)


def add_rumble(y: np.ndarray, sr: int, freq_hz: float, amp: float,
               rng: np.random.Generator) -> np.ndarray:
    """Add low-frequency rumble below the guitar range (HPF should remove it)."""
    t = np.arange(len(y)) / sr
    phase = rng.uniform(0, 2 * np.pi)
    return (y + amp * np.sin(2 * np.pi * freq_hz * t + phase)).astype(np.float32)


def degrade_audio(y: np.ndarray, sr: int, config: DegradeConfig | None = None) -> np.ndarray:
    """Apply the configured degradation chain to a clean waveform."""
    cfg = config or DegradeConfig()
    rng = np.random.default_rng(cfg.seed)
    out = np.asarray(y, dtype=np.float32)
    if cfg.reverb_decay_sec:
        out = apply_reverb(out, sr, cfg.reverb_decay_sec, cfg.reverb_wet, rng)
    if cfg.rumble_hz:
        out = add_rumble(out, sr, cfg.rumble_hz, cfg.rumble_amp, rng)
    if cfg.snr_db is not None:
        out = add_noise(out, cfg.snr_db, rng)
    if cfg.gain_db is not None:
        out = out * (10.0 ** (cfg.gain_db / 20.0))
    return out.astype(np.float32)
