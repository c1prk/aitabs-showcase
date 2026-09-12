"""Tests for the audio augmentation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from aitabs.pipeline.audio.augment import (
    add_noise,
    augment,
    augment_kong,
    gain_jitter,
    highpass,
    pitch_shift,
    soft_clip,
)

_SR = 22050
_SINE = (np.sin(2 * np.pi * 440 * np.arange(_SR) / _SR) * 0.5).astype(np.float32)


def test_gain_jitter_changes_amplitude():
    rng = np.random.default_rng(42)
    out = gain_jitter(_SINE, rng=rng)
    assert out.shape == _SINE.shape
    assert out.dtype == _SINE.dtype
    assert not np.allclose(out, _SINE)


def test_gain_jitter_stays_within_range():
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = gain_jitter(_SINE, db_range=6.0, rng=rng)
        ratio = float(np.max(np.abs(out))) / float(np.max(np.abs(_SINE)))
        assert 10 ** (-3.0 / 20) <= ratio <= 10 ** (3.0 / 20) + 1e-6


def test_add_noise_increases_power():
    rng = np.random.default_rng(0)
    out = add_noise(_SINE, snr_db=20.0, rng=rng)
    assert float(np.mean(out ** 2)) > float(np.mean(_SINE ** 2))


def test_add_noise_preserves_shape_and_dtype():
    rng = np.random.default_rng(1)
    out = add_noise(_SINE, rng=rng)
    assert out.shape == _SINE.shape
    assert out.dtype == _SINE.dtype


def test_add_noise_silent_input_unchanged():
    silence = np.zeros(1000, dtype=np.float32)
    out = add_noise(silence, snr_db=40.0)
    np.testing.assert_array_equal(out, silence)


def test_highpass_attenuates_dc():
    dc = np.ones(_SR, dtype=np.float32) * 0.5
    out = highpass(dc, _SR, cutoff_hz=80.0)
    assert float(np.abs(out).mean()) < 0.05


def test_highpass_passes_440hz():
    out = highpass(_SINE, _SR, cutoff_hz=80.0)
    # 440 Hz is well above 80 Hz cutoff; amplitude should be largely preserved
    assert float(np.max(np.abs(out))) > 0.4


def test_highpass_preserves_shape_and_dtype():
    out = highpass(_SINE, _SR)
    assert out.shape == _SINE.shape
    assert out.dtype == _SINE.dtype


def test_augment_returns_same_shape_and_dtype():
    rng = np.random.default_rng(7)
    out = augment(_SINE, _SR, rng=rng)
    assert out.shape == _SINE.shape
    assert out.dtype == _SINE.dtype


def test_augment_is_deterministic_with_same_seed():
    out1 = augment(_SINE, _SR, rng=np.random.default_rng(99))
    out2 = augment(_SINE, _SR, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(out1, out2)


def test_augment_differs_with_different_seeds():
    out1 = augment(_SINE, _SR, rng=np.random.default_rng(1))
    out2 = augment(_SINE, _SR, rng=np.random.default_rng(2))
    assert not np.allclose(out1, out2)


# ---------------------------------------------------------------------------
# New augmentation functions (Kong fine-tuning)
# ---------------------------------------------------------------------------

_SINE_16K = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.5).astype(np.float32)


def test_soft_clip_bounds_output():
    noisy = np.random.default_rng(0).standard_normal(1000).astype(np.float32) * 5
    out = soft_clip(noisy, drive=3.0)
    # Output is bounded by 1/tanh(drive), not 1.0 (tanh saturation norm)
    import math
    upper = 1.0 / math.tanh(3.0) + 1e-4
    assert float(np.max(np.abs(out))) <= upper


def test_soft_clip_reduces_peaks():
    noisy = np.random.default_rng(0).standard_normal(1000).astype(np.float32) * 5
    out = soft_clip(noisy, drive=3.0)
    assert float(np.max(np.abs(out))) < float(np.max(np.abs(noisy)))


def test_soft_clip_identity_at_low_amplitude():
    small = np.ones(100, dtype=np.float32) * 0.01
    out = soft_clip(small, drive=1.0)
    # tanh(x)/tanh(1) ≈ x for small x; should be close to input
    assert np.allclose(out, small, atol=0.01)


def test_pitch_shift_preserves_length():
    out = pitch_shift(_SINE_16K, sr=16000, n_semitones=2)
    assert out.shape == _SINE_16K.shape


def test_pitch_shift_zero_preserves_energy():
    out = pitch_shift(_SINE_16K, sr=16000, n_semitones=0)
    # Phase vocoder at n_steps=0 is not exactly identity but energy should be similar
    assert out.shape == _SINE_16K.shape
    rms_in = float(np.sqrt(np.mean(_SINE_16K ** 2)))
    rms_out = float(np.sqrt(np.mean(out ** 2)))
    assert abs(rms_out - rms_in) < 0.05 * rms_in


def test_augment_kong_preserves_length():
    notes = [{"start": 0.1, "end": 0.5, "pitch_midi": 52}]
    out_audio, out_notes = augment_kong(_SINE_16K, sr=16000, notes=notes)
    assert out_audio.shape == _SINE_16K.shape
    assert out_audio.dtype == np.float32


def test_augment_kong_pitch_shift_updates_notes():
    notes = [{"start": 0.1, "end": 0.5, "pitch_midi": 52}]
    # With shift_range=(2,2), always shift up 2
    out_audio, out_notes = augment_kong(
        _SINE_16K, sr=16000, notes=notes,
        rng=np.random.default_rng(0),
        pitch_shift_range=(2, 2),
    )
    assert all(n["pitch_midi"] == 54 for n in out_notes)


def test_augment_kong_drops_out_of_range_notes():
    # A note at MIDI 109 (above C8=108) should be dropped after +2 semitone shift
    notes = [{"start": 0.1, "end": 0.5, "pitch_midi": 107}]
    _, out_notes = augment_kong(
        _SINE_16K, sr=16000, notes=notes,
        rng=np.random.default_rng(0),
        pitch_shift_range=(2, 2),
    )
    assert all(n["pitch_midi"] <= 108 for n in out_notes)
