"""Tests for the Tier-1 audio preprocessing front-end and degradation synth."""

from __future__ import annotations

import numpy as np
import pytest

from aitabs.pipeline.audio.preprocess import (
    PreprocessConfig,
    dereverb_spectral,
    highpass_filter,
    normalize_loudness,
    preprocess_audio,
    remove_dc_offset,
)
from aitabs.eval.degrade import DegradeConfig, add_noise, apply_reverb, degrade_audio


SR = 16000


def _sine(freq: float, dur: float = 1.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(y):
    return float(np.sqrt(np.mean(np.square(y))))


def test_remove_dc_offset():
    y = _sine(200.0) + 0.5  # add DC bias
    out = remove_dc_offset(y)
    assert abs(float(np.mean(out))) < 1e-4


def test_highpass_attenuates_rumble_passes_guitar():
    rumble = _sine(45.0)     # below low-E → should be cut
    note = _sine(200.0)      # in guitar range → should pass
    r_out = highpass_filter(rumble, SR, cutoff_hz=78.0)
    n_out = highpass_filter(note, SR, cutoff_hz=78.0)
    assert _rms(r_out) < 0.25 * _rms(rumble)   # rumble strongly attenuated
    assert _rms(n_out) > 0.8 * _rms(note)      # note mostly preserved


def test_normalize_lifts_level_and_respects_ceiling():
    y = _sine(200.0, amp=0.01)  # very quiet
    out = normalize_loudness(y, SR, target_lufs=-18.0, target_rms=0.05)
    assert _rms(out) > _rms(y) * 2             # quiet input made louder
    assert float(np.max(np.abs(out))) <= 0.98 + 1e-6  # no clipping


def test_normalize_lufs_when_available():
    pyln = pytest.importorskip("pyloudnorm")
    y = _sine(200.0, amp=0.02)
    out = normalize_loudness(y, SR, target_lufs=-18.0)
    meter = pyln.Meter(SR)
    assert abs(meter.integrated_loudness(out) - (-18.0)) < 1.0  # hits LUFS target


def test_preprocess_preserves_length_and_runs():
    y = _sine(200.0) + 0.3 * _sine(45.0) + 0.2
    out = preprocess_audio(y, SR)
    assert out.shape == y.shape
    assert np.isfinite(out).all()


def test_preprocess_toggles_off():
    y = _sine(200.0)
    out = preprocess_audio(y, SR, PreprocessConfig(remove_dc=False, highpass=False, normalize=False))
    assert np.allclose(out, y, atol=1e-6)


def test_add_noise_lowers_snr():
    y = _sine(200.0)
    noisy = add_noise(y, 10.0, np.random.default_rng(0))
    resid = noisy - y
    assert _rms(resid) > 0                       # noise was added
    assert not np.allclose(noisy, y)


def test_degrade_deterministic_and_length_preserved():
    y = _sine(200.0)
    a = degrade_audio(y, SR, DegradeConfig(seed=1))
    b = degrade_audio(y, SR, DegradeConfig(seed=1))
    assert a.shape == y.shape
    assert np.allclose(a, b)                      # deterministic given seed
    assert not np.allclose(a, y)                  # actually degraded


def test_degrade_disable_all_is_identity_ish():
    y = _sine(200.0)
    out = degrade_audio(y, SR, DegradeConfig(reverb_decay_sec=None, snr_db=None,
                                             gain_db=None, rumble_hz=None))
    assert np.allclose(out, y, atol=1e-6)


def test_dereverb_attenuates_reverb_tail():
    # An onset burst followed by silence; reverb smears energy into the tail.
    y = np.zeros(SR, dtype=np.float32)
    y[:1600] = _sine(200.0, dur=0.1)          # 100 ms burst, then silence
    rev = apply_reverb(y, SR, decay_sec=0.4, wet=0.6, rng=np.random.default_rng(0))
    drv = dereverb_spectral(rev, SR)
    tail = slice(4000, SR)                      # well after the burst
    assert _rms(drv[tail]) < _rms(rev[tail])    # tail energy reduced
    assert drv.shape == y.shape
    assert np.isfinite(drv).all()


def test_dereverb_via_config_runs():
    y = _sine(200.0) + 0.2
    out = preprocess_audio(y, SR, PreprocessConfig(dereverb=True))
    assert out.shape == y.shape
    assert np.isfinite(out).all()
