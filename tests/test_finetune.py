"""Tests for finetune.py helpers — no model or audio I/O required."""
from __future__ import annotations

import numpy as np
import pytest

from aitabs.pipeline.audio.finetune import (
    _MIDI_OFFSET,
    _N_PITCHES,
    _SR,
    _WINDOW_FRAMES,
    _WINDOW_SAMPLES,
    _FPS,
    _iter_windows,
    _window_labels,
)


# ── _window_labels ─────────────────────────────────────────────────────────────

def _note(start, end, pitch):
    return {"start": start, "end": end, "pitch_midi": pitch, "confidence": 1.0}


def test_window_labels_shape():
    note_gt, onset_gt = _window_labels([], t_start=0.0)
    assert note_gt.shape == (_WINDOW_FRAMES, _N_PITCHES)
    assert onset_gt.shape == (_WINDOW_FRAMES, _N_PITCHES)


def test_window_labels_dtype():
    note_gt, onset_gt = _window_labels([], t_start=0.0)
    assert note_gt.dtype == np.float32
    assert onset_gt.dtype == np.float32


def test_window_labels_no_notes_all_zero():
    note_gt, onset_gt = _window_labels([], t_start=0.0)
    assert note_gt.sum() == 0.0
    assert onset_gt.sum() == 0.0


def test_window_labels_active_note_sets_frames():
    # A4 = MIDI 69; bin = 69 - 21 = 48
    pitch = 69
    note = _note(start=0.0, end=_WINDOW_SAMPLES / _SR, pitch=pitch)
    note_gt, _ = _window_labels([note], t_start=0.0)
    assert note_gt[:, pitch - _MIDI_OFFSET].sum() > 0


def test_window_labels_onset_only_at_first_frame():
    pitch = 60
    t_start = 0.0
    note = _note(start=0.0, end=0.5, pitch=pitch)
    _, onset_gt = _window_labels([note], t_start=t_start)
    col = onset_gt[:, pitch - _MIDI_OFFSET]
    assert col.sum() == pytest.approx(1.0)
    assert col[0] == pytest.approx(1.0)


def test_window_labels_note_outside_window_ignored():
    window_dur = _WINDOW_SAMPLES / _SR
    note = _note(start=window_dur + 0.1, end=window_dur + 0.5, pitch=60)
    note_gt, onset_gt = _window_labels([note], t_start=0.0)
    assert note_gt.sum() == 0.0
    assert onset_gt.sum() == 0.0


def test_window_labels_note_before_window_ignored():
    note = _note(start=-0.5, end=-0.1, pitch=60)
    note_gt, onset_gt = _window_labels([note], t_start=0.0)
    assert note_gt.sum() == 0.0
    assert onset_gt.sum() == 0.0


def test_window_labels_out_of_midi_range_ignored():
    # MIDI 8 is below A0 (21); MIDI 120 is above C8 (108)
    low_note = _note(start=0.0, end=0.2, pitch=8)
    high_note = _note(start=0.0, end=0.2, pitch=120)
    note_gt, onset_gt = _window_labels([low_note, high_note], t_start=0.0)
    assert note_gt.sum() == 0.0
    assert onset_gt.sum() == 0.0


def test_window_labels_note_spanning_window_boundary():
    window_dur = _WINDOW_SAMPLES / _SR
    # note starts in window, ends after it
    pitch = 50
    note = _note(start=window_dur - 0.1, end=window_dur + 0.2, pitch=pitch)
    note_gt, onset_gt = _window_labels([note], t_start=0.0)
    col = note_gt[:, pitch - _MIDI_OFFSET]
    assert col.sum() > 0  # at least some frames are active
    assert onset_gt[:, pitch - _MIDI_OFFSET].sum() == pytest.approx(1.0)


def test_window_labels_note_enters_from_before():
    # Note started before this window — no onset in this window
    pitch = 45
    note = _note(start=-0.1, end=0.3, pitch=pitch)  # started before t_start=0
    note_gt, onset_gt = _window_labels([note], t_start=0.0)
    col_note = note_gt[:, pitch - _MIDI_OFFSET]
    col_onset = onset_gt[:, pitch - _MIDI_OFFSET]
    assert col_note.sum() > 0        # note is active in window
    assert col_onset.sum() == 0.0   # but no onset (started before window)


# ── _iter_windows ──────────────────────────────────────────────────────────────

def _make_audio(n_seconds: float) -> np.ndarray:
    return np.zeros(int(n_seconds * _SR), dtype=np.float32)


def test_iter_windows_count_exact_multiple():
    n_windows = 3
    y = _make_audio(n_windows * _WINDOW_SAMPLES / _SR)
    windows = _iter_windows(y, [])
    assert len(windows) == n_windows


def test_iter_windows_remainder_dropped():
    # 3.7 windows worth of audio → 3 windows
    y = _make_audio(3.7 * _WINDOW_SAMPLES / _SR)
    windows = _iter_windows(y, [])
    assert len(windows) == 3


def test_iter_windows_audio_shape():
    y = _make_audio(2 * _WINDOW_SAMPLES / _SR)
    windows = _iter_windows(y, [])
    audio, note_gt, onset_gt = windows[0]
    assert audio.shape == (_WINDOW_SAMPLES, 1)
    assert note_gt.shape == (_WINDOW_FRAMES, _N_PITCHES)
    assert onset_gt.shape == (_WINDOW_FRAMES, _N_PITCHES)


def test_iter_windows_too_short_returns_empty():
    y = _make_audio(0.5)  # shorter than one window
    assert _iter_windows(y, []) == []


def test_iter_windows_custom_stride():
    y = _make_audio(3 * _WINDOW_SAMPLES / _SR)
    half_stride = _WINDOW_SAMPLES // 2
    windows = _iter_windows(y, [], stride=half_stride)
    # With stride = W/2 and 3W of audio, positions: 0, W/2, W, 3W/2, 2W → 5 windows
    assert len(windows) == 5


# ── FinetunedDetector ─────────────────────────────────────────────────────────

def test_finetuned_detector_get_detector():
    from aitabs.pipeline.audio.pitch import FinetunedDetector, get_detector
    det = get_detector("finetuned", model_path="/fake/path")
    assert isinstance(det, FinetunedDetector)


def test_get_detector_still_raises_on_unknown():
    from aitabs.pipeline.audio.pitch import get_detector
    with pytest.raises(ValueError, match="Unknown detector"):
        get_detector("definitely_not_real")
