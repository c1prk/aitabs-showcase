"""Tests for the live pitch-monitor core (no audio / model needed)."""

from __future__ import annotations

import numpy as np

import pytest

from aitabs.pipeline.audio.live import (
    MIDI_BASE,
    HysteresisTracker,
    PosteriorgramHistory,
    RingBuffer,
    frame_peaks,
    freq_to_midi,
    transcribe_posteriorgram,
)


def _note_frame(active: dict[int, float], n: int = 88) -> np.ndarray:
    """Build an 88-bin frame; active maps MIDI pitch -> confidence."""
    f = np.zeros(n)
    for midi, conf in active.items():
        f[midi - MIDI_BASE] = conf
    return f


def test_freq_to_midi():
    assert abs(freq_to_midi(440.0) - 69.0) < 1e-9
    assert abs(freq_to_midi(82.41) - 40.0) < 0.02  # E2


def test_midi_to_name():
    from aitabs.pipeline.audio.live import midi_to_name

    assert midi_to_name(69) == "A4"
    assert midi_to_name(60) == "C4"
    assert midi_to_name(40) == "E2"   # low open E
    assert midi_to_name(64) == "E4"   # high open e
    assert midi_to_name(61) == "C#4"
    assert midi_to_name(61.4) == "C#4"  # rounds


def test_ring_buffer_rolls():
    rb = RingBuffer(5)
    rb.append(np.array([1, 2, 3], dtype=np.float32))
    assert rb.filled == 3
    rb.append(np.array([4, 5, 6], dtype=np.float32))  # overflow by 1
    assert rb.filled == 5
    assert list(rb.latest()) == [2, 3, 4, 5, 6]
    assert list(rb.latest(2)) == [5, 6]


def test_ring_buffer_large_append_truncates():
    rb = RingBuffer(4)
    rb.append(np.arange(10, dtype=np.float32))
    assert list(rb.latest()) == [6, 7, 8, 9]


def test_frame_peaks_picks_local_maxima_and_gates_range():
    # E2 (40, strong), E3 (52, medium), and an out-of-range A0 (21).
    frame = _note_frame({40: 0.9, 52: 0.6, 21: 0.95})
    peaks = frame_peaks(frame, threshold=0.5, min_midi=40, max_midi=88)
    midis = [m for m, _ in peaks]
    assert 40 in midis and 52 in midis
    assert 21 not in midis  # gated out (below E2)
    assert peaks[0][0] == 40  # sorted by confidence desc


def test_frame_peaks_threshold_and_polyphony_cap():
    frame = _note_frame({40: 0.9, 44: 0.85, 47: 0.8, 52: 0.4})
    peaks = frame_peaks(frame, threshold=0.5, min_midi=40, max_midi=88, max_polyphony=2)
    assert len(peaks) == 2  # capped
    assert 52 not in [m for m, _ in peaks]  # below threshold


def test_frame_peaks_contour_fractional_midi():
    # 264-bin contour, 3 bins/semitone: a peak 1 bin above A4 base -> +1/3 semitone.
    n = 264
    f = np.zeros(n)
    bin_idx = (69 - MIDI_BASE) * 3 + 1
    f[bin_idx] = 0.9
    peaks = frame_peaks(f, threshold=0.5, min_midi=21, max_midi=108, bins_per_semitone=3)
    assert abs(peaks[0][0] - (69 + 1 / 3)) < 1e-6


def test_hysteresis_attack_sustain_release():
    tr = HysteresisTracker(onset_threshold=0.6, sustain_threshold=0.3, release_frames=2)
    # below onset -> not active
    assert tr.update([(40, 0.5)]) == set()
    # crosses onset -> active
    assert tr.update([(40, 0.7)]) == {40}
    # sustains above sustain threshold
    assert tr.update([(40, 0.4)]) == {40}
    # drop below sustain: stays for release_frames, then releases
    assert tr.update([]) == {40}   # below=1
    assert tr.update([]) == {40}   # below=2
    assert tr.update([]) == set()  # below=3 > release_frames -> off


def test_control_state_keys():
    from aitabs.pipeline.audio.live import ControlState, handle_key

    st = ControlState(confidence=0.5, sustain=0.3)
    assert handle_key(st, "]") and st.confidence == 0.55
    assert handle_key(st, "[") and st.confidence == 0.5
    assert handle_key(st, "'") and st.sustain == 0.35
    assert handle_key(st, ";") and st.sustain == 0.3
    assert handle_key(st, "h") and st.hysteresis is False
    assert handle_key(st, "c") and st.head == "contour" and st.bins_per_semitone == 3
    assert handle_key(st, "c") and st.head == "note" and st.bins_per_semitone == 1
    assert handle_key(st, ".") and st.max_polyphony == 7
    assert handle_key(st, ",") and st.max_polyphony == 6
    assert handle_key(st, " ") and st.paused is True
    assert handle_key(st, "q") and st.quit is True
    # unknown key → no change
    assert not handle_key(st, "z")


def test_control_state_clamps_and_detect_threshold():
    from aitabs.pipeline.audio.live import ControlState, handle_key

    st = ControlState(confidence=0.92, sustain=0.3)
    handle_key(st, "]")
    handle_key(st, "]")
    assert st.confidence <= 0.95  # clamped
    # hysteresis: detect threshold is the lower of the two so sustain can work
    st2 = ControlState(confidence=0.6, sustain=0.3, hysteresis=True)
    assert st2.detect_threshold() == 0.3
    st2.hysteresis = False
    assert st2.detect_threshold() == 0.6


def test_posteriorgram_history_scrolls_newest_right():
    h = PosteriorgramHistory(n_bins=4, width=3)
    h.push(np.array([[1, 0, 0, 0]], dtype=np.float32))  # one frame
    h.push(np.array([[0, 2, 0, 0]], dtype=np.float32))
    img = h.image()
    assert img.shape == (4, 3)
    # newest frame is the rightmost column
    assert list(img[:, -1]) == [0, 2, 0, 0]
    assert list(img[:, -2]) == [1, 0, 0, 0]


def test_posteriorgram_history_overflow_keeps_latest():
    h = PosteriorgramHistory(n_bins=2, width=2)
    frames = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)  # 3 frames into width-2
    h.push(frames)
    img = h.image()
    assert list(img[:, -1]) == [1, 1]   # newest
    assert list(img[:, -2]) == [0, 1]   # one before


def test_posteriorgram_history_rejects_wrong_bins():
    h = PosteriorgramHistory(n_bins=4, width=3)
    with pytest.raises(ValueError):
        h.push(np.zeros((1, 3), dtype=np.float32))


def test_transcribe_posteriorgram_with_tracker():
    frames = np.stack([
        _note_frame({40: 0.8}),
        _note_frame({40: 0.5, 52: 0.7}),
    ])
    tr = HysteresisTracker(onset_threshold=0.6, sustain_threshold=0.3)
    out = transcribe_posteriorgram(frames, tracker=tr, threshold=0.4, min_midi=40, max_midi=88)
    assert len(out) == 2
    assert [m for m, _ in out[0].pitches] == [40]
    got = sorted(m for m, _ in out[1].pitches)
    assert got == [40, 52]
    assert out[1].time_sec > out[0].time_sec
