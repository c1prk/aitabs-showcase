"""Low-latency live pitch monitoring on top of BasicPitch posteriorgrams.

BasicPitch's ``predict()`` is file-based and its note-event creation is
*non-causal* (it walks the posteriorgram backward), so it is unsuitable for live
use. The model itself, however, is a tiny fully-convolutional CNN with no
recurrent state, so it can be run on a sliding window and its **raw
posteriorgrams** read per-frame: each value is a pitch activation in [0, 1] — i.e.
"pitch with a confidence", exactly what a live monitor needs.

This module holds the *pure* pieces (no audio I/O, no model dependency) so they
can be unit-tested:

- :class:`RingBuffer` — fixed-size rolling audio buffer.
- :func:`frame_peaks` — per-frame peak-pick of a posteriorgram into
  ``(midi, confidence)`` with range gating and a polyphony cap.
- :class:`HysteresisTracker` — causal on/off smoothing to tame near-threshold
  flicker (separate attack vs sustain thresholds).
- :func:`transcribe_posteriorgram` — turn a (T, bins) posteriorgram into per-frame
  pitch lists using the above.

The audio capture + model inference live in ``scripts/live_monitor.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# BasicPitch frame timing (FFT_HOP=256 @ 22050 Hz).
SAMPLE_RATE = 22050
FFT_HOP = 256
FRAME_SEC = FFT_HOP / SAMPLE_RATE  # ~0.0116 s (~86 fps)

# Posteriorgram bin 0 of the 88-key "note"/"onset" heads is MIDI 21 (A0).
MIDI_BASE = 21


def freq_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * math.log2(freq / 440.0)


def midi_to_freq(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_name(midi: int) -> str:
    """MIDI number → note name with octave (A4 = 69 = 440 Hz, C4 = 60)."""
    m = int(round(midi))
    return f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"


class RingBuffer:
    """Fixed-capacity rolling buffer of mono float samples."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self.filled = 0

    def append(self, samples: np.ndarray) -> None:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = x.shape[0]
        if n >= self.capacity:
            self._buf[:] = x[-self.capacity :]
            self.filled = self.capacity
            return
        self._buf = np.roll(self._buf, -n)
        self._buf[-n:] = x
        self.filled = min(self.capacity, self.filled + n)

    def latest(self, n: int | None = None) -> np.ndarray:
        n = self.capacity if n is None else min(int(n), self.capacity)
        return self._buf[-n:].copy()


def frame_peaks(
    frame: np.ndarray,
    *,
    threshold: float = 0.5,
    min_midi: int = 40,
    max_midi: int = 88,
    max_polyphony: int = 6,
    midi_base: int = MIDI_BASE,
    bins_per_semitone: int = 1,
) -> list[tuple[float, float]]:
    """Peak-pick one posteriorgram frame into ``(midi, confidence)`` pairs.

    A bin is a peak if it is a local maximum and at least ``threshold``. Works for
    the 88-bin ``note`` head (``bins_per_semitone=1``, integer MIDI) and the
    264-bin ``contour`` head (``bins_per_semitone=3``, fractional MIDI for cents).

    Returns peaks sorted by confidence (desc), gated to ``[min_midi, max_midi]``
    and truncated to ``max_polyphony``.
    """
    f = np.asarray(frame, dtype=np.float64).reshape(-1)
    n = f.shape[0]
    peaks: list[tuple[float, float]] = []
    for b in range(n):
        c = f[b]
        if c < threshold:
            continue
        left = f[b - 1] if b > 0 else -1.0
        right = f[b + 1] if b < n - 1 else -1.0
        if c < left or c < right:
            continue  # not a local maximum
        midi = midi_base + b / bins_per_semitone
        if midi < min_midi or midi > max_midi:
            continue
        peaks.append((float(midi), float(c)))
    peaks.sort(key=lambda mc: -mc[1])
    return peaks[:max_polyphony]


@dataclass
class HysteresisTracker:
    """Causal on/off smoothing for per-frame pitch activations.

    A pitch turns ON when confidence ≥ ``onset_threshold`` and stays ON while
    confidence ≥ ``sustain_threshold``; it turns OFF after ``release_frames``
    consecutive frames below sustain. Reduces flicker near the threshold without
    look-ahead (so it stays low-latency).
    """

    onset_threshold: float = 0.5
    sustain_threshold: float = 0.3
    release_frames: int = 2
    _active: dict[int, int] = field(default_factory=dict)  # rounded-midi -> frames-below

    def update(self, peaks: list[tuple[float, float]]) -> set[int]:
        conf = {int(round(m)): c for m, c in peaks}

        # Activate new pitches that cross the onset threshold.
        for pitch, c in conf.items():
            if pitch not in self._active and c >= self.onset_threshold:
                self._active[pitch] = 0

        # Sustain / release existing pitches.
        for pitch in list(self._active):
            c = conf.get(pitch, 0.0)
            if c >= self.sustain_threshold:
                self._active[pitch] = 0
            else:
                self._active[pitch] += 1
                if self._active[pitch] > self.release_frames:
                    del self._active[pitch]

        return set(self._active)


@dataclass
class FrameResult:
    frame_index: int
    time_sec: float
    pitches: list[tuple[float, float]]  # (midi, confidence), sustained-only if tracked


def transcribe_posteriorgram(
    note: np.ndarray,
    *,
    tracker: HysteresisTracker | None = None,
    threshold: float = 0.5,
    min_midi: int = 40,
    max_midi: int = 88,
    max_polyphony: int = 6,
    bins_per_semitone: int = 1,
    start_frame: int = 0,
    time_offset_sec: float = 0.0,
) -> list[FrameResult]:
    """Convert a (T, bins) posteriorgram into per-frame pitch+confidence lists.

    If ``tracker`` is given, only hysteresis-sustained pitches are emitted (with
    the frame's measured confidence); otherwise every peak is emitted raw.
    """
    arr = np.asarray(note, dtype=np.float64)
    out: list[FrameResult] = []
    for t in range(arr.shape[0]):
        peaks = frame_peaks(
            arr[t],
            threshold=threshold,
            min_midi=min_midi,
            max_midi=max_midi,
            max_polyphony=max_polyphony,
            bins_per_semitone=bins_per_semitone,
        )
        if tracker is not None:
            active = tracker.update(peaks)
            peaks = [(m, c) for m, c in peaks if int(round(m)) in active]
        out.append(
            FrameResult(
                frame_index=start_frame + t,
                time_sec=time_offset_sec + (start_frame + t) * FRAME_SEC,
                pitches=peaks,
            )
        )
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class ControlState:
    """Live-tunable monitor controls (mutated by key presses)."""

    confidence: float = 0.5   # peak threshold to trigger a note
    sustain: float = 0.3      # threshold to keep a note (hysteresis)
    hysteresis: bool = True
    head: str = "note"        # 'note' | 'contour'
    max_polyphony: int = 6
    paused: bool = False
    quit: bool = False

    @property
    def bins_per_semitone(self) -> int:
        return 3 if self.head == "contour" else 1

    def detect_threshold(self) -> float:
        """Peak threshold fed to the picker (low enough for sustain to work)."""
        return min(self.confidence, self.sustain) if self.hysteresis else self.confidence


# key → (label) mapping is documented in the control bar; logic lives here.
def handle_key(state: ControlState, ch: str) -> bool:
    """Apply a key press to ``state``. Returns True if anything changed."""
    snapshot = (
        state.confidence, state.sustain, state.hysteresis,
        state.head, state.max_polyphony, state.paused, state.quit,
    )
    if ch == "]":
        state.confidence = round(_clamp(state.confidence + 0.05, 0.05, 0.95), 2)
    elif ch == "[":
        state.confidence = round(_clamp(state.confidence - 0.05, 0.05, 0.95), 2)
    elif ch == "'":
        state.sustain = round(_clamp(state.sustain + 0.05, 0.05, 0.95), 2)
    elif ch == ";":
        state.sustain = round(_clamp(state.sustain - 0.05, 0.05, 0.95), 2)
    elif ch in ("h", "H"):
        state.hysteresis = not state.hysteresis
    elif ch in ("c", "C"):
        state.head = "contour" if state.head == "note" else "note"
    elif ch == ".":
        state.max_polyphony = int(_clamp(state.max_polyphony + 1, 1, 8))
    elif ch == ",":
        state.max_polyphony = int(_clamp(state.max_polyphony - 1, 1, 8))
    elif ch == " ":
        state.paused = not state.paused
    elif ch in ("q", "Q", "\x03"):  # q or Ctrl-C
        state.quit = True
    after = (
        state.confidence, state.sustain, state.hysteresis,
        state.head, state.max_polyphony, state.paused, state.quit,
    )
    return snapshot != after


class PosteriorgramHistory:
    """Rolling (bins × width) display buffer of the most recent frames.

    Newest frame is the rightmost column. Pure numpy so the scroll logic is
    testable; the actual heatmap rendering (matplotlib) lives in the live script.
    """

    def __init__(self, n_bins: int, width: int) -> None:
        self.n_bins = int(n_bins)
        self.width = int(width)
        self.buf = np.zeros((self.n_bins, self.width), dtype=np.float32)

    def push(self, frames: np.ndarray) -> np.ndarray:
        """Scroll in one or more frames, shape (T, n_bins) or (n_bins,)."""
        f = np.asarray(frames, dtype=np.float32)
        if f.ndim == 1:
            f = f[None, :]
        if f.shape[1] != self.n_bins:
            raise ValueError(f"expected {self.n_bins} bins, got {f.shape[1]}")
        cols = f.T  # (n_bins, T)
        t = cols.shape[1]
        if t >= self.width:
            self.buf = cols[:, -self.width :].astype(np.float32).copy()
        else:
            self.buf = np.roll(self.buf, -t, axis=1)
            self.buf[:, -t:] = cols
        return self.buf

    def image(self) -> np.ndarray:
        """(n_bins, width) array; row 0 = lowest pitch (use imshow origin='lower')."""
        return self.buf
