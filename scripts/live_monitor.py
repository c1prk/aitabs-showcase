#!/usr/bin/env python3
"""Live guitar pitch monitor: mic → BasicPitch posteriorgram → pitch + confidence.

Streams audio from an input device, runs BasicPitch on a sliding window, and
prints (and optionally logs) per-frame pitch detections with their confidence —
using the raw ``note`` posteriorgram, not the non-causal note-event creation, so
it stays low-latency. Intended as a debugging / threshold-tuning tool: play, and
watch when octave ghosts / sustain re-triggers / noise appear.

Requires the full runtime: ``sounddevice`` (audio I/O) + ``basic-pitch`` (model).
The pitch-extraction core (``aitabs.pipeline.audio.live``) is dependency-light and
unit-tested separately.

Examples::

    python scripts/live_monitor.py --list-devices
    python scripts/live_monitor.py --onset-threshold 0.5 --log session.jsonl
    python scripts/live_monitor.py --contour   # cents-accurate (264-bin head)

Notes:
- Inherent latency is ~120 ms (BasicPitch CNN context) plus the block size.
- BasicPitch frame indexing drifts over long streams (issue #190); we re-window
  and timestamp from wall clock, never from a long continuous frame counter.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

from aitabs.pipeline.audio.live import (
    FRAME_SEC,
    MIDI_BASE,
    SAMPLE_RATE,
    ControlState,
    HysteresisTracker,
    PosteriorgramHistory,
    RingBuffer,
    freq_to_midi,
    handle_key,
    midi_to_name,
    transcribe_posteriorgram,
)


def _controls_bar(st: ControlState) -> list[str]:
    onoff = "on" if st.hysteresis else "off"
    paused = "   ⏸ PAUSED" if st.paused else ""
    return [
        f"  conf [ ] {st.confidence:.2f}    sustain ; ' {st.sustain:.2f}    "
        f"hysteresis (h) {onoff}{paused}",
        f"  head (c) {st.head}    polyphony , . {st.max_polyphony}    "
        f"(space) pause    (q) quit",
    ]


class TerminalUI:
    """Clean in-place 'now playing' display: note names + confidence bars."""

    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    def __init__(self, *, head: str, min_midi: int, max_midi: int, bar_width: int = 28) -> None:
        self.head = head
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.bar_width = bar_width
        self._started = False

    def start(self) -> None:
        sys.stdout.write("\033[2J\033[?25l")  # clear screen, hide cursor
        self._started = True

    def _bar(self, conf: float) -> str:
        filled = max(0, min(self.bar_width, int(round(conf * self.bar_width))))
        color = self._GREEN if conf >= 0.7 else self._YELLOW
        return f"{color}{'█' * filled}{self._DIM}{'·' * (self.bar_width - filled)}{self._RESET}"

    def render(
        self,
        active: list[tuple[float, float]],
        *,
        elapsed: float,
        controls: list[str] | None = None,
    ) -> None:
        lines = [
            f"  🎸 live pitch monitor   "
            f"{self._DIM}{elapsed:5.1f}s{self._RESET}",
            "",
        ]
        if not active:
            lines.append(f"     {self._DIM}… listening …{self._RESET}")
        else:
            for midi, conf in sorted(active, key=lambda mc: mc[0]):
                name = midi_to_name(round(midi))
                lines.append(f"   {name:>4}  {self._bar(conf)}  {conf:0.2f}")
        # Pad to a stable height so old rows are overwritten cleanly.
        while len(lines) < 11:
            lines.append("")
        if controls:
            lines.append(f"{self._DIM}{'─' * 60}{self._RESET}")
            lines.extend(controls)
        sys.stdout.write("\033[H" + "\n".join(f"\033[K{ln}" for ln in lines) + "\n")
        sys.stdout.flush()

    def stop(self) -> None:
        if self._started:
            sys.stdout.write("\033[?25h\n")  # show cursor
            sys.stdout.flush()


class LiveHeatmap:
    """Scrolling posteriorgram heatmap (matplotlib). Ghosts show as bright bands.

    Kept here (not in the package core) so matplotlib stays an optional, runtime-
    only dependency of the live script.
    """

    def __init__(
        self,
        n_bins: int,
        *,
        seconds: float,
        bins_per_semitone: int,
        min_midi: int,
        max_midi: int,
    ) -> None:
        import matplotlib.pyplot as plt  # noqa: PLC0415 (optional runtime dep)

        self._plt = plt
        width = max(int(seconds / FRAME_SEC), 1)
        self.history = PosteriorgramHistory(n_bins, width)
        top_midi = MIDI_BASE + n_bins / bins_per_semitone

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 4))
        self.im = self.ax.imshow(
            self.history.image(),
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
            extent=(-seconds, 0.0, MIDI_BASE, top_midi),
            interpolation="nearest",
        )
        self.ax.set_ylim(min_midi, max_midi)
        self.ax.set_xlabel("time (s, now = 0)")
        self.ax.set_ylabel("MIDI pitch")
        self.ax.set_title("BasicPitch posteriorgram (live)")
        self.fig.colorbar(self.im, ax=self.ax, label="confidence")
        self.fig.tight_layout()

    def update(self, frames: np.ndarray) -> None:
        img = self.history.push(frames)
        self.im.set_data(img)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def alive(self) -> bool:
        return bool(self._plt.fignum_exists(self.fig.number))


def _build_model_infer(min_freq: float, max_freq: float):
    """Return a callable: window samples (float32 @ 22050) -> posteriorgram dict.

    The dict has 'note' (T,88), 'onset' (T,88) and 'contour' (T,264) so the head
    can be switched live. Uses BasicPitch via a short temp WAV + ``predict`` with
    a preloaded model (stable API; can later be a direct ONNX/TFLite call).
    """
    import os
    import tempfile

    import soundfile as sf
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    from aitabs.pipeline.audio.basic_pitch_infer import quiet_basic_pitch_output

    # Silence TensorFlow / CoreML chatter before the backend loads.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    with quiet_basic_pitch_output():
        model = Model(ICASSP_2022_MODEL_PATH)

    def infer(window: np.ndarray) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, window, SAMPLE_RATE)
            with quiet_basic_pitch_output():
                model_output, _midi, _events = predict(
                    tmp.name,
                    model_or_model_path=model,
                    minimum_frequency=min_freq,
                    maximum_frequency=max_freq,
                )
        return {k: np.asarray(v) for k, v in model_output.items()}

    return infer


class KeyPoller:
    """Non-blocking single-key reader (Unix cbreak). No-op where unavailable."""

    def __init__(self) -> None:
        self.enabled = False

    def __enter__(self) -> "KeyPoller":
        try:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # leaves ISIG on, so Ctrl-C still works
            self.enabled = True
        except Exception:  # noqa: BLE001 - not a tty / Windows
            self.enabled = False
        return self

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        import select

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if ready else None

    def __exit__(self, *_exc) -> None:
        if self.enabled:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live BasicPitch pitch+confidence monitor")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--window-sec", type=float, default=2.0, help="Sliding window length (s)")
    parser.add_argument("--update-hz", type=float, default=10.0, help="Inference updates per second")
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--sustain-threshold", type=float, default=0.3)
    parser.add_argument("--min-freq", type=float, default=82.0, help="E2 by default")
    parser.add_argument("--max-freq", type=float, default=1319.0, help="~E6 by default")
    parser.add_argument("--max-polyphony", type=int, default=6)
    parser.add_argument("--contour", action="store_true", help="Use 264-bin contour head (cents)")
    parser.add_argument("--no-hysteresis", action="store_true", help="Emit raw peaks (no smoothing)")
    parser.add_argument("--log", type=Path, default=None, help="Append per-frame results as JSONL")
    parser.add_argument("--plot", action="store_true", help="Show a live scrolling posteriorgram heatmap")
    parser.add_argument("--plot-seconds", type=float, default=6.0, help="Heatmap time window (s)")
    parser.add_argument("--raw", action="store_true", help="Print one line per frame instead of the clean UI")
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        print(f"sounddevice required for live capture: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.list_devices:
        print(sd.query_devices())
        return

    min_midi = int(round(freq_to_midi(args.min_freq)))
    max_midi = int(round(freq_to_midi(args.max_freq)))

    state = ControlState(
        confidence=args.onset_threshold,
        sustain=args.sustain_threshold,
        hysteresis=not args.no_hysteresis,
        head="contour" if args.contour else "note",
        max_polyphony=args.max_polyphony,
    )

    infer = _build_model_infer(args.min_freq, args.max_freq)
    buffer = RingBuffer(int(args.window_sec * SAMPLE_RATE))
    tracker = HysteresisTracker(
        onset_threshold=state.confidence, sustain_threshold=state.sustain
    )

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, _frames, _time, status):  # sounddevice thread
        if status:
            print(status, file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    log_fh = args.log.open("a", encoding="utf-8") if args.log else None
    period = 1.0 / max(args.update_hz, 1e-3)
    fresh_frames = max(int(period / FRAME_SEC) + 2, 1)

    heatmap = None
    if args.plot:
        n_bins = 264 if state.head == "contour" else 88
        try:
            heatmap = LiveHeatmap(
                n_bins,
                seconds=args.plot_seconds,
                bins_per_semitone=state.bins_per_semitone,
                min_midi=min_midi,
                max_midi=max_midi,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"--plot disabled (matplotlib unavailable): {exc}", file=sys.stderr)

    interactive = not args.raw and not args.plot
    ui = TerminalUI(head=state.head, min_midi=min_midi, max_midi=max_midi) if interactive else None
    if ui is not None:
        ui.start()
    else:
        print(
            f"Listening @ {SAMPLE_RATE} Hz | window {args.window_sec}s | "
            f"range {min_midi}-{max_midi} | head {state.head} | Ctrl+C to stop",
            flush=True,
        )

    start_t = time.time()
    last_active: list = []
    keys = KeyPoller() if interactive else contextlib.nullcontext()
    try:
        with keys, sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, device=args.device,
            blocksize=int(SAMPLE_RATE * period), callback=callback,
        ):
            while True:
                # Handle live controls (interactive terminal only).
                if interactive:
                    ch = keys.poll()
                    while ch is not None:
                        handle_key(state, ch)
                        ch = keys.poll()
                    if state.quit:
                        break
                    tracker.onset_threshold = state.confidence
                    tracker.sustain_threshold = state.sustain

                got = False
                while not audio_q.empty():
                    buffer.append(audio_q.get())
                    got = True

                if state.paused:
                    if ui is not None:
                        ui.render(last_active, elapsed=time.time() - start_t, controls=_controls_bar(state))
                    time.sleep(period / 2)
                    continue

                if not got or buffer.filled < SAMPLE_RATE // 2:
                    if ui is not None:
                        ui.render(last_active, elapsed=time.time() - start_t, controls=_controls_bar(state))
                    time.sleep(period / 2)
                    continue

                out = infer(buffer.latest())
                arr = out[state.head]
                if heatmap is not None:
                    heatmap.update(arr[-fresh_frames:])
                    if not heatmap.alive():
                        break
                frames = transcribe_posteriorgram(
                    arr[-fresh_frames:],
                    tracker=tracker if state.hysteresis else None,
                    threshold=state.detect_threshold(),
                    min_midi=min_midi,
                    max_midi=max_midi,
                    max_polyphony=state.max_polyphony,
                    bins_per_semitone=state.bins_per_semitone,
                    time_offset_sec=time.time(),
                )

                if log_fh:
                    for fr in frames:
                        if fr.pitches:
                            log_fh.write(json.dumps({"t": fr.time_sec, "pitches": fr.pitches}) + "\n")

                if ui is not None:
                    last_active = frames[-1].pitches if frames else []
                    ui.render(last_active, elapsed=time.time() - start_t, controls=_controls_bar(state))
                elif args.raw:
                    for fr in frames:
                        if fr.pitches:
                            label = "  ".join(f"{midi_to_name(round(m))}@{c:.2f}" for m, c in fr.pitches)
                            print(label, flush=True)
                time.sleep(period / 2)
    except KeyboardInterrupt:
        pass
    finally:
        if ui is not None:
            ui.stop()
        else:
            print("\nstopped", flush=True)
        if log_fh:
            log_fh.close()


if __name__ == "__main__":
    main()
