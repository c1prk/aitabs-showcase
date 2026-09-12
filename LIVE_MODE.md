# Live mode — guide

`scripts/live_monitor.py` runs BasicPitch on your **live mic input** and prints
the detected pitches with their confidence in real time. It reads BasicPitch's
raw posteriorgram per-frame (not the offline note events), so it's low-latency
and shows the model's *raw* confidence — ideal for hearing/seeing where it errs
and for tuning thresholds.

> Expect ~120 ms latency (BasicPitch's CNN context) plus your audio block. It's a
> monitoring/debugging tool, not zero-latency monitoring through an amp.

## 1. Install

Already in `requirements.txt`: `basic-pitch`, `soundfile`, `matplotlib`,
`sounddevice`. The one system dependency is **PortAudio** (for mic capture):

```bash
# Linux:
sudo apt-get install libportaudio2
# macOS:
brew install portaudio
# then (in your venv):
pip install -r requirements.txt
```

Plug in / select your guitar input (audio interface or a mic in front of the
guitar). The model resamples everything to 22050 Hz mono internally.

## 2. Pick your input device

```bash
python scripts/live_monitor.py --list-devices
```

Note the index of your interface/mic, e.g. `2`, and pass it with `--device 2`
(omit to use the system default input).

## 3. Run it

```bash
# simplest: default device, smoothed output
python scripts/live_monitor.py

# choose a device + show the live heatmap
python scripts/live_monitor.py --device 2 --plot
```

By default you get a **clean in-place display** that refreshes in place (no
scrolling, no BasicPitch debug spam) showing the notes sounding *now* with a
confidence bar:

```
  🎸 live pitch monitor   head=note   range E2–E6    12.4s  Ctrl+C to stop

    E3  ██████████████████████····  0.84
    B3  ████████████████··········  0.57
```

Note names use scientific pitch (A4 = 440 Hz): E2 = low open E, E4 = high open e.

### Live controls (no restart needed)

The clean UI shows a control bar and responds to single key presses, so you can
tune while playing:

| Key | Control |
|-----|---------|
| `[` / `]` | confidence threshold − / + (trigger sensitivity) |
| `;` / `'` | sustain threshold − / + (hysteresis release) |
| `h` | toggle hysteresis smoothing on/off |
| `c` | switch head note ↔ contour (cents-accurate) |
| `,` / `.` | max polyphony − / + |
| `space` | pause / resume |
| `q` | quit |

Raise `]` (confidence) until ghost notes disappear; the value you settle on maps
to the eval config's `confidence_threshold` / `onset_threshold`. (Live keys work
in the default terminal UI; `--raw` and `--plot` are non-interactive.)

For a raw, per-frame text dump instead (good for piping/grepping), add `--raw`:

```
E3@0.84  B3@0.57      <- one line per analysis frame
```

## 4. The heatmap (`--plot`) — see the errors

```bash
python scripts/live_monitor.py --plot --no-hysteresis
```

A scrolling image of the raw posteriorgram (time → right, pitch → up, brightness =
confidence). Play a single note and watch what lights up:

- **Octave ghost** → a bright band ~12 semitones **above** the note you played.
- **Sustain re-trigger** → a horizontal **smear** trailing after the note.
- **Noise** → scattered **speckle** with no clear pitch.

This is the visual version of the false-positive buckets in `analyze_errors.py`.

## 5. A debugging session recipe

1. Start raw, no smoothing, with the plot:
   ```bash
   python scripts/live_monitor.py --plot --no-hysteresis --log session.jsonl
   ```
2. Play slow single notes up the neck. Confirm the bright bin matches the pitch
   and watch for octave bands / smears.
3. Play the passage that over-detects in batch (e.g. fast fingerpicking). See
   which artifact dominates.
4. Raise/lower `--onset-threshold` until ghosts disappear without losing real
   notes. That threshold transfers back to the eval config
   (`confidence_threshold` / `onset_threshold`).
5. Review `session.jsonl` afterwards (one JSON object per emitted frame:
   `{"t": <unix_seconds>, "pitches": [[midi, confidence], ...]}`).

## 6. Useful flags

| Flag | Default | What it does |
|------|--------:|--------------|
| `--list-devices` | — | List input devices and exit |
| `--device N` | system default | Input device index |
| `--onset-threshold` | 0.5 | Confidence to *start* a note (raise to cut ghosts) |
| `--sustain-threshold` | 0.3 | Confidence to *keep* a note (hysteresis) |
| `--no-hysteresis` | off | Emit raw per-frame peaks (most sensitive; best for debugging) |
| `--contour` | off | Use the 264-bin head (cents-accurate; shows bends/vibrato) |
| `--min-freq` / `--max-freq` | 82 / 1319 Hz | Pitch gate (E2–E6 = the guitar range) |
| `--max-polyphony` | 6 | Max simultaneous notes per frame |
| `--window-sec` | 2.0 | Sliding analysis window |
| `--update-hz` | 10 | Inference updates per second (higher = lower latency, more CPU) |
| `--plot` / `--plot-seconds` | off / 6 | Live heatmap and its time span |
| `--raw` | off | Per-frame text dump instead of the clean UI |
| `--log FILE` | — | Append per-frame results as JSONL |

## 7. Troubleshooting

- **`sounddevice` import error / no PortAudio** → install `libportaudio2`
  (Linux) or `portaudio` (macOS), then `pip install sounddevice`.
- **No output while playing** → wrong device (`--list-devices`), input gain too
  low, or pitch outside `--min-freq/--max-freq`. Try `--no-hysteresis` and lower
  `--onset-threshold 0.3`.
- **Too many ghost notes** → raise `--onset-threshold`; narrow the freq range.
- **Laggy / high CPU** → lower `--update-hz` (e.g. 6) or `--window-sec` (e.g. 1.5).
- **Plot doesn't open / headless machine** → drop `--plot` and use `--log`; the
  monitor degrades gracefully if matplotlib can't open a window.

See `RESEARCH_LIVE.md` for how this works under the hood and its limits.
