# Live pitch monitoring with BasicPitch — capabilities & feasibility

Goal: play guitar live and get back **pitch + per-detection confidence at low
latency**, as a tool to see where the model errs and tune thresholds interactively.

## BasicPitch capabilities (what we can use)

- **Model:** tiny, fully-convolutional CNN (~17k params, a few-MB), instrument-
  agnostic (ICASSP 2022). **No recurrent state** → runs on a sliding window.
- **Input:** audio resampled to **22050 Hz mono**; front end is a **Harmonic CQT**
  (CQT @ 3 bins/semitone + harmonic stacking).
- **Frame rate:** `FFT_HOP=256` → **~86 fps (11.6 ms/frame)**.
- **Three raw posteriorgrams** (time × freq, values in [0,1]):
  - `note` — frame activation, **(T, 88)** (one bin per piano key, MIDI 21–108).
  - `onset` — onset strength, **(T, 88)**.
  - `contour` — multipitch, **(T, 264)** at 3 bins/semitone (cents / pitch bend).
  - **A posteriorgram value is "pitch with a confidence" — exactly our signal.**
- **API:** `predict(path, model_or_model_path=…, onset_threshold, frame_threshold,
  minimum_note_length, minimum/maximum_frequency, melodia_trick, infer_onsets,
  multiple_pitch_bends)` → `(model_output, midi_data, note_events)`. Accepts a
  **preloaded model** (reuse in a loop).
- **Backends:** TensorFlow, CoreML, **TFLite**, **ONNX**, plus **basic-pitch-ts**
  (browser/Node). TFLite/ONNX are CPU-light.

## Why the default pipeline is not live (the real constraints)

1. **`predict()` is file-based** — loads a file, windows the whole thing.
2. **Note-event creation is non-causal** — it walks the posteriorgram *backward*
   (future→past). Low-latency notes are impossible from it. **For live, skip
   note_creation and threshold the posteriorgrams per-frame, causally.**
3. **Frame-time drift (issue #190):** frame index vs audio time drifts over long
   continuous output. **Timestamp from your own buffer/wall clock and re-window.**

## Feasibility — proven

- **NeuralNote** (open-source JUCE/VST) runs BasicPitch live via **RTNeural (CNN) +
  ONNXRuntime (CQT + harmonic stacking)**, CNN split into 4 sequential sub-models.
  Reported **~120 ms** added latency from the CNN context. Existence proof.
- **basic-pitch-ts + Web Audio AudioWorklet:** browser mic → live pitch.
- **Latency floor** ≈ ~120 ms (CNN context) + audio block. Fine for a monitor/
  feedback tool; not for zero-latency monitoring through an amp.

## Chosen method (this repo): Python streaming monitor

Reuses the BasicPitch we already depend on; ideal for debugging/threshold tuning.

```
mic → ring buffer (22050 Hz) → sliding ~1–2 s window → BasicPitch (preloaded)
    → take newest stabilized frames' `note`/`contour` posteriorgram
    → per-frame peak-pick (range-gated, polyphony-capped) → causal hysteresis
    → emit (pitch, confidence); plot posteriorgram heatmap; log JSONL
```

Implemented:
- `aitabs/pipeline/audio/live.py` — **pure, unit-tested core**: `RingBuffer`,
  `frame_peaks` (note + contour), `HysteresisTracker` (attack/sustain/release),
  `transcribe_posteriorgram`. No audio/model deps.
- `scripts/live_monitor.py` — mic capture (`sounddevice`) + BasicPitch inference,
  the streaming cadence, and JSONL logging. Run:

  ```bash
  python scripts/live_monitor.py --list-devices
  python scripts/live_monitor.py --onset-threshold 0.5 --log session.jsonl
  python scripts/live_monitor.py --contour      # cents-accurate head
  python scripts/live_monitor.py --no-hysteresis # raw per-frame confidence
  python scripts/live_monitor.py --plot          # scrolling posteriorgram heatmap
  ```

  The `--plot` heatmap renders the raw posteriorgram over a rolling window
  (`PosteriorgramHistory` core + matplotlib in the script): **octave ghosts show
  as a bright band a 12th/octave above the played note, sustain as a horizontal
  smear, noise as scattered speckle** — the false-positive types made visible.

### How this serves the project
Playing live, you **see when octave ghosts / sustain re-triggers / noise appear**
in the raw confidence — the same false-positive buckets `analyze_errors.py`
measures (octave / 5th / duplicate / other / isolated), but interactively. JSONL
logs become labeled material for the over-detection cases (test5/test7-style).

### Limits / honest notes
- We must write the streaming loop ourselves; BasicPitch gives the model, not a
  stream API (NeuralNote / basic-pitch-ts are the references).
- Current `scripts/live_monitor.py` infers via a short temp-WAV + `predict()` for
  a stable API; a later optimization is a **direct ONNX/TFLite call** on the window
  (drop temp-file I/O, lower latency) — the streaming core stays unchanged.
- Raw per-frame output is noisier than note_events (no backward smoothing); that's
  good for debugging, and hysteresis tames flicker without look-ahead.

## Sources
- BasicPitch paper — arXiv 2203.09893; Spotify Engineering "Meet Basic Pitch".
- spotify/basic-pitch (GitHub), DeepWiki overview, frame-drift issue #190.
- DamRsn/NeuralNote (real-time plugin), spotify/basic-pitch-ts (browser).
