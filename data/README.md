# Local data

- `recordings/` — your WAV/MP4 clips (gitignored)
- `annotations/` — ground-truth tabs for eval (gitignored)
- `runs/` — Demucs stems and intermediate outputs (gitignored)

Example:

```
data/recordings/solo_01.wav
data/annotations/solo_01.json
```

### Evaluation pairs (GP5 + audio)

For `scripts/eval_dataset.py` and `scripts/tune_knobs.py`:

```
data/eval/audio/solo_01.mp3
data/eval/reference/solo_01.gp5
```

See [eval/README.md](../eval/README.md).
