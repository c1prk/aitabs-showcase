# Eval results (version-controlled)

JSON score outputs from `scripts/eval_dataset.py`. Committed so tools (e.g. Claude Code) can aggregate and compare runs without local audio.

## Layout

```text
data/eval_results/
  summary.json              # Latest full GP5 eval (all clips under data/eval/)
  test1.json                # Per-clip breakdown (pitch/tab F1, onset MAE, rhythm if enabled)
  guitarset_smoke/
    summary.json              # Sub-run aggregate
    00_BN1-129-Eb_solo.json   # Per-clip GuitarSet scores
  runs/                       # gitignored — norm WAV + pred GP5 work files
```

## Regenerate

```bash
# GP5 reference eval (your clips)
.venv/bin/python scripts/eval_dataset.py data/eval \
  --config data/eval/test1_best/best_combined.json \
  --no-demucs

# GuitarSet real-acoustic smoke (20 solo/mic clips)
.venv/bin/python scripts/eval_dataset.py data/eval/guitarset --guitarset \
  --config data/eval/test1_best/best_combined.json \
  --output-dir data/eval_results/guitarset_smoke
```

## Per-clip JSON fields

| Field | Meaning |
|-------|---------|
| `pitch_f1` | Correct MIDI + onset within tolerance |
| `tab_f1` | Correct string + fret + onset |
| `pitch_onset_mae_sec` | Mean timing error on matched notes |
| `time_offset_sec` | Auto sync shift applied to prediction |
| `rhythm` | Optional GP5 rhythm block (GP5 eval only) |

`summary.json` adds `mean_*` aggregates and the `config` used for the run.
