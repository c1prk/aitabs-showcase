# Evaluation (Phase 2)

Compare **pipeline output** to **reference Guitar Pro (.gp5)** tabs and search knob settings.

## Data layout

Put pairs under `data/eval/` (gitignored audio/gp5):

```text
data/eval/
  audio/song_01.mp3
  reference/song_01.gp5
  audio/song_02.mp3
  reference/song_02.gp5
```

Or flat in one folder:

```text
data/eval/song_01.mp3
data/eval/song_01.gp5
```

Optional pre-separated stems (skips Demucs, much faster):

```text
data/eval/stems/song_01.wav
```

## Real-recording eval: GuitarSet (clean acoustic, MVP)

The reference GP5 pairs above are scored against **MIDI renders** — clean,
harmonic-poor synthetic audio. Note accuracy on **real recordings** is the real
target. [GuitarSet](https://zenodo.org/records/3371780) is the gold standard for
this: real **acoustic** guitar, recorded with a reference mic, **MIT-licensed
(commercial-safe)**, with per-string note ground truth. Each excerpt has a
`solo` (single-note) and `comp` (chord) take — for a clean-guitar MVP we use the
**`solo` + `mic`** subset.

```bash
# 1. Fetch + arrange the solo/mic subset (or pass --guitarset-dir to a local copy)
python scripts/prepare_guitarset.py --download --limit 20   # small smoke set
#   → data/eval/guitarset/{audio,annotation}/

# 2. Score the pipeline on real acoustic recordings (Demucs auto-skipped)
python scripts/eval_dataset.py data/eval/guitarset --guitarset
```

Notes:

- Ground truth is read **directly from the JAMS** (`load_guitarset_notes`), not
  via `.gp5`, so the real performed onset times are preserved for timing metrics.
- `--guitarset` implies no Demucs (the mic take is already isolated solo guitar)
  and defaults to solo-only; add `--include-comp` for chord takes,
  `--guitarset-audio mix|hex` for other variants.
- GuitarSet has no string/fret-derived rhythm grid, so only **pitch F1 / tab F1 /
  onset MAE** are reported (no GP5 rhythm scoring).

For classical (nylon) guitar, [GAPS](https://aim-qmul.github.io/GAPS/) is similar
but **non-commercial research only** — keep it for internal R&D, not shipping.

## Commands

```bash
# Score current defaults on all pairs
python scripts/eval_dataset.py data/eval

# Faster iteration on solo guitar (no Demucs)
python scripts/eval_dataset.py data/eval --no-demucs

# Search knobs (hold-out validation)
python scripts/tune_knobs.py data/eval --trials 24 --no-demucs

# Re-run with best config
python scripts/eval_dataset.py data/eval --config tune_results/best_config.json

# Fingering-only tuning (after caching audio stage once)
python scripts/prepare_eval_cache.py data/eval --no-demucs
python scripts/tune_fingering.py data/eval --cache-dir data/eval/cache --trials 40

# Full pipeline with best fingering JSON
python scripts/eval_dataset.py data/eval --config pipeline.json
# (set fingering_config_path in PipelineConfig to fingering_tune_results/best_fingering.json)
```

## Metrics

| Metric | Meaning |
|--------|---------|
| **pitch F1** | Correct MIDI pitch + onset within tolerance (default ±50 ms) |
| **tab F1** | Same + correct string and fret |
| **onset MAE** | Mean timing error for matched notes |
| **time offset** | Auto-estimated shift applied to prediction (sync) |

Reference BPM from the GP5 file can drive tempo/dedup when `use_reference_bpm` is true (default in tuning).

## Outputs

Results are written under **`data/eval_results/`** and **committed to git** (JSON only; `runs/` artifacts stay local).

- `data/eval_results/summary.json` — mean scores + config used
- `data/eval_results/<clip_id>.json` — per-clip breakdown
- `data/eval_results/<run_name>/` — optional sub-runs (e.g. `guitarset_smoke/`)

See `data/eval_results/README.md` for layout and regeneration commands.
- `tune_results/best_config.json` — best knob set from search
- `tune_results/leaderboard.json` — all trials ranked

## Knobs (`PipelineConfig`)

- BasicPitch: `confidence_threshold`, `onset_threshold`, `pitch_merge_sec`
- Onsets: `snap_to_librosa_onsets`, `librosa_onset_delta`
- Tempo: `use_reference_bpm`, `ticks_per_beat`
- Fingering: `fingering_mapper`, `fingering_config_path` (JSON from `tune_fingering.py`)
- Export rhythm: `snap_to_grid`, `chord_grid_ticks`, `duration_mode`

### Fingering (`FingeringConfig`)

Tuned separately via `scripts/tune_fingering.py` using cached notes in `data/eval/cache/`:

- `mapper`: `viterbi` (chord DP) or `greedy` (per-note)
- `string_cost`, `fret_cost`, `open_string_bonus`
- `max_fret_span`, `span_base_penalty`, `span_extra_penalty`, `low_fret_penalty`

Load/save JSON for notebook or CI:

```python
from aitabs.eval import PipelineConfig
cfg = PipelineConfig.load_json("tune_results/best_config.json")
```
