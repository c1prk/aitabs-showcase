# Eval analysis — Phase 2 (regenerated 2026-05-29)

Config: `data/eval/test1_best/best_combined.json`, `--no-demucs`, fixed GP5
loader (mid-song tempo changes + capo/tuning concert pitch).

## Headline numbers (before → after tempo fix)

| Set | Clips | pitch F1 | tab F1 | onset MAE |
|-----|------:|---------:|-------:|----------:|
| **test1–12 songs** (old loader) | 11 | 0.558 | 0.429 | 18 ms |
| **test1–12 songs** (this run) | 11 | **0.626** | **0.479** | 17 ms |
| **GuitarSet solo** (unchanged) | 20 | 0.776 | 0.423 | 18 ms |

Rhythm (GP5 export vs reference): pos F1 **0.741**, note-value acc **0.707**, rest F1 **0.218**.

**+0.068 pitch F1** and **+0.050 tab F1** vs the pre-fix snapshot — almost entirely from
honest reference timing (test1) plus capo-aware import on capped clips.

## Per-clip breakdown (this run)

| Clip | pitch F1 | tab F1 | P | R | pred/ref | offset | mode |
|------|---------:|-------:|--:|--:|---------:|-------:|------|
| test8 River Flows | 0.921 | 0.734 | 0.93 | 0.91 | 0.97 | −10 ms | ✅ strong |
| test9 Kiki | 0.908 | 0.687 | 0.90 | 0.92 | 1.02 | −50 ms | ✅ strong |
| test6 True Colors | 0.907 | 0.743 | 0.89 | 0.93 | 1.05 | −10 ms | ✅ strong |
| test1 | **0.779** | 0.628 | 0.78 | 0.77 | 0.99 | 0 ms | ✅ was 0.18 (tempo fix) |
| test12 Coco | 0.758 | 0.616 | 0.75 | 0.76 | 1.01 | 0 ms | ✅ good |
| test4 Fur Elise | 0.654 | 0.424 | 0.75 | 0.58 | 0.77 | +60 ms | ⚠️ recall-limited |
| test3 Claude de Lune | 0.531 | 0.295 | 0.54 | 0.52 | 0.96 | **+580 ms** | ⚠️ alignment |
| test11 My Fav | 0.507 | 0.403 | 0.47 | 0.55 | 1.17 | −10 ms | ⚠️ over-detect |
| test10 WY | 0.432 | 0.378 | 0.46 | 0.41 | 0.90 | −40 ms | ⚠️ mixed |
| test5 OTR | 0.288 | 0.234 | 0.25 | 0.34 | **1.35** | −10 ms | ❌ over-detect |
| test7 Let Her Go | 0.202 | 0.131 | 0.17 | 0.24 | **1.39** | −10 ms | ❌ over-detect |

Onset MAE on matched notes stays ~3–36 ms — timing is tight when notes align.

## Key findings

### 1. Tempo-change ground truth fix — confirmed

test1 pitch F1 **0.18 → 0.779** after `load_gp5_notes` honours `mixTableChange.tempo`
(120 → 190 BPM at measure 25). Note counts were always right (783/794); the old
loader drifted reference times by ~62 s at the end.

### 2. Two failure modes remain

- **Over-detection (low precision):** test7, test5, test11 — pred/ref **1.17–1.39**.
  Harmonic / sustain / octave ghosts from BasicPitch on fingerpicked open-string material.
  Onset MAE is excellent (3–11 ms) so this is not a timing problem.
- **Alignment / recall:** test3 (+580 ms offset, half-step-down tuning), test4 (recall 0.58).

### 3. Strong clips are near-ceiling

test6/8/9 all **≥0.90 pitch F1** — the tuned pipeline is sound on clean material.

### 4. GuitarSet vs song clips

GuitarSet solo (0.776 pitch) still beats the song-set mean (0.626) because real
acoustic mic takes are cleaner than user recordings, and the config was MIDI-tuned.
Song clips add harder real-world cases (capo, over-detect, long-form alignment).

## Recommended next steps

1. **test3 alignment** — investigate +0.58 s constant offset (half-step-down tuning
   in reference; possible stem/clip boundary or tempo origin mismatch).
2. **Harmonic post-filter** — opt-in knob to drop octave/overtone ghosts; A/B on test7/test5.
3. **GuitarSet retune** — lower `onset_threshold` / `pitch_merge_sec` for recall on real audio.
4. **Keep pitch F1 as north star** for note accuracy; rhythm pos F1 ~0.74 is already solid.

Regenerate:

```bash
.venv/bin/python scripts/eval_dataset.py data/eval \
  --config data/eval/test1_best/best_combined.json --no-demucs
```
