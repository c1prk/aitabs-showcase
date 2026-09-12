# Plan — improving note accuracy on real guitar recordings

Status north star: **pitch F1** (correct MIDI pitch + onset within ±50 ms).
Current: GuitarSet solo (real acoustic) **0.776**; test1–12 songs **0.626**
(after the GP5 tempo + capo fixes). See `data/eval_results/ANALYSIS.md`.

## Diagnosis — three failure buckets

Mapping each clip's score to its material (from the committed `.gp5`s):

- **A. Harmonic / sustain over-detection (precision).** Biggest cluster.
  Let Her Go (pred/ref **1.39**, capo 7), OTR (**1.35**), My Fav (**1.17**, dense
  chords) predict far more notes than truth. Onset MAE on matched notes is tiny
  (3–11 ms), so this is octave/overtone/ringing ghosts, not timing.
- **B. Score-time ≠ performance-time (eval artifact).** A `.gp5` encodes
  *notated* onsets. For **MIDI renders** that equals performance time (test1 works),
  but for an **expressive real recording** (Claude de Lune, rubato) the notated grid
  never aligns — a single constant offset can't track rubato, so correct pitches
  still miss the ±50 ms window (test3: +580 ms offset, balanced counts, 0.53 F1).
- **C. Under-detection (recall).** Fast/quiet passages (Fur Elise, recall 0.58)
  and the recall-limited GuitarSet story (0.83 P / 0.73 R).

Clean clips (River Flows / Kiki / True Colors / test1) already score 0.78–0.92, so
the pipeline is fundamentally sound; the losses are material- and timbre-dependent.

## Tracks

### Track 0 — Separate "detection quality" from "timing mismatch"  ✅ DONE
Result (`data/eval_results/synthetic_gap.json`): test7/test5 have high synth
ceilings (0.88–0.89) but low real scores (0.20–0.29), pred/ref 1.35–1.39 →
**bucket A (real-audio harmonic ghosts)**, not rubato. test3 synth 0.88 / real
0.53 with +580 ms offset → **bucket B (timing/rubato)**. test6/8/9 synth ≈ real
→ near ceiling. Confirms Track 2 is the right next lever.

<details><summary>original Track 0 description</summary>
Render every `.gp5` to a clean synthetic WAV (mechanical timing = the reference's
own note times) and eval it. The synthetic score is the **detection ceiling** with
the timing-mismatch variable removed; the **gap between synthetic and real-audio**
scores tells us, per clip, whether the loss is timbre/detection (bucket A/C) or
performance timing (bucket B). Cheapest, most clarifying step; needs no new audio.

Deliverables: `aitabs/eval/render_gp5.py`, `scripts/diagnose_synthetic_gap.py`.
</details>

### Track 2 — Cut over-detection  ← v1 (octave filter) REJECTED; re-diagnosing
**v1 result: net negative.** `suppress_octave_harmonics` (coincident ±12/19/24,
not-stronger) dropped pitch F1 on *every* clip (mean −0.017): −0.060 on clean
test6 (removed 75 real octaves) while only −0.001 on target test7. Conclusion:
the test7/test5 over-detection is **not** exact-octave coincident ghosts, and
coincident-onset is too weak a guard against real octave doublings. Kept in code
but **default off**; not recommended until the actual FP type is known.

**Re-diagnose first (data-driven):** classify what the false positives actually
are before building another filter.

```bash
python scripts/eval_dataset.py data/eval \
  --config data/eval/test1_best/best_combined.json --no-demucs \
  --dump-notes data/eval_results/notes
python scripts/analyze_errors.py data/eval_results/notes
```

`aitabs/eval/error_analysis.py` buckets each unmatched prediction vs concurrent
reference: octave / fifth-or-12th / duplicate-sustain / other-wrong / isolated.
The dominant bucket on test7/test5 picks the next move:
- octave/5th high → smarter overtone removal (spectral fundamental check, not just
  interval+onset) or a guitar-aware model;
- duplicate high → tune `pitch_merge_sec` / min-note-length (sustain re-triggers);
- other/isolated high → genuine detection noise → preprocessing or Track 4 model.

### Track 1 — Trust the right benchmark
Make **GuitarSet (real *performed* onsets)** the primary real-recording north star.
Treat notated-gp5 song clips as a secondary, MIDI-render-fair set. Adopt **mir_eval**
for standardized note/onset metrics, and add a tempo-robust (offset-curve / DTW)
matching mode so rubato pieces report pitch accuracy honestly.

### Track 2 follow-ups (after A/B)
If the octave filter helps but plateaus: add sustain/overlap pruning and a
min-note-length bump, then re-tune `confidence` / `onset` on the corrected metric.

### Track 3 — Recall-first config for clean solo
Lower `onset_threshold` / `confidence_threshold` / `pitch_merge_sec`, validated on
GuitarSet. Kept separate from the precision config — one global config can't win
both over- and under-detection.

### Track 4 — Model upgrade and training path (active direction)
Post-processing is no longer the main bet. The upstream detector is the ceiling:
BasicPitch misses inner voices in dense let-ring polyphony and can emit false
positives that simple octave filtering does not fix.

See `RESEARCH_MODELS.md` for the current Tier 0-3 roadmap:
- Tier 0: record the owned five-clip held-out eval set and never train on it.
- Tier 1: add a pluggable detector backend and A/B BasicPitch, recall-first
  BasicPitch, and license-approved guitar-specialized models.
- Tier 2: fine-tune on commercial-safe paired data such as GuitarSet and verified
  GOAT, plus augmentation.
- Tier 3: build the data/model moat with a larger sequence model and provenance
  tracking.

Do not use GAPS/non-commercial data for product training. Do not use CC YouTube
as labeled training data; reserve it for later unlabeled robustness/domain work.

## Order
Current order:

1. Record/check the owned held-out clips (`RECORDING_GUIDE.md`,
   `scripts/check_recordings.py`).
2. Run eval plus note dumps, then `scripts/analyze_errors.py`.
3. Add the Tier 1 pluggable detector backend and A/B backends.
4. Move to Tier 2 fine-tuning once the A/B harness is reproducible.
5. Keep rhythm/rubato/meter work moving, but do not let it distract from the
   upstream pitch bottleneck.

## Research basis
- Octave/harmonic errors dominate transcription false positives — arXiv 1706.08231.
- mir_eval onset match standard is ±50 ms — github.com/craffel/mir_eval.
- Synthetic→real gap & domain adaptation for guitar — arXiv 2402.15258, GAPS 2408.08653.
