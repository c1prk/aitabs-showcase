# CONTEXT — project state & handoff

Snapshot for continuing work (e.g. in Cursor). Ties together the focused docs:
`PLAN.md`, `RESEARCH_RHYTHM.md`, `RESEARCH_LIVE.md`, `LIVE_MODE.md`,
`data/eval_results/ANALYSIS.md`.

## What this project is
Audio (+ future vision) → guitar tablature. Core stages:
`Demucs → BasicPitch → tempo/meter → note cleanup → fingering (string/fret) →
GP5/MIDI export`. Focus so far: **note accuracy** and **rhythm notation**.
Commercial intent: a startup — **license-safe only** (see "Licensing").

## Current accuracy (data/eval_results/)
- **GuitarSet solo (real acoustic, MIT data):** pitch F1 **0.810** (BasicPitch calibrated, onset=0.6, frame=0.3, conf=0.5 — best calibrated). BasicPitch fine-tuned on GuitarSet: **0.808** (calibrated: onset=0.2, frame=0.1, conf=0.5). Fine-tuning did not beat baseline.
- **Song clips (test1–12, MIDI renders of GP5):** mean pitch F1 **0.626**; strong clips (River Flows/Kiki/True Colors) ≥0.90; weak clips test5/test7 ~0.2–0.3 (over-detection), test3 rubato.
- **vidtest1 (real *acoustic* guitar, self-recorded video, no Demucs):** pitch F1 **0.448** — SAME for BasicPitch and fine-tuned Kong (step-4000). NOTE: reference is a notated GP5, so onset/notated-time mismatch likely caps F1 for both (a known eval-methodology confound, not necessarily a detector limit). This is a real off-GuitarSet-distribution owned clip; the GuitarSet fine-tune gave no gain here — but the confound means we can't cleanly call it "no generalization" without a timing-robust metric.
- Onset MAE ~17 ms (timing is tight when notes match).

## Tier 3 — Kong model sprint (started 2026-06-29)
Active 7-day sprint to implement Kong et al. (2021) CRNN:
- **Day 1 (2026-06-29):** `KongDetector` wired into harness. Zero-shot GuitarSet eval running. IMPORTANT: zero-shot Kong on guitar is NOT better than calibrated BasicPitch — expected ~55-65% F1 due to guitar vs. piano onset mismatch. Fine-tuning is required for the 87% target.
- Zero-shot results (first 9 clips): F1 range 0.21-0.91, high precision (0.6-0.95), low recall — consistent with piano threshold miscalibration for guitar.
- Zero-shot calibrated: F1 **0.537** (180 clips). Confirms fine-tuning is required.
- **Fine-tuning target:** ~87-91% F1 on GuitarSet after 10K steps overnight.
- **Day 2 (2026-06-30):** Fine-tuning underway on GuitarSet (players 01–05 train, player 00 val). Two bugs fixed that had blocked all learning:
  1. **Double-sigmoid**: Kong's `note_model` outputs are already sigmoid probabilities [0,1]; the loss was applying `.sigmoid()` again, collapsing everything to ~0.5 (constant BCE 5.544, zero gradient). Fixed in `finetune_kong.py::compute_loss` (use `F.binary_cross_entropy` directly).
  2. **Checkpoint format**: `PianoTranscription.load_state_dict` expects nested `{'note_model':{...},'pedal_model':{...}}` with prefix-stripped keys; PyTorch's flat `state_dict()` did not match. Fixed via `_nested_state()`. All saved checkpoints now load cleanly.
  - After fixes, loss dropped 5.544 → ~0.020 (real learning), plateauing ~step 1900.
  - **Step-2500 checkpoint, 5-clip calibration: pitch F1 0.9046** (onset=0.2, frame=0.1, conf=0.42). Optimum pinned at the grid floor → extended `calibrate_detector.py` grid down to onset=0.05, frame=0.05 for the final calibration.
  - Caveat: BasicPitch's calibration-set F1 (0.8761) runs higher than its full 180-clip eval (0.810). The 0.9046 above is a 5-clip calibration number — the real comparison vs 0.810 needs the full 180-clip eval on the final checkpoint (in progress).
  - Training resumed step 2500 → 5000 (lr=1e-4, accum=1, save every 500). Final checkpoint → `data/models/kong_guitarset_ft/kong_ft_final.pth`.
  - **Held-out result (fine-tuned Kong, 5000 steps, player-00 30 clips): pitch F1 = 0.8096** at onset=0.1/frame=0.1/conf=0.2 (`calibration_kong_final.json`). Ultra-low thresholds (onset=0.05) reach ~0.826 but are impractically slow to sweep.
  - **Verdict: this recipe does NOT beat BasicPitch.** Same held-out 30 clips: BasicPitch baseline = **0.8761**, fine-tuned Kong = **0.81–0.826**. (BasicPitch full 180-clip = 0.810; but that is the fair bar only for a model not trained on GuitarSet — Kong was, so its held-out number is what counts.)
  - **Underfitting signature**: the calibration optimum pins at the grid floor and the model's frame/onset probabilities are muted (frame_output max ~0.38), so notes are only recovered at very low thresholds. Training loss plateaued at ~0.02 by step ~1900 and did not improve to step 5000. Likely levers: more steps (5000 << Riley et al.'s 100K), lighter/again-tuned augmentation, or onset-loss weighting (5x on a sparse target invites predicting ~0). Next: 180-clip eval to measure train-vs-held-out gap (confirm underfit vs overfit), then an improved-recipe retrain.
  - Infra hardening (Day 2): background jobs kept dying — root cause was harness-"killed" tasks leaving **orphaned python processes** that starved CPU (sweep crawled at ~25 s/combo). Fixed by killing orphans by explicit PID / using TaskStop. Made heavy scripts resumable: `calibrate_detector.py --cache-dir` (npz inference cache) + resumable sweep sidecar; `eval_dataset.py --resume`.
  - **FIX WORKS — positive-class weighting (`--pos-weight`)**: retrained fresh from pretrained with `--pos-weight 5` into `data/models/kong_guitarset_ft_posw5/`. Loss no longer collapses to 0.02 (sits ~0.07, reflecting real note prediction). **At only step 1000, held-out F1 = 0.8301** (onset=0.2/frame=0.1/conf=0.2) — already beats the plain-BCE 5000-step model (0.8129) with 1/5 the steps, and the onset-threshold optimum lifted off the grid floor (0.1→0.2), confirming outputs un-muted.
  - **GPU RUN BEATS BASICPITCH.** Moved training to Colab GPU (T4) via `scripts/colab_finetune_kong.py` (self-contained, batch=8, augment OFF — we were underfitting, so aug hurt). pos_weight=5, fresh from pretrained. Held-out player-00 (30 clips), same harness/BasicPitch bar (0.8761):
    - step 2000 (~16k sample-presentations): **F1 = 0.9056** (onset=0.6)
    - step 4000: **F1 = 0.9089** (onset=0.4/frame=0.2/conf=0.2)
    - Smooth climb, both **decisively above BasicPitch's 0.8761** (+0.033). The onset optimum moved from the muted model's 0.1 floor to 0.4–0.6, hard-confirming the outputs un-muted. Calibration grid extended up to onset=0.8 (`_ONSET_GRID`) to fit the now-confident model.
  - **Interpretation of the 0.81→0.90 jump**: not "2× steps" — it's ~16× more effective training (batch 8 vs CPU batch 1), augment off (fits the clean eval distribution), better gradients, and F1's threshold-sensitivity causing a fast crossover as outputs un-mute. The CPU run would likely have reached ~0.90 too, just far slower — which is why we went GPU.
  - **step 14000 (~112k presentations): F1 = 0.8985** — *below* the step-4000 peak (0.9089). Held-out rose to ~4000 then declined → **mild overfitting** (augment OFF + only 150 clips → memorization by ~112k presentations). So **step 4000 is the current best model** (0.9089, onset=0.4/frame=0.2/conf=0.2). pw=5's ceiling on this recipe is ~0.91.
  - **Best model = `kong_ft_step4000.pth`** (GPU pw5). Running full 180-clip eval on it now (`data/eval_results/kong_gpu_s4000_180/`, thresholds onset=0.4/frame=0.2/conf=0.2) for the complete number + train-vs-held-out split cross-check.
  - **Full-pipeline vs raw — eval-methodology fix (IMPORTANT)**: the full eval pipeline scored step-4000 held-out at only **0.818**, vs raw detector **0.909**. Root cause is NOT the detector (both extraction paths give byte-identical notes) and NOT the note-merge (`pitch_merge=0` → still 0.818). It is **`_apply_rhythm_quantization` (default `rhythm_quantizer="adaptive"`)** snapping note onsets to a beat grid — and it runs *even with `--no-rhythm`* (that flag only disables rhythm scoring). On real-performance timing this moves onsets off the true times and breaks the 50 ms pitch match. Fix: measure pitch/onset accuracy with **`--rhythm-quantizer none`** → held-out recovers to **0.8976** (≈ raw). The same artifact deflated BasicPitch (0.876 raw → 0.810 full-eval). New eval_dataset flags: `--pitch-merge-sec`, `--rhythm-quantizer`, `--clip-filter`, `--limit-clips`.
  - **vidtest1** (owned acoustic self-recorded video, notated-GP5 reference): the 0.448 was a **metric artifact**, not a detector limit. GP5 places onsets on a constant 96 bpm grid; the performance was ~94.6 bpm with drift, and the metric only corrected a constant offset. Kong step-4000 actually gets **pitch-multiset 0.913** (time-agnostic) and, with the new tempo-robust matcher, **F1 0.862 (`--align warp`)** / 0.80 (`--align dtw`) — so the detector **does** generalize to the owned recording. **Metric fix shipped**: `compare_note_lists(align=...)` + `eval_dataset --align none|warp|dtw`; `warp` = global linear tempo warp (use for notated GP5 refs), `none` = constant offset (correct for performance-timed GuitarSet JAMS). Latent `load_gp5_notes` bugs still open (ties emit phantom onsets; repeats not expanded; dead/ghost notes kept; voice-2 ignored) — matter for other GP5 clips, not vidtest1.
  - **To push past 0.91 (next)**: turn augmentation back ON for long runs (regularizes → the missing ingredient that let 14k overfit; matches Riley's aug + ~100k recipe); optionally calibrate 6000/8000 to pinpoint the peak; try pw=3. Validate the winner on the owned off-distribution clips (`RECORDING_GUIDE.md`). Licensing: base weights are MAESTRO-pretrained (research-only) — clean re-train needed before shipping.

### Kong licensing note
Architecture: Apache 2.0 (bytedance/piano_transcription). Weights: CC-BY-4.0 (Zenodo 4034264). KNOWN TODO: weights pre-trained on MAESTRO (CC BY-NC-SA 4.0). Fine-tuning data: GuitarSet (MIT) — clean. For commercial release: need clean pre-train from scratch using MIDI+FluidR3 (public domain). This is the post-rough-milestone Phase 2 task.

### Pluggable detector state (Tier 1/2A complete)
`get_detector(name)` factory supports: `"basic_pitch"`, `"recall_first"`, `"finetuned"` (TF BasicPitch), `"kong"` (Kong CRNN, PyTorch).
Calibration files: `data/eval_results/calibration_baseline.json` (F1=0.8761), `data/eval_results/calibration_finetuned.json` (F1=0.8630).

## Tablature (string/fret) accuracy — research + prototype (2026-07-01)
Pitch detection is strong (Kong 0.90) but **TAB is the weak link**: held-out
player-00 **tab F1 = 0.530** (string+fret), i.e. only **59% of correctly-pitched
notes land on the right string/fret** — the current fingering *optimizer* guesses
string with no audio evidence.

SOTA research (TabCNN ISMIR'19, FretNet, TART'25): don't do "pitch then guess
fingering" — predict **string+fret directly from audio**. FretNet is MIT but
ships **no pretrained weights** (must train from scratch on a heavy dep chain).
Decision: **build our own** string-aware model (owns IP, reuses our infra).
Key simplification: pitch is known (Kong) → only predict **string** (6-way,
feasibility-masked); fret = pitch − open-string pitch.

**Prototype findings (audio-only string accuracy, held-out player-00):**
- fingering heuristic (current): 0.590
- timbre CNN on onset CQT patch: 0.59–0.61 (per-note timbre is a *weak* cue)
- **timbre CNN + Viterbi neck-position continuity: 0.646** (best; lam=0.6)

**Conclusion: audio-only string caps ~0.65.** A full TabCNN/FretNet joint model
might reach ~0.7–0.8 note-level, but the same-pitch/multiple-string ambiguity is
a hard wall for audio. **Vision (fretboard + hand tracking) is the real unlock**
for accurate tabs — it directly observes string/fret. This empirically justifies
prioritizing the vision-fusion track. Prototypes in scratchpad (not committed);
approach: `aitabs/pipeline/mapping/` string model when productionized.

## Rhythm — replace heuristic quantizer with PM2S (2026-07-01)
Old rhythm (librosa tempo + heuristic snap in `_apply_rhythm_quantization`) is
"completely off" — it never tracked beat phase/downbeats. Correct approach is
**performance-to-score conversion** (beat/downbeat tracking → metrical quantize).
Adopting **PM2S** (Liu/Kong/Benetos, ISMIR 2022): CRNN, **MIT, pretrained (Zenodo
10520196), no madmom, CPU-fast**. API: `CRNNJointPM2S().convert(perf_midi,
score_midi, end_time=<dur>, include_time_signature=True)`; also
`RNNJointBeatProcessor().process(midi) -> beats, downbeats` and
`RNNJointQuantisationProcessor -> (onset_positions_beats, note_values_beats)`.

**Validated on vidtest1**: PM2S beats land exactly on the notated grid (phase
correct). On the CLEAN reference notes it produces textbook rhythm (onset_beats
1,2,3…,8,8.5,9,9.5; durations 1.0/0.5 = clean quarters/eighths). On our detection
it's mostly clean with some messy values (0.38/0.88/0.33) — from residual
detection ghosts/jitter, NOT PM2S. So **PM2S is the right engine; remaining error
is detection-cleanliness**. Note PM2S tracks at the 8th level (2× the notated
quarter) — benign metrical choice. Piano-trained (ASAP); guitar fine-tune on
GuitarSet beats (MIT) is the upgrade if needed. Clone at ../PM2S (not vendored).

**Next**: integrate — feed Kong-detected notes → PM2S → per-note beat position +
note value → build GP5 merging PM2S rhythm with our string/fret; handle t=0 edge
case; validate on GuitarSet beat labels. GAPS (non-commercial) and madmom
(banned) avoided.

## Rhythm v2 — audio beats + MusicXML (2026-07-01, supersedes PM2S-only)
PM2S note-based beat tracking COLLAPSES on our noisy detection (crammed 95 notes
into ~4 bars). Root cause fixed by taking beats from AUDIO, not notes:
`librosa.beat.beat_track(vidtest1)` → clean 96bpm quarter grid, 28 beats, full
span, immune to note errors. Then snap detected onsets to a **32nd grid** (gives
73 distinct attack positions = exactly the reference's 73; 16th grid only gives
69→59 after export). 

**GP5 is the limiter**: `export_gp5` chord-merges attacks 73→59 regardless of
`quantize_division`. Switched output to **MusicXML via music21** (auto note
values, rests, beams; no merging) → preserves all 73 attacks, note values mostly
8ths/16ths matching the reference. Rhythm metric `compare_rhythm` (repo) on the
GP5 variants: posF1 0.894, note_value_accuracy 0.93 (32nd + to-next-onset best);
attack recovery needed MusicXML.

Research (how accurate systems do it): beat-based rhythm quantization + note-VALUE
modeling (Cambouropoulos'00 IOI, Takeda'02 HMM+Viterbi, PM2S CRNN, T5 transformer
onsetF1 0.97/nvAcc 0.83); MuseScore MIDI-import = adaptive grid + tuplet detection
+ rests + voice separation. TODO to finish: bias grid to 8th/16th (clean 32nd
jitter), tuplet detection, voice separation (melody vs open-G pedal), then wire
audio-beat+MusicXML as the pipeline rhythm stage. music21 (BSD) + librosa (ISC) —
license-clean. GAPS/madmom still avoided.

## What's been built (by area)

### Eval harness (the backbone)
- `aitabs/eval/` scores predictions vs reference. `gp5_import.load_gp5_notes`
  now honours **mid-song tempo changes** (`mixTableChange.tempo`) and
  **capo/alternate tuning** (concert pitch). Big early win: test1 0.18 → 0.78.
- **GuitarSet integration** (real acoustic, MIT): `guitarset_import.py`,
  `discover_guitarset_pairs`, `scripts/prepare_guitarset.py`,
  `eval_dataset.py --guitarset`.
- **Diagnostics:**
  - `scripts/diagnose_synthetic_gap.py` + `render_gp5.py` (Track 0: synth-vs-real
    detection ceiling).
  - `error_analysis.py`: `categorize_false_positives` (octave/5th/dup/other/isolated)
    and `categorize_false_negatives` (chord_inner/sustain/octave/bass/isolated).
    Dump notes with `eval_dataset.py --dump-notes DIR`, analyze with
    `scripts/analyze_errors.py DIR`.

### Note accuracy (over/under-detection)
- **Over-detection (test5/test7):** an opt-in octave/harmonic filter
  (`suppress_octave_harmonics`, `PipelineConfig.suppress_harmonics`) was A/B'd and
  **rejected — net negative** (removed real octaves). Left default-OFF. FP
  characterization showed octave ghosts are NOT the dominant FP.
- **Under-detection / test1 ceiling:** diagnosed as **dense let-ring polyphony**
  (43% of notes overlap a later onset), not repeats/harmonics → a recall problem,
  detection-side. Use the miss categorizer on dumped notes to confirm per clip.

### Rhythm notation (RESEARCH_RHYTHM.md — Steps 1–3 done)
Root problem was notation built on a flat frame (constant tempo + hardcoded 4/4).
- **Step 1:** `TempoMap` (seconds↔beats), reads time signature + tempo map from
  reference GP5; notation quantizes in **beat space**; `time_signature`/`tempo_map`
  plumbed through `export_gp5`; `run.py:_resolve_meter`. Fixed test1: 4/4→3/4,
  tie-fraction 0.230→0.003.
- **Step 2:** license-clean (numpy) **`estimate_meter`** (accent-autocorrelation,
  picks 2/3/4 + downbeat phase) and `TempoMap.from_beat_times`, used when no
  reference. **NOT madmom** (CC-BY-NC).
- **Step 3:** downbeat anchoring (`lead_in_beats`); **coarsest-grid-that-fits
  quantizer** (`quantize_onset_ticks` prefers 8th, drops to 16th only when needed)
  → absorbs onset jitter + collapses ghost notes → kills false syncopation while
  preserving real 16ths/triplets. Full metrical-HMM note-value model deferred
  (needs training data).

### Live mode (LIVE_MODE.md, RESEARCH_LIVE.md)
`scripts/live_monitor.py`: mic → BasicPitch posteriorgram → per-frame
pitch+confidence. Clean in-place terminal UI (note names + confidence bars),
`--plot` scrolling posteriorgram heatmap, **interactive keys** (`[`/`]` confidence,
`;`/`'` sustain, `h` hysteresis, `c` head, `,`/`.` polyphony, space/q). Pure core
in `aitabs/pipeline/audio/live.py` (RingBuffer, frame_peaks, HysteresisTracker,
PosteriorgramHistory, ControlState) — unit-tested; needs PortAudio + sounddevice.

### Owned eval recordings + model roadmap
- `RECORDING_GUIDE.md` defines the immediate owned held-out eval set: five
  fully tabbed ~30s clips covering monophonic, let-ring fingerpicking, strummed
  chords, distorted/electric timbre, and capo/alternate voicing. These are a gold
  validation set and must **never** be training data.
- `scripts/check_recordings.py` is the intake/preflight tool for new clips. It
  checks GP5/audio pairing, tempo/meter/tuning/capo, note count, duration, mean
  polyphony, let-ring overlap, fastest IOI, and clip diversity.
- `RESEARCH_MODELS.md` records the strategic pivot: upstream pitch detection is
  the bottleneck, so the next real jump is a pluggable detector A/B path, then
  Tier 2 fine-tuning on commercial-safe data, then a Tier 3 data/model moat.
- CC YouTube is not a labeled-data solution. It can be considered later only as
  unlabeled robustness/domain-diversity material after ownership, attribution,
  and redistribution constraints are reviewed.

## How to run
```bash
# Eval (song clips) with rhythm scoring
python scripts/eval_dataset.py data/eval --config data/eval/test1_best/best_combined.json --no-demucs --rhythm
# Real acoustic (GuitarSet)
python scripts/prepare_guitarset.py --download --limit 20
python scripts/eval_dataset.py data/eval/guitarset --guitarset
# GuitarSet with Kong detector (zero-shot)
python scripts/eval_dataset.py data/eval/guitarset --guitarset --detector kong
# GuitarSet with Kong (calibrated thresholds from calibration_kong.json)
python scripts/eval_dataset.py data/eval/guitarset --guitarset --detector kong --onset-threshold 0.2 --frame-threshold 0.1 --confidence-threshold 0.1
# Calibrate Kong thresholds
python scripts/calibrate_detector.py --detector kong --max-clips 30 --out data/eval_results/calibration_kong.json
# Fine-tune Kong on GuitarSet overnight
python scripts/finetune_kong.py --steps 10000 --save-every 1000
# Eval fine-tuned Kong
python scripts/eval_dataset.py data/eval/guitarset --guitarset --detector kong --model-path data/models/kong_guitarset_ft/kong_ft_final.pth
# Error breakdown
python scripts/eval_dataset.py data/eval --config ... --no-demucs --dump-notes data/eval_results/notes
python scripts/analyze_errors.py data/eval_results/notes
# New owned clips
python scripts/check_recordings.py data/eval
# Live
python scripts/live_monitor.py --plot
# Tests (needs full env)
pytest
```

## Open threads / next steps (priority order)

Current priority override (2026-06-29): **7-day Kong fine-tuning sprint**.
1. **Kong calibration:** run `python scripts/calibrate_detector.py --detector kong --max-clips 30 --out data/eval_results/calibration_kong.json` (or check in-progress job). Apply best thresholds to eval.
2. **Kong fine-tuning (overnight):** `python scripts/finetune_kong.py --steps 10000 --save-every 1000`. ~15h on CPU. Output: `data/models/kong_guitarset_ft/kong_ft_step*.pth`.
3. **Post-fine-tune eval (Day 5):** eval + calibrate fine-tuned Kong. Target: ~87-91% F1.
4. **vidtest1 with Demucs (Day 6):** download Demucs weights with `! python -m demucs.utils` on good connection, then eval with `--detector kong`.
5. **Owned held-out eval set:** record/tab 5 clips per RECORDING_GUIDE.md. Deferred but important.
6. **Post-rough Phase 2:** clean Kong pre-train using MIDI+FluidR3 to clear MAESTRO licensing risk.
7. **Rhythm/rubato:** keep validating, but downstream from pitch accuracy work.

Older diagnosis notes below remain useful as background:

1. **Polyphonic recall** (the test1 ceiling and GuitarSet recall): confirm with the
   miss categorizer on dumped notes; try recall-first thresholds
   (lower `confidence_threshold`/`onset_threshold`); consider a guitar-specialized
   model. This is the highest-value accuracy lever.
2. **Over-detection (test5/test7):** the simple octave filter failed; the real FP
   type is still unconfirmed — run `analyze_errors.py` on dumped notes to decide
   (sustain re-trigger vs noise vs polyphony).
3. **Rhythm validation on real audio:** Steps 1–3 are verified on synthetic/round-trip;
   exercise on real GuitarSet/song audio and watch `note_value_acc` / `pos F1`.
   Consider exposing `coarse_margin` as a config knob.
4. **test3 rubato (+580 ms):** notated GP5 time ≠ expressive performance time;
   needs tempo-robust/DTW matching in the metric (research in RESEARCH_RHYTHM Step 1
   note + earlier discussion).
5. **6/8 / compound meters:** `estimate_meter` only covers /4 meters today.

## Licensing (startup constraint — see audit in chat)
- **Code:** clean (no vendored/copied code). Deps are permissive (Apache/BSD/MIT/ISC).
  Mind: **PyGuitarPro = LGPL** (fine as unmodified import); **FFmpeg-bundled wheels**
  (opencv/torchaudio) if redistributing binaries.
- **madmom intentionally avoided** (models CC-BY-NC).
- **Training data policy:** GuitarSet is preferred (MIT). Verify GOAT's CC-BY
  attribution before use. Do not use GAPS/non-commercial datasets for product
  training. Treat CC YouTube as unlabeled robustness/domain-diversity material
  only, not labeled ground truth.
- **⚠️ TODO before going public/commercial:** `data/eval/reference/*.gp5` are
  **copyrighted third-party song transcriptions** (one embeds "© Donny Dwiputra
  2017"). They're committed AND in git history. Remove + gitignore + scrub history
  (git filter-repo/BFG), and replace eval data with GuitarSet (MIT) / your own
  originals. User chose to defer (private repo). No LICENSE file yet.

## Audio ensemble — Kong ⊕ BasicPitch (2026-07-02)
Built role-specialized fusion of two detectors with complementary front-ends
(Kong = log-mel, crisp onsets; BasicPitch = HCQT, harmonic-aware polyphony).
New: `aitabs/pipeline/audio/ensemble.py` (`reconcile_notes` + `EnsembleDetector`,
registered as detector `"ensemble"`), `tests/test_ensemble.py` (9 tests),
`scripts/eval_ensemble.py` (resumable A/B; Kong from npz cache, BasicPitch cached
to `data/eval_results/cache_bp_gs/`).

**Held-out steel (GuitarSet player-00, 16 clips, kong2 step-4000, onset tol 50 ms):**
| method | P | R | F1 |
|---|---|---|---|
| Kong only | 0.934 | 0.887 | 0.909 |
| BasicPitch only | 0.587 | **0.928** | 0.715 |
| Union | 0.579 | **0.943** | 0.714 |
| **Intersection (Kong onset 0.3)** | **0.962** | 0.874 | **0.915** |
| Gated (≥0.6) + harm-suppress BP | 0.907 | 0.893 | 0.899 |

Findings: (1) **BasicPitch genuinely out-recalls Kong (0.928 vs 0.887)** — its
HCQT front-end catches polyphonic notes Kong structurally misses; union ceiling
R=0.943. (2) BP's false positives are largely octave/harmonic ghosts →
`suppress_octave_harmonics` on BP recovers ~11 pts of union precision
(0.579→0.688) at ~0 recall cost. (3) **Best F1 = intersection = 0.915** (P 0.962,
kills ghost notes / the "too many notes" complaint) — beats Kong-alone 0.909; the
agreed-note recall plateaus ~0.875 regardless of Kong threshold. (4) `"gated"` is
a P/R dial: recall up to ~0.94 at precision cost. **Does not reach 95/95** —
consistent with published GuitarSet SOTA ~0.88 F1; 0.95/0.95 is SOTA-beyond.
`EnsembleDetector` defaults to `mode="intersection"`. **Open decision: default
mode (intersection = precision/F1 vs gated = recall dial).** Next levers for
recall: PCEN/HCQT trainable front-end + augmentation retrain (see research memos).

## Tier-1 preprocessing front-end + degraded bench (2026-07-03)
New: `aitabs/pipeline/audio/preprocess.py` (`preprocess_audio`: DC removal →
78 Hz high-pass → loudness normalize [pyloudnorm LUFS if installed, else RMS]),
`aitabs/eval/degrade.py` (synthetic reverb/noise/rumble/level for a two-sided
bench), `tests/test_preprocess.py` (8 tests), `scripts/eval_preprocess.py`
(resumable, budgeted Kong A/B on {clean, clean+prep, degraded, degraded+prep}).
`audio_to_tab.py` gained `--preprocess`. Default ensemble mode set to **gated**
(recall-leaning, `min_singleton_conf=0.5`) per product decision.

**Two-sided A/B (GuitarSet player-00, 6 clips, kong2 step-4000, onset 0.3):**
| variant | P | R | F1 |
|---|---|---|---|
| clean | 0.928 | 0.908 | 0.917 |
| clean+prep | 0.899 | 0.926 | 0.910 |
| degraded | 0.983 | 0.698 | 0.814 |
| degraded+prep | 0.960 | **0.861** | **0.907** |

**Recovery +0.093 F1** on degraded audio (recall 0.698→0.861, +16 pts — reverb/
noise/quiet-lost recall largely recovered). **Regression −0.007 F1** on clean
(near-neutral; recall actually *rises* 0.908→0.926 from normalization lifting
quiet notes). Confirms Tier-1 preprocessing is a real, license-clean recall lever
for real-world audio, safe on clean audio. This + the ensemble are two
independent measured recall levers. TabCNN string-expert deferred by user.

**Update (LUFS + dereverb, 7 clips):** installed `pyloudnorm` (MIT, in
requirements) → `normalize_loudness` now uses ITU-R BS.1770 LUFS (RMS is
fallback). Re-run: **Tier-1 recovery rose to +0.147 F1** (degraded recall
0.628→0.842) — LUFS beats the RMS fallback, worth it. **Dereverb (Tier-3,
`dereverb_spectral`, spectral-subtraction) tested and did NOT win: marginal
−0.004 F1** (slightly cut recall 0.842→0.833). Kept **off by default** as a
recorded experiment (like harmonic suppression) — the synthetic-reverb bench may
understate its value on *real* reverb, but unproven, so not enabled. Net
preprocessing default = DC + 78 Hz HPF + LUFS normalize.

## PCEN + augmentation retrain — scaffolded, launch-ready (2026-07-03)
Next recall tier (needs Colab GPU). New: `aitabs/pipeline/audio/pcen.py`
(`TrainablePCEN` learnable per-band AGC/compression + `MelPCEN` +
`attach_pcen_frontend` — swaps Kong's `logmel_extractor` for PCEN with zero
changes to Kong's forward), `tests/test_pcen.py` (5 tests incl. full Kong
attach+backprop), `RESEARCH_PCEN.md` (the recipe). `colab_finetune_kong.py`
gained `--pcen / --pcen-warmup / --pcen-fe-lr / --pcen-crnn-lr`: attaches PCEN,
param-groups (PCEN+bn0 fast, CRNN slow), freezes CRNN for warmup steps to absorb
the log-mel→PCEN distribution shift, then unfreezes. Augmentation = reuse
`aitabs/eval/degrade.py` (reverb/noise/gain, label-preserving) + ±10-cent micro
pitch-shift. **Known gotcha (documented):** PCEN checkpoints need PCEN attached
*before* load — won't load into stock `KongDetector`; first follow-up is a
`--pcen` flag on the eval path. Honest target ~0.92–0.93 F1 (0.95 is above
GuitarSet SOTA ~0.88). Not launched (user's GPU); scaffold + tests green.

## BasicPitch threshold calibration → ensemble win (2026-07-03)
`RESEARCH_BASICPITCH.md` written (diagnoses why the prior BP fine-tune stalled at
0.808: contour head never trained, hard single-frame onset labels, no held-out
selection, no window overlap; official BP training shipped in v0.4.0). **Lever A
done (no training):** stock BasicPitch was badly untuned in the ensemble
(0.5/0.3/0.3 → P 0.587). Calibrated on GuitarSet held-out →
**onset 0.7/frame 0.3/conf 0.5 (BP-alone F1 0.898)**. Re-ran ensemble A/B: recall
modes transformed — **Union F1 0.714→0.900, gated-0.5 default F1 0.875→0.903**
(P 0.84→0.90 @ R 0.90). Baked the calibrated point into `EnsembleDetector`
defaults + `eval_ensemble.py`. Also added `--pcen` support across `KongDetector` /
`get_detector` / `EnsembleDetector` / `eval_preprocess.py` / `audio_to_tab.py`
(PCEN checkpoints attach-before-load; smoke-tested). Next (Lever B, needs GPU):
fine-tune BP on GuitarSet+Guitar-TECHS via official v0.4.0, judged by ensemble F1;
BP-only-on-clean-data would be a fully commercial-clean detector (unlike Kong).

## Parallel GPU: BasicPitch fine-tune (Kaggle) + Kong PCEN (Colab) (2026-07-03)
New: `scripts/kaggle_finetune_basicpitch.py` — **self-contained** BasicPitch
fine-tune (inlines GuitarSet JAMS + Guitar-TECHS MIDI loaders; no repo import) for
a fresh Kaggle notebook. Applies the `RESEARCH_BASICPITCH.md` fixes: Gaussian-
blurred onset targets (`--onset-blur`), held-out val split + keep-best-epoch,
overlapping windows (`--stride-frac`), Guitar-TECHS domain-balanced 50/50. Data
pipeline smoke-tested locally (180 clips, shapes + blur verified); TF training
runs on GPU only. `KAGGLE_GPU.md` = step-by-step to run **both** retrains at once
on **separate quotas** (Colab Pro = Kong PCEN, Kaggle = BasicPitch, via File ▸
Link to Colab). `eval_ensemble.py` gained `--bp-model-path` (+ distinct cache key)
so a fine-tuned BasicPitch can be A/B'd inside the ensemble — keep it iff it lifts
**ensemble** F1. Fine-tuned-BP is commercial-clean (Apache-2.0 + MIT + CC-BY).

## PCEN retrain launched + diagnosis in progress (2026-07-03)
PCEN+augmentation retrain ran on Colab (L4) resuming from kong2_ft_step4000.
Fixed a self-containment bug first (the `--pcen` edit had added `from aitabs...`
which crashes on Colab with no repo — inlined `TrainablePCEN/MelPCEN/
attach_pcen_frontend` into `colab_finetune_kong.py`). Monitored via Google Drive
MCP: read `kong_pcen_gpu/training_log.jsonl` — **training healthy**: warmup loss
0.248→0.160 (steps 1–1500, CRNN frozen), then the designed **0.160→0.106 drop at
step 1500–1600** (CRNN unfrozen), settling ~0.06 (≈/below the non-PCEN ~0.07). No
collapse. Checkpoints step2000/4000/6000 downloaded to `eval/pkong_ft_step*.pth`
(172,031,117 B — the +3.7 KB vs non-PCEN = melW + alpha/delta/r; verified PCEN
params trained). Added `--pcen` to `calibrate_detector.py` (attach-before-load).
Diagnosis running: `calibrate_detector --detector kong --pcen` for clean held-out
steel (regression check vs kong2_ft_step4000's 0.9108) + `eval_preprocess --pcen`
for degraded recovery. Colab run still going (7700+/30000).

## Ensemble × preprocessing DO stack — "augment" mode (2026-07-04)
`scripts/eval_ensemble_preprocess.py`: runs Kong-alone vs ensemble across
{clean, degraded, degraded+prep} on the same clips (6, cached). Found the old
**gated** default was a bug — it gated BOTH detectors, dropping real Kong notes
BasicPitch didn't corroborate (degraded+prep: Kong 0.914 → gated-ensemble 0.900).
Fix: new **`augment`** mode = keep ALL Kong notes (reliable timing-authority),
add only BasicPitch notes ≥ min_conf. Results (F1):
| variant | Kong | augment0.7 | union |
|---|---|---|---|
| clean | 0.9168 | **0.9207** | 0.9109 |
| degraded | 0.8144 | 0.8222 | **0.8750** |
| degraded+prep | 0.9144 | 0.9144 | **0.9178** |

**The levers stack:** best clean = ensemble(augment)=0.921 (beats Kong 0.917);
best real-world = ensemble(union)+preprocess=0.918 (beats Kong+prep 0.914). Both
beat any single lever. `augment` never regresses Kong, so it's the new
`EnsembleDetector` default (min_conf 0.7); `union` for max degraded recall.
Updated ensemble.py (+augment mode), audio_to_tab.py default, test_ensemble.py.
(Aside: BasicPitch fine-tune failed — FT-BP 0.753 vs stock 0.898, onset head
collapsed from blurred targets; keep stock calibrated BP. PCEN parked: works but
~matched by free preprocessing.)

**Extensive sweep validation (16-clip clean + 6-clip degraded):** confirmed
ensemble > Kong in ALL conditions (clean 0.918 vs 0.911; degraded 0.875 vs 0.814)
— robust, not noise. `gated` confirmed bad (gating Kong collapses it, 0.72 clean).
The one real knob is BP inclusiveness, and the optimum is **adaptive**: clean →
selective BP (≥0.7, augment) = 0.921; messy → inclusive BP (union) = 0.918+prep.
No static config wins both. Wired the adaptive rule into `audio_to_tab.py`:
`--ensemble-mode auto` (default) = union when `--preprocess` else augment.
Audio-lever tuning is now near its ceiling (~0.92); next real gain = vision layer.

## Vision Stage-1 implemented (2026-07-04)
Started the vision track (the real string/fret unlock; audio caps ~0.65 there).
Stage 1 = geometric, no training, license-clean (OpenCV ArUco + MediaPipe, both
Apache — no AGPL YOLO, no MANO). New:
- `aitabs/pipeline/vision/fretboard_geometry.py` — canonical fretboard grid
  (12-TET fret spacing), homography apply/invert (pure numpy), `nearest_cell`,
  `image_point_to_cell`. Fully unit-tested (`tests/test_fretboard_geometry.py`,
  incl. synthetic-homography round-trip recovering every cell).
- `aitabs/pipeline/fusion/fusion.py` — implemented `fuse()`: audio owns pitch+
  timing, vision picks the (string,fret) a finger was on near the onset among the
  pitch's playable positions; lowest-fret fallback else. `source_distribution()`
  = the vision-contribution metric (dev-rule gate). Tested (`tests/test_fusion.py`).
- `scripts/vision_to_tab.py` — runnable video entry point (ArUco->homography->
  MediaPipe fingertips->cells->fuse). I/O layer; needs a marked clip + `--markers`
  JSON (aruco_id -> 4 canonical corners) to run; not unit-tested (needs video).
12 vision tests green. Next: record one marked clip, verify grid lands on the real
strings, measure tab F1 + source distribution vs audio-only mapper (M1/M2 in
RESEARCH_VISION.md §6). Stage 2 (cross-attention learned fusion) gated on data.

## Rhythm — beat/downbeat validation on GuitarSet ground truth (2026-07-08)
First honest test of the **metrical grid** (the foundation under all rhythm
notation) against **real performed beats**, not the synthetic constant-tempo GP5
grid. New: `load_guitarset_beats` (parses JAMS `beat_position` → beats, downbeats
via `position==1`, meter, tempo) + `score_beats` (mir_eval beat/downbeat
F-measure, octave-folded tempo error, meter-correct) in
`aitabs/eval/{guitarset_import,rhythm_metrics}.py`; CLI
`scripts/eval_rhythm_guitarset.py`; 6 tests in `tests/test_rhythm_beats.py`.
Results on 20 player-00 solo clips (`data/eval_results/rhythm_beats_raw.json`):
- **tempo (BPM) is solid**: mean octave-folded err **2.64%** (1 bad clip, 147→99).
- **beat tracking is moderate**: mean beat_F **0.623** (median 0.747, 9/20 > 0.8).
- **downbeat tracking was BROKEN → FIXED**: was mean downbeat_F **0.125**
  (median 0.000, 1/20 > 0.5). Root cause: `estimate_meter` picked bar-phase from
  onset accents, but solo/fingerstyle guitar carries no energy accent on beat 1,
  so phase was noise (conf 0.11–0.25). Fix: **origin anchor** — assume the excerpt
  starts on a downbeat and extrapolate the beat grid to t≈0
  (`phase = round(beat0/period) % numerator`). Validated: origin phase correct
  19/20 vs accent ~0/20; GuitarSet excerpts start on a downbeat 20/20. Gated by
  accent confidence (`_ACCENT_PHASE_MIN_CONF=0.5`): a *confident* accent (real
  pickup) still wins, else fall back to origin. **Result: downbeat_F
  0.125 → 0.511** (median 0.667, 10/20 > 0.5). beat_F/tempo unchanged (phase-only).
  In `aitabs/pipeline/audio/tempo.py::estimate_meter` (+`_phase_from_origin`);
  tests in `tests/test_meter_estimation.py`.
- **`refine_beats_to_onsets` HURTS on GuitarSet**: beat_F 0.623→**0.522**,
  downbeat 0.125→0.022. Snapping beats to performed onsets pulls them off the
  true grid on solo material (onsets ≠ beats under syncopation/rests). It was
  built for the "live recording reads badly" case; on metronomic GuitarSet it's a
  regression — do NOT enable `cfg.refine_beats_to_onsets` by default.
**Next rhythm lever = beat tracking itself.** downbeat_F is now capped by beat_F
(0.623): the 4 clips still at downbeat_F 0 are all beat-tracking failures
(beat_F 0.0–0.52), not phase. librosa is weak on sparse solo guitar. Improving the
beat tracker (or onset-informed beat correction that doesn't over-snap) is the next
gain. Validate any change with `eval_rhythm_guitarset.py`.

## Ensemble+preprocess on song clips — rhythm eval (2026-07-08)
Ran the best combined **BP+Kong ensemble + preprocessing** through the notated-
rhythm eval (`compare_rhythm` vs GP5) on the 12 song clips (test1,test3–12,
vidtest1), 4-variant matrix (driver: `$job/tmp/rhythm_ensemble_exp.py`; summary
`data/eval_results/rhythm_songclips_ensemble_summary.txt`). Ensemble =
`eval/kong2_ft_step4000.pth` + BasicPitch (augment), Kong thresholds
onset=0.4/frame=0.2/conf=0.2, preprocess = DC+HPF78+LUFS, base best_combined.json.

| variant | pitchF1 | posF1 | nvAcc | restF1 |
|---|---|---|---|---|
| BP baseline (no pp) | **0.643** | **0.839** | **0.806** | 0.188 |
| Ensemble+pp | 0.632 | 0.836 | 0.796 | 0.179 |
| Ensemble no-pp | 0.630 | 0.821 | 0.775 | 0.093 |
| Ensemble+pp refineOFF | 0.632 | 0.836 | 0.796 | 0.179 |

**Synthetic vs real is the whole story.** test1–12 are **MIDI renders** (clean
synthetic) — there the ensemble+Kong is out-of-distribution (Kong trained on real
GuitarSet) and slightly *dilutes* tuned BasicPitch; rhythm pos/nv are driven by the
quantizer + reference tempo/meter, so the detector barely moves them. On
**vidtest1 (the one REAL recording)** the ensemble+preprocess **wins clearly**:
pitch 0.828→**0.870**, pos 0.712→**0.761**, nvAcc 0.638→**0.667**; and preprocessing
is *critical* there (no-pp pos collapses to 0.661). Takeaways:
1. **Preprocessing helps the ensemble** (+0.015 pos, +0.086 restF1 vs no-pp) but on
   synthetic clips only recovers to ~BP-baseline; on real audio it's a big win.
2. **`refine_beats_to_onsets` is a no-op here** (refineOFF ≡ Ensemble+pp exactly):
   these clips carry GP5 references, so `use_reference_bpm/meter/tempo_map` override
   the audio beat grid — the GuitarSet "refine hurts" finding only bites when
   estimating meter from audio (no reference).
3. **restF1 is poor everywhere (~0.09–0.19)** — rest notation is the consistent weak
   spot across all configs (same gap as the failing `rest_f1` test).
**Verdict:** keep tuned BasicPitch for the synthetic song-clip eval; ensemble+
preprocess is the right default for **real recordings** (validated on vidtest1) — i.e.
the existing `--ensemble-mode auto` (union+preprocess for real audio) is the right call.

## Rest notation — diagnosis (2026-07-08)
Chased the low restF1 (~0.09–0.19 across all configs). Findings:
1. **The premise is inverted — rests aren't *missing*, they're *over-produced*.**
   vidtest1 (0 ref rests) → 9 pred rests; test1 (3 ref) → 14 pred. The spurious
   rests are short (240 ticks = 1/16 note) and land mid-phrase where the reference
   notates connected/legato notes.
2. **Mechanism:** `build_note_value_schedule` emits a rest when
   `sounding_q + rest_thresh <= ioi` with `rest_thresh = STRAIGHT_BASE (240)`.
   Median dur/IOI on these clips is **high (1.3–2.0)** — most notes sustain past the
   next onset (pedaled/legato) — so `offsets_reliable` correctly trusts offsets, but
   the ~2–11% of notes with an anomalously *short* detected release each spawn a
   1/16 rest. It's per-note detector-offset jitter, not a global mode error.
3. **restF1 is a poor metric on this set.** References carry **0–11 rests** vs
   hundreds of notes (test10/11/5/vidtest1 = 0), and some are leading pickups
   (negative ticks). With a 1/32-note position tolerance, one extra/missing rest
   swings F1 0↔1. The ~0.1 aggregate is sparse-target variance + editorial notation,
   not a reliable quality bar.
4. **Levers tried (not shipped):** `legato=True` removes ALL rests → vidtest1
   restF1 0→1.0 but blunt (loses genuine rests on test8/test3). Raising the min-rest
   gap to an 1/8 in `auto_rhythm` cut spurious rests (vidtest1 6→1) **but didn't move
   restF1** (still 1 rest vs 0 ref) and cost ~0.05 nvAcc on vidtest1 — reverted
   (marginal, net-negative). Genuine rest *placement* needs reliable offset
   detection, a large effort with small payoff.
**Verdict:** stop treating restF1 as a target on synthetic clips. If pursued, the
right move is per-note legato-clamp for real recordings (readability), scored by
spurious-rest count, not this restF1. Note quantiser + reference tempo/meter, not
rests, drive the notation numbers that matter here.

## Working branch
`claude/cool-cori-ywtx20` (merged to `main` via PRs as work landed).
