# Training BasicPitch to maximize pitch on GuitarSet + Guitar-TECHS

Goal: raise BasicPitch's pitch accuracy on guitar — especially **precision**,
which is its weak axis (in the ensemble it ran **P=0.587 / R=0.928**; it out-
recalls Kong but drowns in false positives). A tighter BasicPitch lifts the whole
Kong⊕BasicPitch ensemble. Two levers: **(A) threshold calibration** (cheap, no
training) and **(B) proper fine-tuning** on GuitarSet + Guitar-TECHS.

## What already exists in the repo
- `aitabs/pipeline/audio/finetune.py` — custom TF GradientTape loop over the
  ICASSP-2022 SavedModel.
- `scripts/finetune_basicpitch.py` + `scripts/build_training_data.py` (manifest).
- `FinetunedDetector` + `calibrate_detector.py --detector finetuned` (eval path).
- **Prior result:** GuitarSet-only fine-tune = **0.808 F1**, did **not** beat the
  0.810 baseline.

## Why the prior fine-tune didn't beat baseline (from reading `finetune.py`)
Concrete, fixable causes — not "fine-tuning doesn't work":
1. **The contour head is never trained.** Loss = `bce(note) + 5·bce(onset)` only.
   BasicPitch is a *3-head* model (contour → note → onset); the note head consumes
   the contour posteriorgram, and note-creation at inference uses contour. Leaving
   contour unsupervised caps the gain and can desync the heads.
2. **Hard single-frame onset labels, no blur.** `_window_labels` sets `onset_gt`
   to 1.0 at exactly one frame. BasicPitch's original targets are **Gaussian-
   blurred** in time (and the note targets have soft edges). A one-hot onset at
   ~86 fps is a brutal target → the onset head under-trains → precision suffers.
3. **No held-out model selection.** Trains fixed epochs, checkpoints each epoch,
   keeps the *last* — no early stopping on a validation split → overfits 150 clips.
4. **Full-window stride (no overlap).** `_iter_windows` steps a whole window, so
   each frame is seen once per epoch → little data, no onset-phase augmentation.
5. **Tiny data.** GuitarSet solo ≈ 150 clips; without the contour regularizer and
   with most layers unfrozen, it overfits. Guitar-TECHS is the fix.

## Lever A — threshold calibration first ✅ DONE (2026-07-03), big win, no training
The ensemble ran BasicPitch at an **untuned** point (onset 0.5/frame 0.3/conf 0.3
→ P 0.587). Calibrated stock BP on GuitarSet held-out (16 clips):
**best = onset 0.7 / frame 0.3 / conf 0.5 (BP-alone F1 0.898)**. Re-ran the
ensemble A/B at the calibrated point — the recall modes were transformed:

| mode | untuned BP (0.5/0.3/0.3) | **calibrated BP (0.7/0.3/0.5)** |
|---|---|---|
| BasicPitch only | .587 / .928 / **.715** | .917 / .886 / **.900** |
| Union | .579 / .943 / **.714** | .883 / .921 / **.900** |
| Gated (≥0.5, default) | .838 / .919 / **.875** | .903 / .904 / **.903** |
| Gated (≥0.7) | .937 / .890 / **.912** | .969 / .866 / **.914** |

The calibrated BP point is now the `EnsembleDetector` default
(`_DEFAULT_BP_ONSET=0.7/FRAME=0.3/CONF=0.5`) and the `eval_ensemble.py` default.
The gated-0.5 default rose **F1 0.875→0.903** (precision 0.84→0.90 at recall
0.90). Fastest lever, zero training. **Follow-up:** calibrate on Guitar-TECHS too
(add a `load_guitar_techs_clips` path to `calibrate_detector`) to confirm the
electric operating point; pick the BP point by **ensemble** F1.

## Lever B — fine-tune properly

**B1 (recommended): use BasicPitch's official training code.** Spotify shipped a
full training pipeline (data preprocessing + loop, correct 3-head targets incl.
contour and blurred onset/note labels) in **v0.4.0** — which `requirements.txt`
already pins. Prefer it over the custom loop: it fixes causes #1–#2 for free.
Point it at a guitar dataset built from GuitarSet + Guitar-TECHS, fine-tune from
the ICASSP-2022 weights, low LR.

**B2 (if staying on the custom loop): fix the five causes.**
- add a **contour-head loss** (build contour targets like the note targets);
- **Gaussian-blur** the onset (σ≈1–2 frames) and note target edges;
- add a **validation split + keep best epoch** (early stopping);
- **overlap windows** (stride = ½ or ¼ window);
- **freeze more** of the front-end (or lower LR) to fight overfitting on small data.

**Data — add Guitar-TECHS (both levers):** extend `build_training_data.py` to
append Guitar-TECHS clips via `load_guitar_techs_clips(root)` (returns
`{clip_id, audio_path, notes}` with the standard note dicts) into the same
manifest, and **domain-balance** GuitarSet vs Guitar-TECHS per batch (mirror the
50/50 sampling proven in `colab_finetune_kong.py`) so electric doesn't dominate.

**Alternative runtime:** `basic-pitch-torch` (gudgud96) is a PyTorch port — handy
if you want BasicPitch training in the same torch stack as Kong/PCEN, but the
official TF v0.4.0 pipeline is the authoritative target construction.

## Validation — measure the right thing
Fine-tuned BP must be judged on **two** metrics, not one:
1. **Standalone** P/R/F1 on held-out GuitarSet (calibrated) — did guitar fine-
   tuning tighten precision without losing the recall that makes it useful?
2. **Ensemble** F1 — the real goal. Re-run `eval_ensemble.py` swapping stock BP
   for the fine-tuned model; keep it only if **ensemble F1 rises**. BP's value is
   catching Kong's polyphonic misses at usable precision, so a fine-tune that
   trades recall for precision may or may not help the ensemble — measure it.

## Licensing
GuitarSet (MIT) + Guitar-TECHS (CC BY 4.0) — both self-recorded, commercial-clean.
BasicPitch is Apache-2.0 (code + ICASSP-2022 weights) — clean, unlike Kong's
MAESTRO-pretrained weights. So a **BasicPitch fine-tuned only on GuitarSet +
Guitar-TECHS is fully commercial-clean** — a strategic plus over the Kong path.

## Recommended sequence
1. **Calibrate stock BP on GuitarSet+Guitar-TECHS**, re-run ensemble A/B — fastest
   possible gain, no training. (Lever A.)
2. **Fine-tune via official v0.4.0** on GuitarSet+Guitar-TECHS (domain-balanced),
   held-out model selection. (Lever B1.)
3. Calibrate the fine-tuned model; re-run the ensemble A/B; keep it iff ensemble
   F1 rises. Bonus: it's a **license-clean** detector, unlike Kong.
