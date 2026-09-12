# PCEN + augmentation retrain — launch-ready recipe

Goal: push **recall** past the current ~0.89 ceiling by giving Kong a **trainable
front-end** (PCEN) plus **realistic augmentation**, retrained on GuitarSet +
Guitar-TECHS. Thresholds are exhausted (recall is flat vs threshold); the
ensemble and Tier-1 preprocessing are the no-train levers already landed. This is
the next tier and needs a GPU (Colab).

What each piece targets:
- **PCEN** (`aitabs/pipeline/audio/pcen.py`) — learnable per-channel AGC + dynamic
  compression replacing static log-mel. Hits the **quiet/short** recall bucket
  and adds loudness/reverb robustness.
- **Augmentation** (reverb + micro pitch-shift + light noise/EQ) — makes the
  model robust to real-world conditions. Evidence (Cheuk/Benetos 2024, on *Kong's
  model*): reverb + pitch-shift are the big wins; augmentation mostly lifts
  **precision/robustness**, so pair it with PCEN for recall.

Honest expectation: realistic **~0.92–0.93 F1**; **0.95/0.95 is above published
GuitarSet SOTA (~0.88)** — treat it as a stretch, not a plan.

---

## 1. Architecture change (zero surgery on Kong's forward)

Kong's `note_model.forward` does `x = self.logmel_extractor(power)` then runs the
CRNN on `x`. We swap `logmel_extractor` for a PCEN front-end — the forward is
untouched:

```python
from aitabs.pipeline.audio.pcen import attach_pcen_frontend

# after building + loading the resume checkpoint, before training:
pcen = attach_pcen_frontend(model.note_model, s=0.025)   # returns TrainablePCEN
model = model.to(device).train()
```

`MelPCEN` reuses Kong's frozen mel matrix (`melW`, 1025→229) then applies
`TrainablePCEN` (learnable `alpha/delta/r` per band). Verified: gradients flow
through the whole note model to the PCEN params (`tests/test_pcen.py`).

## 2. The critical risk + mitigation — distribution shift

The pretrained CRNN was trained on **log-mel**; PCEN output is a different
distribution. `bn0` + the CRNN must adapt, so **do not** hit them with a high LR
from step 0. Two mitigations, use both:

**(a) Param groups — front-end fast, CRNN slow:**
```python
fe_params  = list(pcen.parameters()) + list(model.note_model.bn0.parameters())
fe_ids     = {id(p) for p in fe_params}
crnn_params = [p for p in model.parameters() if id(p) not in fe_ids]
opt = torch.optim.Adam([
    {"params": fe_params,  "lr": 1e-3},   # PCEN + bn0 adapt quickly
    {"params": crnn_params, "lr": 3e-5},  # CRNN nudged gently
])
```

**(b) Warmup — freeze the CRNN for the first ~1–2k steps** so PCEN+bn0 settle
into a log-mel-like range before the CRNN moves:
```python
WARMUP = 1500
for p in crnn_params: p.requires_grad_(False)   # freeze CRNN
# ... in the loop, at step == WARMUP:
if step == WARMUP:
    for p in crnn_params: p.requires_grad_(True)
```

## 3. Augmentation (label-preserving)

Reuse `aitabs/eval/degrade.py` — the *same* degradations the preprocessing bench
uses (so training and eval agree). All are **label-preserving** (notes unchanged):

- **reverb** (synthetic exp-decay IR), **light noise** (SNR 15–30 dB), **gain**
  (±12 dB), **sub-bass rumble** — from `DegradeConfig`.
- **micro pitch-shift ±10 cents** — timbral variation that does **not** cross a
  semitone, so integer MIDI labels stay valid (this is the safe pitch-shift the
  piano study used; larger shifts would require transposing labels).

Apply per-batch on CPU (numpy) before moving to GPU, randomized per clip, ~50%
probability per stage:
```python
import numpy as np, random
from aitabs.eval.degrade import degrade_audio, DegradeConfig

def augment_waveform(y, sr, rng):
    cfg = DegradeConfig(
        reverb_decay_sec = rng.uniform(0.2, 0.5) if rng.random() < 0.5 else None,
        snr_db           = rng.uniform(15, 30)   if rng.random() < 0.5 else None,
        gain_db          = rng.uniform(-12, 0)   if rng.random() < 0.5 else None,
        rumble_hz        = rng.uniform(35, 55)   if rng.random() < 0.3 else None,
        seed             = rng.integers(1 << 30),
    )
    y = degrade_audio(y, sr, cfg)
    if rng.random() < 0.5:                       # ±10 cent micro pitch-shift
        import librosa
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=rng.uniform(-0.1, 0.1))
    return y
```
(Keep the existing GPU `augment_batch` gain+noise too, or replace it with this.)

## 4. Colab cells (add to the existing COLAB_GPU.md flow)

```python
# after cloning repo + installing deps + downloading pretrained/checkpoints
!pip -q install pyloudnorm
# In colab_finetune_kong.py, after model build + --resume load, insert §1 (attach
# PCEN), §2 (param groups + warmup), and swap the aug call for §3. Then:
!python colab_finetune_kong.py \
    --guitarset-dir /content/guitarset \
    --guitar-techs-dir /content/guitar_techs \
    --resume /content/drive/MyDrive/kong_mixed_gpu/kong2_ft_step4000.pth --resume-step 0 \
    --pcen --augment-reverb \
    --checkpoint-dir /content/drive/MyDrive/kong_pcen_gpu \
    --pos-weight 5 --batch 8 --steps 30000 --save-every 2000
```
(`--pcen` / `--augment-reverb` are the flags you'll add to gate §1 and §3.)

## 5. Validation — the two-sided bench, per checkpoint

PCEN checkpoints need PCEN attached **before** load (the `logmel_extractor.pcen.*`
keys don't exist in a plain model). This is handled for you — `KongDetector(ckpt,
pcen=True)` / `get_detector("kong", model_path=ckpt, pcen=True)` build on the base
pretrained weights, attach PCEN, then load. So the eval + tab scripts take a
`--pcen` flag:

```powershell
# recovery on degraded audio (the point of PCEN+aug) + regression check on clean:
python scripts/eval_preprocess.py --kong-ckpt eval/kong_pcen_stepN.pth --pcen --max-clips 12
# generate a tab with a PCEN checkpoint:
python scripts/audio_to_tab.py clip.mp3 --detector kong --model-path eval/kong_pcen_stepN.pth --pcen --preprocess
```
(`eval_ensemble.py` reads Kong from the npz cache via `calibrate_detector`, which
doesn't yet take `--pcen`; use `eval_preprocess.py --pcen` for the two-sided PCEN
bench, or add pcen to the calibrate inference if you want the ensemble A/B on a
PCEN Kong.)

Decision rule: **keep the checkpoint that raises degraded-set recall without
regressing clean-set F1** (pick by held-out P/R, not train loss). The PCEN failure
mode to watch = **clean-set regression**, which means warmup was too short / CRNN
LR too high — increase `--pcen-warmup` or lower `--pcen-crnn-lr` and retrain.

## 6. Notes / gotchas
- `TrainablePCEN._smooth` loops over time frames — fine for 3 s segments (T≈300);
  if you lengthen segments, it slows. `s=0.025` is a ~1 s reverb-scale smoother.
- PCEN inits to standard values (`alpha=0.8, delta=2, r=0.5`); it does **not**
  reduce to log, hence the warmup.
- License-clean throughout: PCEN is our code; augmentation is synthetic; data is
  GuitarSet (MIT) + Guitar-TECHS (CC BY 4.0). No new NC assets.
- Combine at inference with the landed levers: PCEN model **+** `--preprocess`
  **+** `--detector ensemble --ensemble-mode gated`.
```
