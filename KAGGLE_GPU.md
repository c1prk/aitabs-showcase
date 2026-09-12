# Two GPU jobs at once — Kong PCEN (Colab) + BasicPitch (Kaggle)

Run both heavy retrains **simultaneously** on **separate GPU quotas**:

| Job | Platform | Script | Guide |
|---|---|---|---|
| **Kong PCEN + augmentation retrain** | **Colab Pro** (your compute units) | `scripts/colab_finetune_kong.py --pcen` | `RESEARCH_PCEN.md`, `COLAB_GPU.md` |
| **BasicPitch fine-tune** | **Kaggle** (separate free GPU quota) | `scripts/kaggle_finetune_basicpitch.py` | this file |

Kaggle's GPU quota (T4×2 / P100, ~30 h/week) is **separate** from Colab's compute
units, so the two jobs don't compete. Linking Colab Pro to Kaggle (Kaggle notebook
**File ▸ Link to Colab**) unlocks the extra Kaggle GPU hours without spending Colab
units — start Colab on Job A, Kaggle on Job B, and let both run overnight.

---

## Job A — Colab Pro: Kong PCEN retrain
Follow `COLAB_GPU.md` to set up data + the pretrained checkpoint, then launch with
the PCEN flags (see `RESEARCH_PCEN.md`). Checkpoints save to your Drive:
```python
!python colab_finetune_kong.py \
    --guitarset-dir /content/guitarset --guitar-techs-dir /content/guitar_techs \
    --resume /content/drive/MyDrive/kong_mixed_gpu/kong2_ft_step4000.pth --resume-step 0 \
    --pcen --augment \
    --checkpoint-dir /content/drive/MyDrive/kong_pcen_gpu \
    --pos-weight 5 --batch 8 --steps 30000 --save-every 2000
```
Start this first, then move to Kaggle while it trains.

## Job B — Kaggle: BasicPitch fine-tune (step by step)

1. **New notebook** at kaggle.com → **Code ▸ New Notebook**.
2. **Enable GPU:** right sidebar **Settings ▸ Accelerator ▸ GPU T4 x2** (or P100).
3. **Enable Internet:** **Settings ▸ Internet ▸ On** (needs a one-time phone
   verify). Required to pip-install + download the datasets.
4. *(optional)* **File ▸ Link to Colab** to draw on the extra Colab-linked GPU
   hours rather than your base Kaggle quota.
5. **Get the trainer in.** Easiest: **Add Data ▸ Upload** a folder containing
   `scripts/kaggle_finetune_basicpitch.py`, or paste it into a cell that writes
   the file. Then it's at e.g. `/kaggle/input/<your-upload>/kaggle_finetune_basicpitch.py`
   — copy it to the working dir:
   ```python
   !cp /kaggle/input/*/kaggle_finetune_basicpitch.py /kaggle/working/
   ```
6. **Install deps:**
   ```python
   !pip -q install basic-pitch librosa soundfile pretty_midi scipy
   ```
7. **Get the data** (public Zenodo; MIT + CC BY 4.0):
   ```python
   # GuitarSet (annotation + mono-mic audio)
   !mkdir -p /kaggle/working/guitarset/annotation /kaggle/working/guitarset/audio
   !wget -q https://zenodo.org/records/3371780/files/annotation.zip -O /tmp/a.zip
   !wget -q https://zenodo.org/records/3371780/files/audio_mono-mic.zip -O /tmp/m.zip
   !unzip -q /tmp/a.zip -d /kaggle/working/guitarset/annotation
   !unzip -q /tmp/m.zip -d /kaggle/working/guitarset/audio

   # Guitar-TECHS (electric; a few content zips is enough to start)
   !mkdir -p /kaggle/working/guitar_techs && cd /kaggle/working/guitar_techs && \
     for f in P1_singlenotes P1_scales P2_singlenotes P3_music; do \
       wget -q "https://zenodo.org/records/14963133/files/$f.zip?download=1" -O $f.zip && unzip -q $f.zip; \
     done
   ```
   (GuitarSet audio may unzip into an `audio_mono-mic/` subfolder — point
   `--guitarset-dir` at the folder that actually contains `annotation/` and
   `audio/*_mic.wav`; adjust after checking `!ls`.)
8. **Train** (writes the best-val SavedModel to `/kaggle/working/bp_ft`):
   ```python
   !python /kaggle/working/kaggle_finetune_basicpitch.py \
       --guitarset-dir /kaggle/working/guitarset \
       --guitar-techs-dir /kaggle/working/guitar_techs \
       --out /kaggle/working/bp_ft \
       --epochs 15 --batch 8 --lr 1e-4 --stride-frac 0.5 --onset-blur 1.5
   ```
   Watch `val=` fall and `*best -> saved` mark the kept checkpoint. It keeps the
   **best val-loss** epoch (not the last), so overfitting won't cost you.
9. **Download** the model: **File ▸ Download** `/kaggle/working/bp_ft`, or zip it:
   ```python
   !cd /kaggle/working && zip -qr bp_ft.zip bp_ft && echo done
   ```
   then download `bp_ft.zip` from the Output panel.

## After both finish — evaluate locally
Bring `bp_ft/` (Kaggle) and `kong_pcen_step*.pth` (Colab) back to `eval/` or
`data/models/`, then:
```powershell
# fine-tuned BasicPitch: calibrate + A/B it inside the ensemble
python scripts/calibrate_detector.py --detector finetuned --model-path data/models/bp_ft --max-clips 30
python scripts/eval_ensemble.py --max-clips 16 --bp-budget 16 --bp-model-path data/models/bp_ft \
    --bp-cache data/eval_results/cache_bp_ft --bp-onset <cal_O> --bp-frame <cal_F> --bp-conf <cal_C>
#   keep the fine-tuned BP only if it raises ENSEMBLE F1 (its real job).

# PCEN Kong: two-sided recovery bench (clean must not regress, degraded should recover)
python scripts/eval_preprocess.py --kong-ckpt eval/kong_pcen_stepN.pth --pcen --max-clips 12
```

## Notes
- Both trainers are **standalone** (no repo import) so they run in fresh
  Colab/Kaggle kernels.
- **Licensing:** GuitarSet (MIT) + Guitar-TECHS (CC BY 4.0) only → the fine-tuned
  BasicPitch is **fully commercial-clean** (Apache-2.0 base), unlike the
  MAESTRO-pretrained Kong weights.
- Kaggle sessions cap at ~9–12 h and idle-timeout — `--epochs 15` on this data
  fits comfortably; if you scale data up, checkpoint-and-resume or lower epochs.
- If Kaggle GPU is busy, the same `kaggle_finetune_basicpitch.py` runs unchanged
  on a second Colab runtime — it just consumes Colab units then.
