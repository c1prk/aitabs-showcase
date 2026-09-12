# GPU fine-tuning on Google Colab

Trains the Kong CRNN on GuitarSet with `--pos-weight` on a Colab GPU (T4 ≈
15-40× this CPU), so we can run the ~50-100K steps where fine-tuned Kong reaches
0.87-0.91 F1 (Riley et al.). Self-contained: `scripts/colab_finetune_kong.py`
has no `aitabs`/repo dependency.

## What you do
1. New Colab notebook → **Runtime ▸ Change runtime type ▸ GPU (T4)**.
2. Run the cells below in order.
3. When it finishes (or at any checkpoint), download `kong_ft_*.pth` from your
   Drive back to `data/models/kong_guitarset_ft_gpu/` on this machine — I'll
   calibrate + eval it here.

## Data: two options
- **A (exact match, recommended):** zip your local prepared set and put it on Drive:
  - locally: zip `data/eval/guitarset` → `guitarset.zip`, upload to `MyDrive/`.
- **B (no upload):** download GuitarSet fresh from Zenodo in Colab (public, MIT).
  The `_solo` filter in the script makes both paths give the same 150 train clips.

## Cells

```python
# 1. GPU + deps
!nvidia-smi -L
!pip -q install piano_transcription_inference librosa soundfile

# 2. Mount Drive (for data in + checkpoints out)
from google.colab import drive; drive.mount('/content/drive')

# 3a. Upload the trainer: drag scripts/colab_finetune_kong.py into the Colab
#     file browser (left panel), or copy it from the repo. Then:
!ls -la colab_finetune_kong.py

# 4. Get data — OPTION A (your zip on Drive):
!mkdir -p /content/guitarset && unzip -q /content/drive/MyDrive/guitarset.zip -d /tmp/gs
# point --guitarset-dir at whatever dir ends up containing annotation/ and audio/
!ls /tmp/gs
#   OPTION B (fresh from Zenodo, ~1-2 min):
# !mkdir -p /content/guitarset/annotation /content/guitarset/audio
# !wget -q https://zenodo.org/records/3371780/files/annotation.zip -O /tmp/a.zip
# !wget -q https://zenodo.org/records/3371780/files/audio_mono-mic.zip -O /tmp/m.zip
# !unzip -q /tmp/a.zip -d /content/guitarset/annotation
# !unzip -q /tmp/m.zip && mv annotation/* /content/guitarset/annotation/ 2>/dev/null; mv audio_mono-mic/*.wav /content/guitarset/audio/

# 5. Pretrained Kong checkpoint (~165 MB) from Zenodo
!wget -q "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1" -O /content/kong_pretrained.pth
!ls -la /content/kong_pretrained.pth

# 6. Train (checkpoints to Drive so they survive disconnects)
!python colab_finetune_kong.py \
    --guitarset-dir /content/guitarset \
    --pretrained /content/kong_pretrained.pth \
    --checkpoint-dir /content/drive/MyDrive/kong_posw5_gpu \
    --pos-weight 5 --batch 8 --lr 1e-4 --steps 50000 --save-every 2000
```

## Notes
- Set `--guitarset-dir` to the folder that actually contains `annotation/` and
  `audio/` with `*_mic.wav` (adjust after step 4 depending on how the zip unpacks).
- Checkpoints save in `piano_transcription_inference` nested format → load
  straight into `PianoTranscription` for local calibration/eval.
- If a T4 disconnects, re-run with `--resume /content/drive/MyDrive/kong_posw5_gpu/kong_ft_step<N>.pth --resume-step <N>`.
- Experiments worth queueing on GPU (cheap once it's fast): `--pos-weight 3/5/10`,
  and `--augment` on vs off. Pick the winner by **held-out F1** (calibrate locally),
  not training loss.
- Locally, calibrate a downloaded checkpoint with:
  `python scripts/calibrate_detector.py --detector kong --model-path <ckpt> --max-clips 30 --cache-dir data/eval_results/cache_gpu`

## Mixing Guitar-TECHS (electric, CC BY 4.0) into training

Broadens the detector beyond GuitarSet steel-acoustic to electric + techniques.
Commercial-safe (CC BY 4.0, Zenodo 14963133). Add these cells:

```python
# Download Guitar-TECHS (~4.1 GB) and unzip
!mkdir -p /content/guitar_techs && cd /content/guitar_techs && \
 for f in P1_singlenotes P1_chords P1_scales P1_techniques \
          P2_singlenotes P2_chords P2_scales P2_techniques P3_music; do \
   wget -q "https://zenodo.org/records/14963133/files/$f.zip?download=1" -O $f.zip && unzip -q $f.zip; \
 done && ls /content/guitar_techs
```

Then add `--guitar-techs-dir /content/guitar_techs` to the training command:

```python
!python /content/colab_finetune_kong.py \
    --guitarset-dir /content/guitarset \
    --guitar-techs-dir /content/guitar_techs \
    --pretrained /content/kong_pretrained.pth \
    --checkpoint-dir /content/drive/MyDrive/kong_mixed_gpu \
    --pos-weight 5 --batch 8 --steps 60000 --save-every 2000
```

The trainer prints `GuitarSet: N segments`, `Guitar-TECHS: M segments`, `TOTAL`.
Uses the DI (direct-input) audio + 6-track per-string MIDI. Same loader lives in
`aitabs/eval/guitar_techs_import.py` for local use. Nylon (your own recordings)
can be added the same way once labeled.
