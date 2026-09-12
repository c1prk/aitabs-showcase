# AITabs Pipeline (Research)

Focused repo for **Phase 2** (audio accuracy) and **Phase 3** (vision + fusion).  
No web UI, no FastAPI — just the transcription brain.

The main product repo (`aitabs` monorepo) handles upload, jobs, and deployment.  
When this pipeline is ready, you merge it back — see [INTEGRATION.md](INTEGRATION.md).

## What’s in here

```
aitabs/pipeline/
  audio/      Demucs, BasicPitch, librosa onsets
  vision/     RT-DETR fretboard, MediaPipe hands
  fusion/     Merge audio + vision
  mapping/    String/fret DP optimizer (core IP)
aitabs/export/          .gp5 + MIDI (for notebook validation)
notebooks/              Step-by-step experiments
data/                   Your recordings (gitignored media)
eval/                   Accuracy scripts vs ground truth (add over time)
```

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name aitabs-research
```

Put test clips in `data/` (e.g. `data/solo.wav`).

## Workflow

| Phase | Goal | Start here |
|-------|------|------------|
| **2** | Audio note accuracy | `notebooks/audio_track.ipynb` |
| **3** | Vision + fusion | `notebooks/vision_track.ipynb` → `fusion.ipynb` |

Implement stubs in this order:

1. `aitabs/pipeline/mapping/fingering.py` — `transition_cost`, `map_sequence`
2. `aitabs/pipeline/fusion/fusion.py` — start with audio-only fallback, add vision
3. Tune thresholds in `audio/pitch.py` against your dataset

## Quick smoke test (audio)

```bash
python scripts/smoke_audio.py data/your_clip.wav
```

## Publish as its own GitHub repo

From this directory (not the monorepo root):

```bash
cd aitabs-research   # if you copied this folder out of the monorepo
git init
git add .
git commit -m "Initial pipeline research repo"
gh repo create aitabs-pipeline --private --source=. --remote=origin --push
```

Or copy `aitabs-research/` to a new folder, then run the commands above.

## Reintegrate with the product repo

See [INTEGRATION.md](INTEGRATION.md).
