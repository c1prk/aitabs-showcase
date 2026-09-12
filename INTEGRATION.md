# Reintegrating with the main AITabs repo

This research repo uses the **same Python package layout** as production:

- `aitabs.pipeline.*`
- `aitabs.export.*`

That way notebooks and tests here work unchanged after merge.

## When you’re ready to merge

### Option A — Copy directories (simplest)

In the **main** monorepo:

```bash
# From main repo root, with research repo checked out beside it
rsync -av --delete ../aitabs-pipeline/aitabs/pipeline/ aitabs/pipeline/
rsync -av ../aitabs-pipeline/aitabs/export/ aitabs/export/
```

Run main repo tests:

```bash
AITABS_STUB_PIPELINE=0 ./scripts/test.sh   # when pipeline is ready
```

### Option B — Git subtree (keeps history)

From the **main** repo:

```bash
git remote add pipeline-research git@github.com:YOU/aitabs-pipeline.git
git fetch pipeline-research
git subtree pull --prefix=aitabs/pipeline pipeline-research main --squash
```

Adjust paths if you only subtree `pipeline/`.

### Option C — Pip dependency (optional)

Tag a release in this repo and in main `requirements.txt`:

```
aitabs-pipeline @ git+https://github.com/YOU/aitabs-pipeline@v0.2.0
```

Only worth it if you want versioned releases between repos long-term.

## Contract with the API layer

The main repo’s `aitabs/api/services/pipeline.py` expects:

**Input:** path to uploaded `.wav` / `.mp4` / etc.

**Output:** `list[dict]` with keys:

- `start`, `end` (seconds)
- `pitch_midi`, `string`, `fret`, `confidence`
- `source` (e.g. `"vision"`, `"audio_fallback"`)

**Functions called (in order):**

1. `separate_guitar`
2. `detect_notes` + `detect_onsets` (snap)
3. Vision frame loop (when video)
4. `fuse` or `map_sequence` fallback
5. Main repo calls `export_gp5` / `export_midi` — not your job here

Do not change those dict keys without updating `aitabs/api` and `apps/web` in the main repo.

## Checklist before merge

- [ ] `notebooks/audio_track.ipynb` runs on a real clip end-to-end
- [ ] `mapping/fingering.py` implemented and sane on one known song
- [ ] `fusion.py` at least audio-fallback; vision path tested on one video
- [ ] Documented confidence thresholds in `audio/pitch.py`
- [ ] No extra dependencies without adding them to main `requirements.txt`
