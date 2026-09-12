# CLAUDE.md

Guidance for Claude CLI and Codex when working in this repository.

## Operating Model

Cursor is the driver for daily typing, inline edits, quick visual refactors, and
fast local iteration. Claude CLI and Codex are the manager/reviewer layer:
background code reviews, deep debugging, multi-file refactors, evaluation work,
and careful handoffs while Cursor stays active.

If Claude runs out of tokens or context, stop cleanly and hand off to Codex:

1. Prefer running `/codex:transfer` before the context window is exhausted. This
   imports the current Claude Code session into a resumable Codex thread.
2. Preserve the Codex session ID and the `codex resume <session-id>` command from
   `/codex:transfer`.
3. Summarize the active goal, files touched, commands run, test results, and the
   next concrete step.
4. Leave the worktree in a resumable state. Do not abandon half-applied edits.
5. Tell Codex to read this file, `CONTEXT.md`, `PLAN.md`, and the current diff
   before acting.
6. Preserve the same contracts, tests, and licensing constraints below.

Codex should resume as the same manager/reviewer role, not as a second driver
with a different plan. The two agents should converge on the same project state.

## Verified Claude/Codex Setup

Claude CLI session setup on 2026-06-24 confirmed:

- Codex plugin installed in Claude Code: `codex@openai-codex`.
- Codex CLI available: v0.142.0 with advanced runtime support.
- Codex authentication active through ChatGPT login.
- Claude review gate disabled by default. Keep this unless the user explicitly
  asks for stop-time review blocking.

Claude slash commands to use:

- `/codex:setup`: verify Codex availability/auth; use
  `/codex:setup --enable-review-gate` only if the user wants every stop gated by
  review.
- `/codex:transfer`: transfer the current Claude session into a resumable Codex
  thread. Use this when Claude is near token limits or the user wants a seamless
  Codex continuation.
- `/codex:review --background`: start a background Codex review of local git
  state while Cursor/Claude continues.
- `/codex:review --wait`: run a small foreground review and return the exact
  review output.
- `/codex:rescue --background <task>`: delegate deep debugging, root-cause
  analysis, or a large autonomous refactor to Codex.
- `/codex:status`: check active and recent Codex work for this repository.

## Project Snapshot

AITabs Pipeline is a focused research repo for audio and future vision based
guitar transcription. The product repo handles upload, jobs, web UI, and
deployment. This repo owns the transcription brain:

- `aitabs.pipeline.audio`: Demucs separation, BasicPitch inference, onset/tempo
  analysis, rhythm helpers, and live monitor core.
- `aitabs.pipeline.mapping`: string/fret assignment and guitar setup logic.
- `aitabs.pipeline.fusion`: future audio plus vision merger.
- `aitabs.pipeline.vision`: fretboard and hand tracking stubs.
- `aitabs.export`: MIDI and GP5 export, including rhythm notation.
- `aitabs.eval`: evaluation, GP5/GuitarSet import, rhythm metrics, diagnostics.
- `scripts`: CLI entry points for smoke tests, evaluation, tuning, live monitor,
  and error analysis.
- `notebooks`: research workflows.
- `data`: local recordings, references, caches, and eval outputs.

The current technical focus is note accuracy, rhythm notation, live monitoring,
and preparing a clean path back into the main AITabs monorepo.

## Source Of Truth Docs

Read these before deep work:

- `CONTEXT.md`: current state, metrics, open threads, and licensing notes.
- `PLAN.md`: note-accuracy plan and diagnosis buckets.
- `RESEARCH_MODELS.md`: detector/model roadmap, dataset policy, and YouTube
  licensing stance.
- `RESEARCH_RHYTHM.md`: rhythm notation design and completed migration steps.
- `LIVE_MODE.md` and `RESEARCH_LIVE.md`: live monitor behavior and internals.
- `RECORDING_GUIDE.md`: how to create the owned held-out eval clips.
- `INTEGRATION.md`: contract with the main product repo.
- `README.md`: setup and top-level project shape.

Keep these docs synchronized when work changes project direction, command usage,
or public contracts.

## Hard Contracts

Do not casually change the product-facing note schema. The main AITabs API layer
expects a list of dictionaries with:

- `start`
- `end`
- `pitch_midi`
- `string`
- `fret`
- `confidence`
- `source`

Expected pipeline shape:

1. `separate_guitar`
2. `detect_notes` plus `detect_onsets`
3. optional vision frame loop
4. `fuse` or `map_sequence` fallback
5. product repo calls GP5/MIDI export

If these keys or semantics change, update `INTEGRATION.md` and any downstream
callers in the product repo in the same change.

## Current Priorities

1. Build and preserve the owned five-clip held-out eval set described in
   `RECORDING_GUIDE.md`. Do not train on those clips.
2. Run note dumping plus `scripts/analyze_errors.py` before adding any new pitch
   filter. Let the false-positive and false-negative buckets choose the next move.
3. Scaffold a pluggable detector backend so BasicPitch, recall-first BasicPitch,
   and guitar-specialized models can be A/B scored through the same harness.
4. Move toward Tier 2 fine-tuning and Tier 3 data/model scale once the Tier 1
   detector A/B path is reproducible. See `RESEARCH_MODELS.md`.
5. Continue rhythm validation on real audio, but do not let rhythm work distract
   from the upstream pitch-recall and false-positive bottleneck.
6. Treat rubato and notated-time mismatch as an evaluation problem, likely needing
   tempo-robust or DTW-style matching.

## Licensing And Commercial Safety

This is startup-bound work. Keep it license-clean.

- Do not add non-commercial dependencies or model assets.
- Do not use `madmom`; its model/data licensing is not acceptable here.
- Prefer permissive dependencies already present in `requirements.txt`.
- Treat GuitarSet as the preferred real-guitar training/eval source because it is
  MIT licensed. Verify GOAT's CC-BY terms and attribution requirements before
  using it.
- Do not use GAPS or other non-commercial datasets for product training.
- Do not use CC YouTube videos as labeled training data. They can be considered
  later for unlabeled robustness/domain adaptation only, with attribution and
  legal review before redistribution.
- Be careful with PyGuitarPro: LGPL import is acceptable, but do not vendor or
  modify it without a deliberate licensing review.
- The committed GP5 reference song files are third-party copyrighted material and
  are currently a private-repo risk. Do not expand that footprint. Before public
  release, remove them, gitignore replacements, and scrub history.

## Development Rules

- Preserve existing package layout: `aitabs.pipeline.*`, `aitabs.export.*`,
  `aitabs.eval.*`.
- Prefer small, testable changes over broad rewrites.
- Use typed dictionaries and dataclasses consistently with existing modules.
- Keep pure logic in package modules and CLI/UI concerns in `scripts`.
- Do not invent a web API or frontend in this repo.
- Avoid changing generated caches, `__pycache__`, local recordings, or eval output
  unless the task explicitly requires it.
- Do not remove the default-off harmonic suppression code merely because it failed
  one A/B pass; it is useful as a recorded experiment.
- Do not turn on new detector filters globally without an A/B result against the
  eval harness and a bucket-level explanation from `analyze_errors.py`.
- For vision/fusion, start with audio fallback behavior first, then add vision
  resolution with measurable `source` distribution.

## Commands

Use Python 3.11 or 3.12. This project declares `>=3.11,<3.13`.

Common commands:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
python scripts/smoke_audio.py data/your_clip.wav
python scripts/check_recordings.py data/eval
python scripts/eval_dataset.py data/eval --config data/eval/test1_best/best_combined.json --no-demucs --rhythm
python scripts/eval_dataset.py data/eval/guitarset --guitarset
python scripts/eval_dataset.py data/eval --config data/eval/test1_best/best_combined.json --no-demucs --dump-notes data/eval_results/notes
python scripts/analyze_errors.py data/eval_results/notes
python scripts/live_monitor.py --list-devices
python scripts/live_monitor.py --plot
```

The user prefers RTK-prefixed commands for compact output. If `rtk` is available,
use it for shell commands, including each command in a chain. If `rtk` is not
available on PATH, continue with compact direct commands and mention that RTK is
missing rather than blocking the work.

## Testing Expectations

- Run targeted tests for the touched area.
- Run full `pytest` before completing larger refactors when the environment has
  the required audio/ML dependencies.
- For audio accuracy work, report pitch F1, precision, recall, onset tolerance,
  config path, and whether Demucs was used.
- For rhythm work, include rhythm metrics where relevant: position F1,
  `note_value_acc`, `rest_F1`, tie fraction, meter, and tempo-map behavior.
- For live monitor changes, keep pure logic unit-tested in
  `aitabs/pipeline/audio/live.py` and avoid requiring PortAudio in unit tests.

## Review Checklist

When reviewing or handing off work, check:

- Are note dict keys and units preserved?
- Did the change improve the intended metric without hiding precision/recall
  tradeoffs?
- Are thresholds/configs explicit and reproducible?
- Are no new non-commercial dependencies introduced?
- Are references to copyrighted eval data avoided in new public-facing docs?
- Are docs updated if workflow, commands, or contracts changed?
- Is the next agent able to resume from the diff plus this file?

## Handoff Format

Use this compact handoff shape when pausing or switching from Claude to Codex:

```text
Goal:
Current state:
Files changed:
Commands run:
Results:
Known risks:
Next step:
Codex transfer:
```

Do not make Cursor, Claude, and Codex maintain separate mental models. Update the
project docs when the shared model changes.
