# Detector and model roadmap

This repo's main bottleneck is upstream pitch detection: missed notes and false
positives before string/fret mapping or rhythm export ever get a fair chance.
BasicPitch plus post-processing is useful, but the current evidence says it is
near a ceiling on dense real guitar.

## Working conclusion

The next accuracy jump should come from better detector backends and training
data, not another broad heuristic filter.

Current symptoms:

- Missing notes are mostly a polyphonic recall problem: inner voices, let-ring
  notes, quiet notes, and dense arpeggios fail before mapping/export.
- False positives need per-clip diagnosis before filtering. The simple octave
  harmonic filter was A/B tested and hurt mean pitch F1, so it stays default-off.
- Timing is usually tight when notes match, so pitch/event detection is the first
  lever. Rubato remains an evaluation issue, not a detector issue.

## Commercial-product lesson

Use Songsterr and Songscription as strategic signals, not implementation
templates:

- Songsterr points toward a hybrid of AI plus large human tab coverage. That
  gives a reliability floor that pure audio transcription does not have.
- Songscription-style systems point toward learned sequence models and high
  quality paired data, not hand-built post-processing alone.
- Both directions imply the moat is data plus model training. Heuristics can
  clean edges, but they cannot recover notes the detector never emitted.

Before making product or legal claims about either service, re-check current
public sources. Treat these notes as planning guidance from research, not a
vendor-spec document.

## Roadmap

### Tier 0 - owned held-out eval set

Create five diverse, fully tabbed, approximately 30 second recordings and keep
them as the gold validation set. Never train on these clips.

Target coverage:

1. Clean monophonic single-note line.
2. Let-ring fingerpicking or arpeggio.
3. Strummed chords.
4. Distorted or electric timbre.
5. Capo or alternate voicing.

Use:

```powershell
python scripts/check_recordings.py data/eval
python scripts/eval_dataset.py data/eval --no-demucs --rhythm
python scripts/eval_dataset.py data/eval --no-demucs --dump-notes data/eval_results/notes
python scripts/analyze_errors.py data/eval_results/notes
```

Decision rule:

- If misses are `chord_inner` or `sustain`, prioritize better polyphonic models.
- If misses are mostly `isolated`, try a recall-first threshold/config retune.
- If false positives are duplicate/sustain, tune merge/min-length behavior.
- If false positives are isolated/noisy, test a different detector or front end.

### Tier 1 - pluggable detector A/B

Add a detector abstraction so the same eval harness can compare backends without
changing downstream mapping/export code.

Minimum interface:

```python
detect_notes(audio_path: str, config: PipelineConfig) -> list[NoteEvent]
```

Backends to evaluate:

- current BasicPitch backend
- recall-first BasicPitch config
- high-resolution/domain-adapted guitar transcription model, if license permits
- FretNet or another guitar-specialized transcription model, if license permits
- any CRNN/MT3-style model only after license and dependency review

All backends must emit the existing `NoteEvent` shape:

- `start`
- `end`
- `pitch_midi`
- `confidence`

Score each backend on GuitarSet, existing song clips, and the owned five-clip
held-out set. Report pitch F1, precision, recall, onset tolerance, runtime,
dependency cost, and commercial-license status.

### Tier 2 - fine-tuning

If zero-shot detector swaps are not enough, fine-tune on commercial-safe paired
data and augmentation.

Preferred sources:

- GuitarSet: real guitar, string/fret labels, MIT license.
- GOAT: verify CC-BY 4.0 terms and attribution requirements before use.
- Owned recordings not reserved for held-out eval.
- Synthetic renders only when the underlying tab/audio rights are clean.

Avoid:

- GAPS or other non-commercial datasets for product training.
- Third-party copyrighted GP tabs as training data.
- The five owned held-out eval clips.

Augmentation should cover pickup type, amp/IR, light distortion, EQ, room color,
noise, and gain, while preserving note labels.

### Tier 3 - data moat and sequence model

Longer term, train an audio-to-tab or audio-to-note sequence model on a large
paired corpus. This is the Songscription/Songsterr direction: model capacity plus
data scale, optionally with retrieval for known songs.

Do this only after:

- the eval harness is stable,
- detector backend A/B is automated,
- training data provenance is tracked,
- public/commercial licensing risks are resolved.

## CC YouTube policy

Creative Commons YouTube videos are not a shortcut to labeled training data.

Use them only as unlabeled or weakly labeled domain-diversity material after the
owned/marker-based pipeline exists. They do not provide reliable string/fret
ground truth, and arbitrary camera angles make geometry harder.

Rules:

- Vet that the uploader plausibly owns the video/performance.
- Keep attribution for CC-BY material.
- Do not assume a CC video license covers the underlying song composition.
- Do not redistribute downloaded audio/video or package it in a dataset without
  legal review.
- Prefer self-recorded marker-based video for labeled vision data because it gives
  clean ownership, camera geometry, and automatic labels.

CC YouTube can help later with robustness checks, self-supervised pretraining, or
markerless-domain adaptation. It should not be the foundation.
