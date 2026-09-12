# Recording & tabbing guide (real-audio eval clips)

Make a small, **diverse, held-out** set of real recordings with matching tabs to
measure the pipeline honestly. These clips are your **gold eval set — never train
on them** (Tier-2 fine-tuning uses GuitarSet/GOAT + augmentation, hundreds of
clips; 5 clips would overfit instantly).

## Pick diverse clips (each stresses a different failure mode)
With ~5×30s, don't record 5 of the same thing. Target:
1. **Clean monophonic** single-note line — sanity ceiling.
2. **Let-ring fingerpicking / arpeggio** — the polyphony-recall killer (test1-style).
3. **Strummed chords** — dense simultaneous onsets.
4. **Distorted / electric** — timbre robustness.
5. **Capo or alternate voicing** — string/fret disambiguation.

(`check_recordings.py` labels each clip mono / melody+bass / chordal so you can
confirm the spread.)

## Recording hygiene
- **Record audio + (optional) video together** so they're synced.
- WAV, 44.1/48 kHz; DI or close-mic; **minimal reverb** (reverb = sustain = harder).
- **Tune precisely** before each take (or note the real tuning in the tab).
- Keep a **steady tempo** (play to a click if you can). Rubato hurts onset matching
  — the metric only auto-corrects a *constant* offset, not drift (the test3 problem).

## Tabbing requirements (so the ground truth is honest)
The loader now honours these — get them right in the `.gp5`:
- **Correct tempo**, including **tempo changes** (they're read via `mixTableChange`).
- **Correct time signature** (3/4, 6/8, …) and **tuning + capo** (read as concert pitch).
- Tab the **first track / voice 0** (what the loader reads).
- Name the file so audio and tab share a stem: `clip01.wav` + `clip01.gp5`
  (or `audio/clip01.wav` + `reference/clip01.gp5`).

## Workflow
```bash
# 1. Sanity-check the set BEFORE eval (pairing, tempo/meter/tuning, diversity)
python scripts/check_recordings.py data/eval

# 2. Score (note + rhythm)
python scripts/eval_dataset.py data/eval --no-demucs --rhythm

# 3. Diagnose where the errors are (dump notes, then categorize FP + misses)
python scripts/eval_dataset.py data/eval --no-demucs --dump-notes data/eval_results/notes
python scripts/analyze_errors.py data/eval_results/notes
```

`analyze_errors.py` tells you whether misses are **chord_inner / sustain**
(polyphony recall → needs a better model, Tier 1/2) vs **isolated** (threshold too
high → recall-first retune), and whether false positives are octave / sustain /
noise. That decides the next move before any fine-tuning.
