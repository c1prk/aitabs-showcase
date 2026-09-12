# Rhythm notation — diagnosis and a better system

## The problem (from a MIDI-render test)

The transcribed audio was a **MIDI render of the ground-truth tab**, so onsets and
tempo are exact — yet the exported notation is wrong. Comparing ground truth vs
output on "One Summer's Day" (Joe Hisaishi):

| | Ground truth | Pipeline output |
|---|---|---|
| Meter | **3/4** | **4/4** (barlines all wrong) |
| Tempo | ♩=120 → **♩=190** at m.26 | single 120 (everything after drifts) |
| Note values | clean 8th-note tremolo | 16ths + dots + ties everywhere |

Because detection is exact here, **every error is in the rhythm-notation stage.**

## Root cause: notation is built on a flat frame

The notation engine assumes a constant tempo, a fixed meter, and "first onset =
bar 1, beat 1". Two concrete bugs follow:

1. **Meter is hardcoded.** `Gp5ExportOptions.time_signature = (4, 4)`
   (`aitabs/export/gp5_layout.py:46`); `export_gp5(..., time_signature=(4,4))`
   (`aitabs/export/guitarpro.py:34`). Nothing estimates it and the pipeline never
   passes one, so 3/4 (and 6/8, 5/4, …) are mis-barred.
2. **Single constant tempo.** `build_note_value_schedule` converts seconds→ticks
   with one `bpm` (`aitabs/export/rhythm_notation.py:200`). A mid-piece tempo
   change (120→190) makes every later onset land between grid points → the
   16th/dotted/tied mush.

Both come from quantizing in **seconds × one bpm against a fixed 4/4 grid**
instead of in **beat space against an estimated metrical grid** (beats →
downbeats → meter). Note that `TempoAnalysis` already carries `beat_times` (a
variable beat grid) and an adaptive `phase_template`, but the `note_value` engine
ignores them and uses a flat bpm — so the infrastructure is half-present and
unused.

## What the literature does (state of the art)

Rhythm transcription is a well-studied **two-stage** problem (onset quantization →
note-value/score structuring), and the consensus is to work in a metrical grid
derived from tracked **beats and downbeats**, not a constant tempo:

- **Beat + downbeat + meter from audio — madmom.**
  `DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4,6], fps=100)` returns each
  beat's time *and its position in the bar*, which yields the **time signature
  and the downbeat phase** directly. RNN observations + a Bayesian-network/HMM
  bar model. This is the canonical audio tool.
- **Beat + downbeat + time-signature from MIDI — PM2S** (`cheriell/PM2S`,
  pip-installable). Given our detected notes as MIDI, it predicts beats,
  downbeats, **time signature**, key, and hand parts with a CRNN. Since we already
  produce note events, this is the closest to a **drop-in metrical front end** for
  the notation stage.
- **MIDI-to-score by neural beat tracking** (Liu et al., ISMIR 2022): predict
  which notes are on-beat, then quantize all notes' beat positions. Confirms
  **onsets carry most of the metrical information** — exactly our (clean) signal.
- **Metrical HMM / merged-output HMM** (Nakamura et al. 2016–2017): jointly
  estimate note values, meter, and bar lines with a timing model × a musical
  grammar, "avoiding grammatically incorrect score representations" — i.e. the
  principled fix for the ties/dots mush. SOTA for clean classical notation.
- **Lightweight meter induction** (no heavy deps): autocorrelation of an
  *accented* onset signal picks the bar period (2/3/4 beats); the **Metrogram
  transform** finds time signature from the tempogram and is robust to tempo
  changes. Implementable with numpy + librosa.

## Proposed system — a layered metrical model

Replace "constant bpm + fixed 4/4" with four explicit layers; quantize in **beat
units** throughout.

1. **Tempo map / beat grid.** Use tracked beats (we already have `beat_times`;
   upgrade to a variable map) so onset→beat conversion is correct across tempo
   changes. Express every onset as a **beat position** `b = beat_index + frac`.
2. **Downbeat + meter.** Estimate beats-per-bar and the downbeat phase (madmom or
   PM2S; lightweight accent-autocorrelation fallback). Output a real
   **TimeSignature** and bar anchoring.
3. **Metrical quantization.** Snap beat positions to a per-beat subdivision
   (straight vs triplet, as the engine already chooses) **anchored on downbeats**,
   not on the first onset.
4. **Note-value assignment.** Keep the existing exact-sum decomposition
   (`notation_for_length`) but feed it the correct bar length and beat boundaries;
   later, optionally swap in an HMM/neural note-value model for cleaner beaming.

The export API and `rhythm_metrics` (`pos F1`, `note_value_acc`, `rest_F1`) stay;
we're changing how the grid is derived, and measuring with the metrics we have.

## Migration path

- **Step 1 — stop hardcoding meter + quantize in beat space. ✅ DONE (no new deps).**
  Added `TempoMap` (`audio/tempo.py`) with seconds↔beats; `gp5_import` reads the
  reference time signature and a tempo map (honouring `mixTableChange.tempo`);
  `rhythm_notation` quantizes in **beat space** via the tempo map; `time_signature`
  + `tempo_map` are plumbed through `export_gp5`, and `run_pipeline` resolves them
  (config override → reference GP5 → 4/4/constant) via `_resolve_meter`. Config:
  `time_signature`, `use_reference_meter`, `use_reference_tempo_map`.

  **Verified on test1 (One Summer's Day, 3/4, 120→190):** meter 4/4→**3/4**;
  measures 72→**138** (≈ 137 expected); tie-fraction (mush proxy) **0.230→0.003**.
  Commercial-safe — uses only PyGuitarPro + numpy already in the project.

  Still open (Step 2): **downbeat phase** is still "first onset = bar 1 beat 1";
  real recordings (not reference-seeded) need an estimated meter + tempo map.
- **Step 2 — real meter/downbeat/tempo-map estimation. ✅ DONE (license-clean).**
  `audio/tempo.py` now has `estimate_meter` (accent-autocorrelation over the
  per-beat accent signal; numpy only — picks numerator ∈ {2,3,4} and the downbeat
  phase) and `TempoMap.from_beat_times` (variable tempo from tracked beats).
  `run.py:_resolve_meter` falls back to these when there is no reference
  (config: `estimate_meter`, `estimate_tempo_map`). Verified: synthetic 3/4 vs
  4/4 are classified correctly with the right phase. **No madmom** (its models are
  CC-BY-NC); uses only numpy/librosa (ISC) — commercial-safe.
- **Step 3 — downbeat anchoring (pickup) ✅; full note-value model deferred.**
  `lead_in_beats` (estimated phase) is plumbed through export and shifts the first
  downbeat onto a barline (leading time → rests), verified. The *greedy*
  decomposition is kept — on a correct grid it already yields clean notation
  (test1 tie-fraction 0.003). A learned **metrical-HMM / neural note-value** model
  (Nakamura) remains future work: it needs training data and mainly helps on noisy
  real-recording onsets, so it is deferred until Step 2 estimation is exercised on
  real audio.
- **Step 4 — joint tempo-spacing + phase alignment. ✅ DONE (numpy-free).**
  *Symptom (real recording):* a steady 4/4, ♩=96 eighth-note phrase notated as
  16ths + syncopation with a leading rest. *Root cause:* the estimated BPM was a
  few % off, so the onset grid was mis-**spaced** (not just offset). Steps 1–3
  correct **phase** (`estimate_grid_phase`) and **meter** but not **spacing** — a
  3% spacing error makes clean eighths drift across the grid and snap to off-beat
  sixteenths (verified: onset 10 of 16 lands on 16th-step 21 instead of 20).
  `optimize_grid_alignment` searches a *small* tempo band (±6%, deliberately too
  narrow to reinterpret a real sixteenth as an eighth — that needs ~2×) × phase and
  rescales the whole tick stream to the grid on which onsets fall most simply
  (snap cost penalised for choosing sixteenths over eighths). It is a strict
  **no-op `(1.0, 0.0)`** when the identity grid is already best, so MIDI-render
  takes (test1) are byte-for-byte unchanged — i.e. this fixes the over-quantization
  *without* lowering any threshold or regressing tight material. Resolution is also
  made **jitter-adaptive** (`residual_jitter_beats`): a looser take needs a larger
  margin before the per-beat snapper drops to sixteenths, so timing noise alone no
  longer manufactures 16th notes. Opt-in via `optimize_grid` (config
  `optimize_grid`, plumbed through `export_gp5`/`Gp5ExportOptions`); default-off so
  the GP5-reference eval keeps exact scoring. Unit-tested in
  `tests/test_rhythm_notation.py` (recovers a 3% error, preserves genuine 16ths,
  no-ops on exact onsets).

## Choosing the notation config (which knob for which symptom)

Real recordings fail in **two opposite** ways; the fix differs, so diagnose first
with `scripts/analyze_rhythm.py` (profiles the predicted-onset IOI distribution
and, given a reference, returns a verdict). Decision table:

| Symptom | Diagnostic verdict | Cause | Fix |
|---|---|---|---|
| All eighths, ground truth has 16ths | **NOTATION-FLATTEN** (pred onsets carry sub-eighth IOIs) | `snap_dominant_grid` locked the phrase to one grid | `snap_dominant_grid=False`, keep `legato=True` |
| All eighths, ground truth has 16ths | **DETECTOR/RECALL** (pred onsets are ~eighth-spaced) | fast notes never detected | upstream onset/recall — no notation knob helps |
| False 16ths + syncopation, steady take | (onsets drift across grid) | few-% BPM estimate error → mis-*spaced* grid | `optimize_grid=True` |

**`snap_dominant_grid` flattens mixed rhythm** (verified, `repro_flatten` harness):
it picks ONE global subdivision, so when sub-eighth onsets are a minority the mean
snap error to the eighth grid stays under `dominant_grid_max_mae` (0.12 beats) and
*every* sixteenth collapses to an eighth. The tipping point is ~48% sixteenths
(0.12 / 0.25). With `legato` on top, durations then fill to eighth length →
"nearly all eighth notes" (the vidtest1 symptom: pred onsets were 35% sub-eighth,
render was all eighths). Reserve `snap_dominant_grid` for genuinely steady,
single-resolution takes.

**`optimize_grid` is not a general cleaner.** It targets a tempo-*spacing* error;
its eighth-preferring complexity penalty drops genuine minority sixteenths
(8→4 in the same harness), so it slightly *hurts* mixed-rhythm material that has no
tempo error. Enable only on a confirmed BPM-estimate error.

**Recommended profile for varied real music:** `legato=True`,
`snap_dominant_grid=False`, `optimize_grid=False`. The per-beat coarsest-grid
quantizer (`quantize_onset_ticks`) already chooses eighth-vs-sixteenth *per beat*,
so it keeps mixed resolution while absorbing jitter; `legato` makes durations
readable. This recovered `{8th:17, 16th:6}` from a `[8,8,8,8,16,16]` phrase with
±25 ms jitter (true `{8th:16, 16th:8}`), where `snap_dominant_grid+legato` gave
all eighths.

## Step 5 — `auto_rhythm`: self-configuring notation (✅ DONE)

The decision table above is real, but a caller cannot know per clip whether a take
is legato, has a tempo error, or mixes sixteenths — hardcoding flags per recording
is the wrong design (it is what production transcribers like Songsterr /
Songscription avoid). `auto_rhythm=True` infers all three decisions from the
onsets, so there is **one mode and no per-clip flags**:

1. **Mixed-resolution grid, always.** Uses the per-beat coarsest-grid quantizer;
   never the global dominant-grid flatten. Each beat picks eighth-vs-sixteenth on
   its own, so a phrase that is mostly eighths with occasional sixteenths keeps
   both.
2. **Tempo-spacing correction only when real.** Runs `optimize_grid_alignment`,
   now **self-gating**: it accepts the rescale only when it improves the *raw*
   (un-penalised) snap fit by ≥20% — i.e. a genuine BPM error. On mixed-rhythm
   material with no tempo error the guard returns `(1.0, 0.0)`, so the penalty can
   no longer relabel genuine minority sixteenths as eighths.
3. **Legato vs rests from evidence.** `offsets_reliable` takes the median ratio of
   each note's sounding duration to its IOI: when offsets are uniformly short noise
   (BasicPitch's common failure) the median is low → drive note values from the IOI
   (legato), suppressing the staccato-and-rests artifact; when notes genuinely
   sustain toward the next onset the median is ~1 → trust offsets and insert real
   rests.

Verified across five regimes in `tests/test_rhythm_notation.py` with **no flags**:
mixed 8th/16th + unreliable offsets (vidtest1) keeps the sixteenths and drops the
spurious rests; alternating held/short notes get real rests; a 3% tempo error is
auto-corrected; clean eighths stay clean (no manufactured sixteenths); a dense
sixteenth run is preserved. `auto_rhythm` overrides `snap_dominant_grid` / `legato`
/ `optimize_grid` and is plumbed through `Gp5ExportOptions` / `export_gp5` and the
eval config. It is the recommended setting for real recordings; the three manual
flags remain for targeted experiments and exact GP5-reference eval scoring.

A learned metrical-HMM / neural note-value model (Nakamura) is still the longer-term
ceiling for hard cases (triplets vs swing, nested tuplets), and needs training data;
`auto_rhythm` is the license-clean heuristic that removes per-clip tuning today.

## Beat-grid validation on GuitarSet ground truth (2026-07-08)
Until now rhythm was only scored against notated GP5 references (a synthetic,
constant-tempo grid). GuitarSet ships MIT-licensed `beat_position` annotations —
the *actual performed* beat/downbeat times — so we can now score the metrical
grid honestly on real audio. Harness: `scripts/eval_rhythm_guitarset.py`
(`estimate_tempo` + `detect_onsets` + `estimate_meter` → `score_beats` via
mir_eval); pure loaders/scorer in `aitabs/eval/`; tests in
`tests/test_rhythm_beats.py`.

20 player-00 solo clips:

| metric | raw | +refine_beats_to_onsets |
|---|---|---|
| beat_F | 0.623 (median 0.747) | 0.522 |
| downbeat_F | **0.125** (median 0.000) | 0.022 |
| tempo err (octave-folded) | 2.64% | 2.64% |
| meter correct | 85% | 90% |

Takeaways:
1. **Tempo is solid, beats are OK, downbeats were the failure — now fixed.**
   `estimate_meter` used to pick bar-phase from onset accents, but solo/fingerstyle
   guitar carries no energy accent on beat 1, so phase was noise (1/20 clips
   downbeat_F > 0.5). librosa also drops the first beat, phase-shifting the grid.
2. **`refine_beats_to_onsets` is a regression on metronomic solo material** —
   snapping tracked beats to performed onsets pulls them off the true grid
   (onsets aren't on beats under syncopation/rests). Keep it OFF by default; it
   was designed for jittery live takes, not clean performances.

### Downbeat fix — origin anchor (2026-07-08)
`estimate_meter` now takes bar-*length* (numerator) from accent-autocorrelation
(unchanged, ~85% right) but bar-*phase* from an **origin anchor**: assume the
excerpt starts on a downbeat and extrapolate the beat grid back to t≈0, so
`phase = round(beat_times[0] / period) % numerator` (`_phase_from_origin`).
Empirical basis: GuitarSet excerpts start on a downbeat **20/20**, and the origin
rule gives the correct phase **19/20** vs accent ~0/20. Gated by accent confidence
(`_ACCENT_PHASE_MIN_CONF = 0.5`): a *confident* accent (a real pickup that syncs
the bar to a strong 2nd beat) still wins; only the weak-accent solo case falls
back to origin. **downbeat_F 0.125 → 0.511** (median 0.667, 10/20 > 0.5), beats/
tempo unchanged (phase-only edit). `origin_downbeat=True` default; pass False to
restore pure accent phase.

Next lever is now **beat tracking itself**: downbeat_F is capped by beat_F (0.623),
and the clips still at downbeat_F 0 are beat-tracking failures (beat_F 0.0–0.52),
not phase. Improve the beat tracker (librosa is weak on sparse solo guitar) or add
onset-informed correction that doesn't over-snap, validated with
`eval_rhythm_guitarset.py`. Note-value/rest notation sits on top of the grid, so
grid first.

## Sources
- madmom downbeat tracking — madmom.features.downbeats (DBNDownBeatTrackingProcessor).
- PM2S: Performance MIDI to Score — github.com/cheriell/PM2S.
- Liu et al., "Performance MIDI-to-Score Conversion by Neural Beat Tracking," ISMIR 2022.
- Nakamura et al., "Rhythm Transcription of Polyphonic MIDI Performances" (SMC 2016);
  "Note Value Recognition … Markov Random Fields" (TASLP 2017).
- Toiviainen & Eerola, "Autocorrelation in meter induction: the role of accent structure."
- librosa tempogram / Metrogram transform (time-signature, tempo-change robust).
