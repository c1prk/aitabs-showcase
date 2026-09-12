# Vision track — markers, hands, and technique detection (research)

Vision's job is **string/fret disambiguation + technique**, NOT pitch — audio
(BasicPitch) already gives pitch + onset + pitch-bend. A pitch maps to several
candidate (string, fret) positions; the fretting hand picks the right one. So
vision lifts **tab F1** and adds technique labels; it is a fallback-safe layer.

## 1. Hands: is MediaPipe the best? (commercial verdict)

**Yes, for a commercial product — use MediaPipe.** The more accurate 2025 models
are licensing traps:

| Model | Accuracy / occlusion | Speed | License | For us |
|---|---|---|---|---|
| **MediaPipe Hands** | good (R²≥0.90); recall ~halves under heavy occlusion | real-time | **Apache-2.0** | ✅ ship it |
| **WiLoR** (2025) | SOTA, occlusion-robust, 3D, 130+ FPS | fast | research; **built on MANO** | ❌ R&D only |
| **HaMeR** (2024) | strong ViT 3D; degrades at extreme occlusion | slower | research; **built on MANO** | ❌ R&D only |

The catch: **WiLoR/HaMeR use the MANO hand model, whose license is
non-commercial.** Great for offline R&D/validation, not shippable. MediaPipe
(Apache-2.0, no MANO) is the commercial-safe choice and is accurate enough when
the fretboard geometry is solved by markers (below). Revisit WiLoR only if a
commercial MANO license or a clean re-implementation appears.

## 2. Fretboard geometry: ArUco markers + homography (no annotation)

Don't train a fretboard detector or hand-annotate. Put **ArUco fiducial markers**
on the guitar; `cv2.aruco` (OpenCV, Apache-2.0) detects them every frame and
`cv2.findHomography` rectifies the neck into a **canonical fretboard** with correct
**logarithmic fret spacing** — math, not learning, robust to the guitar moving.
Zero annotation, no training, commercial-safe.

Later (optional) bootstrap to markerless: marker sessions auto-label frames →
train RT-DETR/keypoints (Apache) on those auto-labels → markerless model, still no
hand labeling.

## 3. Technique detection: audio-led, vision-assisted (key reframing)

"Which notes are pressed / barred / slid / bent…" is **not** a pure-vision
classification problem (that needs huge labeled video). Most techniques live in
the **audio pitch-contour / onset** domain — and BasicPitch already emits pitch
bends. Vision *confirms and localizes*. Split:

| Technique | Audio signature (primary) | Vision signature (confirm/locate) |
|---|---|---|
| **Fretted note** | pitch sounds | fingertip at (string, fret) |
| **Open string** | pitch = open-string pitch | no finger on that string |
| **Barre** | several strings sound at one fret | one finger laid **flat** across strings |
| **Slide** | continuous **glissando** between frets | fingertip translates **along** a string |
| **Bend** | pitch rises **without** a fret change (microtonal; BasicPitch bend) | finger pushes string **sideways** |
| **Hammer-on / pull-off** | onset with **no right-hand pluck**, legato; pitch steps up/down | left finger lands/lifts, no pluck |
| **Vibrato** | small rapid pitch oscillation | small rapid finger oscillation |
| **Palm mute** | damped/short/dark timbre | right-hand palm on bridge |

Build technique detection as **fusion**: audio fires the event (bend, slide,
vibrato, legato onset), vision supplies finger/barre/trajectory. Don't try to
train a vision-only technique classifier first.

## 4. Recording setup with markers (step by step)

**Hardware / framing**
1. Phone or webcam on a tripod / overhead arm; frame nut → ~12th fret with the
   fretting hand visible. 1080p, 30–60 fps. **Lock focus + exposure.**
2. Even, diffuse lighting; avoid glare on the strings.
3. Record **video + audio together** (phone does both → synced). If possible also
   capture a clean DI/close-mic for the audio track.

**Markers**
4. Print **ArUco** markers (`cv2.aruco.DICT_4X4_50`), ~3–4 cm, on matte paper,
   distinct ID each. (Or one rigid **ArUco/ChArUco board** clipped to the body —
   most stable, easiest to calibrate.)
5. Exact placement — three zones **neither hand ever touches**, rigid + flat:
   - **M0:** upper bout at the neck/body joint (bass side), flat on the body top.
   - **M1:** lower bout at the neck/body joint (treble side) — M0+M1 straddle the
     neck axis at the body end (wide baseline).
   - **M2:** headstock face above the nut (flat area) — braces the nut end so the
     grid is stable along the whole neck.
   Avoid: on the fretboard/strings (hand covers), over the soundhole/bridge
   (strumming arm), a single marker, or markers close together / on curved
   surfaces.
6. **Geometry:** a flat **homography only works if markers are coplanar with the
   fretboard**; body-top markers are in a lower plane (parallax). **Prefer
   `cv2.solvePnP`** — measure each marker's 3D position once in a guitar frame
   (origin = nut center, X along neck, Y across strings, Z up) + scale length +
   string spacing, then project the 3D fret grid each frame. This lets markers sit
   anywhere rigid/visible. (If using homography instead, shim markers to fretboard
   height so they're coplanar.) Fixed rig = measure once.

**Per session**
7. Start each take with a **2–3 s still calibration shot** (guitar facing camera,
   markers clearly visible) to confirm detection.
8. Play the pieces you've tabbed (reuse your `.gp5`s). Keep the guitar mostly in
   frame; markers can rotate/translate (homography handles it) but must stay visible.

**Software flow (to implement in `aitabs/pipeline/vision/`)**
9. `cv2.aruco.detectMarkers` → marker corners per frame.
10. `cv2.findHomography(marker_img_pts → canonical_fretboard_pts)` → rectify.
11. Canonical grid: fret n at `1 − 1/2^(n/12)` of scale length; strings evenly
    spaced (`fretboard.py::fretboard_coordinates`).
12. MediaPipe Hands → fingertips (8/12/16/20) + index PIP/MCP (for barre/flatness).
13. Warp fingertips through the homography → (string, fret) per finger
    (`hands.py::fingertip_to_string_fret`).
14. Fuse with audio: pitch → candidate positions → vision picks; technique from
    the audio signatures above (`fusion/fuse`).

## 5. Commercial-safe tool stack
- **OpenCV** (incl. `aruco`) — Apache-2.0
- **MediaPipe Hands** — Apache-2.0
- **RT-DETR** via HF `transformers` (later, markerless) — Apache-2.0 (NOT Ultralytics)
- Avoid: Ultralytics YOLO (AGPL), MANO-based hand models (WiLoR/HaMeR, non-commercial)

## 6. Next steps
1. Implement ArUco→homography `fretboard_coordinates` + an overlay-debug script
   (draw the predicted grid on the frame).
2. Implement MediaPipe `track_hands` + `fingertip_to_string_fret`.
3. Record one marked clip; verify the grid sits on the real strings/frets.
4. Implement `fuse()` (audio-led) and measure **tab F1** vs audio-only.
5. Layer technique flags from audio (bend/slide/vibrato/legato) + vision (barre/finger).

**Sources:** WiLoR (arXiv 2409.12259) · HaMeR (CVPR 2024) — both MANO-licensed;
MediaPipe (Apache-2.0); OpenCV ArUco; Cicconet thesis (guitar fiducials +
homography); CV guitar fingering recognition surveys (HMM/GMM/CNN).

---

## 7. 2024–25 literature & the learned-fusion path (research update 2026-07-04)

The marker→homography→MediaPipe→audio-led-fusion plan above is **Stage 1**
(geometric, no training, ship-safe). Recent work confirms it and points to a
higher-ceiling **Stage 2** (learned fusion).

**Direct guitar precedent — TapToTab (2024, arXiv 2409.08618).** Uses the *same
role split we do* (audio = pitch/onset, video = string/fret), synced by onset
detection. It detects the fretboard with YOLOv8-OBB — but **Ultralytics YOLO is
AGPL-3.0**, a product blocker, so our ArUco+homography (§2) is the license-clean
equivalent. No public data/numbers; it validates the *approach*, it's not a model
to reuse.

**The learned ceiling — audio-visual cross-attention (piano is ahead).** Two-Stage
AV Fusion Piano Transcription (TASLP 2024, `itec-hust/AMT_project`): audio encoder
+ visual encoder + **cross-attention** → **F1 98.6%** vs single-modal. This is
Stage 2's target architecture. Retarget the visual branch to guitar, and **use
MediaPipe keypoint *sequences* as the visual features, not raw pixels** — compact,
occlusion-tolerant, license-clean — fed to a small GCN/transformer (cf. CRNN-GCN
piano skeleton models, keypoint + context-aware graph nets).

**Motion blur is the quantified enemy.** Research shows MediaPipe loses **~50% of
hands under diagonal motion blur**; fast strumming/picking is exactly that regime.
Our design already dodges it: **audio owns timing, vision only reads the
slower-moving fretting-hand position** — so picking-hand blur is irrelevant. Add
**temporal (Kalman/HMM) smoothing** over (string,fret) since the fretting hand is
continuous.

**Data is the make-or-break — there is NO open guitar video+tab dataset.**
GuitarSet / EGDB / Guitar-TECHS are audio+tab only (Guitar-TECHS "video" is ego/exo
*mp3 audio*, not usable frames). Paths, in order: (a) **owned self-collected marked
video** (§4), auto-labeled by running the audio pipeline — Stage 1 needs almost
none and it's commercial-clean; (b) **SynthTab (2309.09085)** synthetic tab +
rendered hands to pretrain the visual branch; (c) **self-supervised pretraining**
on unlabeled playing video via audio↔frame temporal correspondence (free labels —
cf. weakly-supervised visual instrument-action detection, 1805.02031) — but YouTube
is copyright, so representation pretraining only, legal review, never redistributed
(same stance as the CC-YouTube audio policy).

**Symbolic cleanup — Fretting-Transformer (2506.14223)** (MIDI→tab) can consume
fused pitch + partial-vision string cues and emit clean, playable tablature: a
learned upgrade to `map_sequence` and a fusion regularizer alongside the physical
playability prior.

**Stage-2 milestones (gated on Stage 1 landing + data):** MediaPipe-keypoint visual
encoder → self-supervised AV pretrain on owned video → cross-attention fusion →
compare **tab F1** to the Stage-1 geometric fusion; ship only if it beats Stage 1
on owned held-out (dev-rules gate).

**Bottom line:** Stage 1 (this doc's marker plan) is the near-term unlock — needs no
dataset and no AGPL code, and slots into the `aitabs.pipeline.vision`/`fusion` stubs
behind the `source` contract. Stage 2 (cross-attention, proven at 98.6% on piano) is
the higher ceiling, gated entirely on getting owned/synthetic paired video+tab data.

## 8. Approach re-validation (2026-07-04) — Stage 1 confirmed, 3 refinements

Stress-tested the Stage-1 choice (ArUco multi-marker homography + MediaPipe +
audio-led fusion) against the alternatives. **Verdict: it is the right approach for
our constraints (no video dataset, license-clean, robust).** The alternatives lose:
- **Markerless learned fretboard** (YOLO + Stacked Hourglass, as in TapToTab/modern
  chord CV) needs **training data we don't have** and the detectors are **AGPL
  (Ultralytics YOLO)**. Correctly deferred — the marker-auto-labeling bootstrap in
  §2 is the clean route to markerless later.
- **End-to-end learned AV fusion** needs lots of paired data and has convergence
  issues on small data; two-stage/pretrained-feature fusion (our Stage 2) is the
  documented fix. Geometric fusion is correct with zero data.

**Refinements to fold in:**
1. **Use ≥2–3 markers / a ChArUco board, never one.** A *single* planar marker at an
   acute angle has a two-solution pose ambiguity → random jitter. Multiple markers'
   corners over-determine the homography and kill it. (`vision_to_tab.py` already
   pools *all* visible markers' corners — keep it that way; a rigid ChArUco board is
   the most stable.) For non-coplanar markers use `solvePnP` (which itself uses a
   homography internally for planar points) + the measured 3D layout (§4).
2. **Add temporal smoothing** (Kalman/HMM over the homography and the per-frame
   (string,fret) cells) — the fretting hand is continuous; this absorbs marker
   jitter, brief occlusion, and motion-blur dropouts. `fuse()` already aggregates
   cells over the onset window (a first-order version); a persistent filter is the
   upgrade.
3. **MMPose (RTMPose) is an Apache-2.0 accuracy upgrade over MediaPipe** — 21 hand
   keypoints, real-time, commercial-clean, generally more accurate/occlusion-aware.
   Start on MediaPipe (simplest deploy, already wired); swap to RTMPose if fingertip
   accuracy is the bottleneck. Both are clean; **OpenPose ($25k/yr) and MANO models
   stay out.**

Net: no change to the plan, just "multiple markers + temporal smoothing + MMPose as
the upgrade lane." Validation sources: markerless CV chord/tab (YOLO+Hourglass),
ArUco planar-pose ambiguity (arXiv 2509.17345 / DIVA moving-board study), MMPose
(Apache-2.0), end-to-end-vs-two-stage AV fusion (limited-data convergence).

**Added sources:** [TapToTab (2409.08618)](https://arxiv.org/abs/2409.08618) ·
[AV Fusion Piano TASLP 2024](https://dl.acm.org/doi/10.1109/TASLP.2024.3426303) /
[code](https://github.com/itec-hust/AMT_project) · [PianoMotion10M (2406.09326)](https://arxiv.org/html/2406.09326v1) ·
[SynthTab (2309.09085)](https://arxiv.org/pdf/2309.09085) ·
[Fretting-Transformer (2506.14223)](https://arxiv.org/pdf/2506.14223) ·
[weakly-supervised visual instrument action (1805.02031)](https://arxiv.org/pdf/1805.02031) ·
[Audio Matters Too — string mocap (2405.04963)](https://arxiv.org/pdf/2405.04963) ·
[MediaPipe hand-region accuracy (2405.03545)](https://arxiv.org/pdf/2405.03545).
