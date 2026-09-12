#!/usr/bin/env python3
"""Stage-1 vision fusion: marked guitar video -> string/fret via geometry + audio.

Pipeline (RESEARCH_VISION.md §2-4, license-clean: OpenCV ArUco + MediaPipe, both
Apache-2.0 — no AGPL YOLO, no MANO):

  per frame:  ArUco markers -> canonical->image homography -> MediaPipe fretting-hand
              fingertips -> (string, fret) cells   [fretboard_geometry, tested]
  then:       audio detector -> note timeline (pitch + onset, ~0.92 F1)
  then:       fuse() -> vision picks the string among each pitch's playable
              positions; lowest-fret fallback when a finger wasn't seen [fusion, tested]

The geometry + fusion core is unit-tested; this script is the video I/O glue and
needs a real marked clip to run. **Marker calibration** (`--markers`) is a JSON
mapping ArUco id -> its four corner (x, y) positions in the *canonical fretboard
frame* (nut=x0, along-neck +x, across-strings +y; any consistent unit). Measure it
once for your rig; place markers where neither hand occludes them (see §4).

    python scripts/vision_to_tab.py play.mp4 --markers rig_markers.json \
        --model-path eval/kong2_ft_step4000.pth --out play.musicxml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aitabs.pipeline.vision.fretboard_geometry import (
    FretboardGeometry, homography, image_point_to_cell,
)
from aitabs.pipeline.fusion.fusion import fuse, source_distribution

# MediaPipe fingertip + distal-joint landmark ids (the fretting fingers).
_FINGERTIPS = (8, 12, 16, 20)   # index, middle, ring, pinky tips


def _detect_markers(gray, detector):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}


def _frame_homography(marker_px, marker_canon):
    """Homography canonical->image from all visible markers' 4 corners."""
    canon, img = [], []
    for mid, px in marker_px.items():
        if mid in marker_canon:
            canon.extend(marker_canon[mid])
            img.extend(px)
    if len(canon) < 4:
        return None
    return homography(np.asarray(canon), np.asarray(img))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1 vision fusion (marked video -> tab)")
    ap.add_argument("video")
    ap.add_argument("--markers", required=True, help="JSON: {aruco_id: [[x,y]x4]} in canonical frame")
    ap.add_argument("--model-path", default="eval/kong2_ft_step4000.pth")
    ap.add_argument("--detector", default="ensemble")
    ap.add_argument("--aruco-dict", default="DICT_4X4_50")
    ap.add_argument("--scale-length", type=float, default=650.0)
    ap.add_argument("--string-spacing", type=float, default=10.0)
    ap.add_argument("--hand", default="Left", help="fretting hand ('Left' for a right-handed player)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2
    import mediapipe as mp

    marker_canon = {int(k): np.asarray(v, dtype=np.float64)
                    for k, v in json.loads(Path(args.markers).read_text()).items()}
    geom = FretboardGeometry(scale_length=args.scale_length, string_spacing=args.string_spacing)

    aruco = cv2.aruco
    a_dict = aruco.getPredefinedDictionary(getattr(aruco, args.aruco_dict))
    detector = aruco.ArucoDetector(a_dict, aruco.DetectorParameters())
    hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                                     min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vision_frames, fi, n_homog = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = _frame_homography(_detect_markers(gray, detector), marker_canon)
        cells = []
        if H is not None:
            n_homog += 1
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w = frame.shape[:2]
            if res.multi_hand_landmarks:
                for lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    if handed.classification[0].label != args.hand:
                        continue
                    for tip in _FINGERTIPS:
                        lm = lms.landmark[tip]
                        cell = image_point_to_cell(np.array([lm.x * w, lm.y * h]), H, geom)
                        if cell is not None:
                            cells.append(cell)
        vision_frames.append({"timestamp": t, "cells": cells})
        fi += 1
    cap.release(); hands.close()
    print(f"frames={fi}  homography-locked={n_homog} ({100*n_homog/max(fi,1):.0f}%)  "
          f"frames-with-fingers={sum(1 for f in vision_frames if f['cells'])}", flush=True)

    # audio note timeline
    import librosa, soundfile as sf, tempfile
    from aitabs.pipeline.audio.pitch import get_detector
    y, sr = librosa.load(args.video, sr=16000, mono=True)
    tmp = Path(tempfile.gettempdir()) / "v2t.wav"; sf.write(str(tmp), y, sr)
    det = get_detector(args.detector, model_path=args.model_path)
    notes = det.detect(str(tmp), onset_threshold=0.4, frame_threshold=0.2, confidence_threshold=0.2,
                       minimum_note_length_ms=58.0, minimum_frequency_hz=82.0,
                       maximum_frequency_hz=1400.0, melodia_trick=False)
    print(f"audio notes={len(notes)}", flush=True)

    fused = fuse([dict(n) for n in notes], vision_frames)
    dist = source_distribution(fused)
    print(f"fused notes={len(fused)}  source distribution={ {k: round(v,3) for k,v in dist.items()} }")
    print("  ^ vision-resolved fraction = how much the camera lifted string/fret over audio-only")
    if args.out:
        Path(args.out).write_text(json.dumps(fused, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
