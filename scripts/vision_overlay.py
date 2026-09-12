#!/usr/bin/env python3
"""Debug overlay: draw the predicted fretboard grid on a frame to check calibration.

Detects ArUco markers, recovers the canonical->image homography, projects the
12-TET fret grid (fret wires + strings + fret-number dots) onto the frame, and
overlays MediaPipe fretting-hand fingertips. Use this to eyeball that the grid
lands on the *real* strings/frets before trusting `vision_to_tab.py` (RESEARCH_
VISION.md M1). Purely diagnostic; needs a marked frame.

    python scripts/vision_overlay.py frame.jpg --markers rig_markers.json --out overlay.png
    python scripts/vision_overlay.py play.mp4  --markers rig_markers.json --out overlay.png  # 1st frame
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aitabs.pipeline.vision.fretboard_geometry import FretboardGeometry, apply_homography, homography


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw predicted fret grid on a frame")
    ap.add_argument("image", help="image or video (first frame used)")
    ap.add_argument("--markers", required=True, help="JSON {aruco_id: [[x,y]x4]} canonical")
    ap.add_argument("--aruco-dict", default="DICT_4X4_50")
    ap.add_argument("--scale-length", type=float, default=650.0)
    ap.add_argument("--string-spacing", type=float, default=10.0)
    ap.add_argument("--out", default="overlay.png")
    args = ap.parse_args()

    import cv2

    p = args.image
    frame = cv2.imread(p) if Path(p).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} else None
    if frame is None:
        cap = cv2.VideoCapture(p); ok, frame = cap.read(); cap.release()
        if not ok:
            sys.exit(f"could not read {p}")

    marker_canon = {int(k): np.asarray(v, float)
                    for k, v in json.loads(Path(args.markers).read_text()).items()}
    geom = FretboardGeometry(scale_length=args.scale_length, string_spacing=args.string_spacing)

    aruco = cv2.aruco
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, args.aruco_dict)),
                                   aruco.DetectorParameters())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        sys.exit("no ArUco markers detected — check lighting / dict / placement")
    canon, img = [], []
    seen = []
    for i, c in zip(ids.flatten(), corners):
        if int(i) in marker_canon:
            canon.extend(marker_canon[int(i)]); img.extend(c.reshape(4, 2)); seen.append(int(i))
    if len(canon) < 4:
        sys.exit(f"only markers {seen} matched --markers; need >=4 corners (>=2 markers)")
    H = homography(np.asarray(canon), np.asarray(img))
    print(f"markers used: {seen}  ({len(canon)} corners)")

    def draw(seg_list, color):
        for (a, b) in seg_list:
            pa = apply_homography(H, np.array([a]))[0]
            pb = apply_homography(H, np.array([b]))[0]
            cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), color, 1, cv2.LINE_AA)

    draw(geom.fret_line_segments(), (0, 255, 255))    # frets = yellow
    draw(geom.string_line_segments(), (255, 128, 0))  # strings = blue
    aruco.drawDetectedMarkers(frame, corners, ids)    # mark detected fiducials
    # fret-number dots at (string 0, fret f)
    for f in range(0, geom.num_frets + 1, 3):
        px = apply_homography(H, np.array([geom.cell_center(0, f)]))[0]
        cv2.putText(frame, str(f), tuple(px.astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(args.out, frame)
    print(f"wrote {args.out} — check the grid lands on the real strings/frets")


if __name__ == "__main__":
    main()
