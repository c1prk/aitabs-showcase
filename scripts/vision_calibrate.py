#!/usr/bin/env python3
"""One-time marker calibration by clicking 4 known fret/string points.

Instead of hand-measuring where each ArUco marker sits, click four landmarks on
one clear frame; this recovers the canonical->image homography for that frame,
then **back-projects every detected marker's corners into the canonical fretboard
frame** and writes ``rig_markers.json``. After that, ``vision_to_tab.py`` /
``vision_overlay.py`` use the markers to re-solve the homography every frame (so it
survives the guitar moving) — you only calibrate once.

Click order (a window opens; click these 4 points, in this order):
    1. low-E string  at the NUT (fret 0)
    2. high-e string at the NUT (fret 0)
    3. low-E string  at the --cal-fret wire (default 12th)
    4. high-e string at the --cal-fret wire

Pick a frame where the guitar is flat-on and the whole neck (nut..cal-fret) plus
the markers are clearly visible.

    python scripts/vision_calibrate.py vidtest1.mp4 --out rig_markers.json
    python scripts/vision_calibrate.py vidtest1.mp4 --frame 90   # use frame 90
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aitabs.pipeline.vision.fretboard_geometry import (
    FretboardGeometry, homography, image_to_canonical,
)

_PROMPTS = [
    "1/4: click LOW-E string at the NUT (fret 0)",
    "2/4: click HIGH-e string at the NUT (fret 0)",
    "3/4: click LOW-E string at the CAL-FRET wire",
    "4/4: click HIGH-e string at the CAL-FRET wire",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Click-4 marker calibration -> rig_markers.json")
    ap.add_argument("video")
    ap.add_argument("--out", default="rig_markers.json")
    ap.add_argument("--frame", type=int, default=0, help="which video frame to calibrate on")
    ap.add_argument("--aruco-dict", default="DICT_4X4_50")
    ap.add_argument("--scale-length", type=float, default=650.0)
    ap.add_argument("--string-spacing", type=float, default=10.0)
    ap.add_argument("--cal-fret", type=int, default=12, help="fret wire used for clicks 3 & 4")
    args = ap.parse_args()

    import cv2

    p = args.video
    if Path(p).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(p)
    else:
        cap = cv2.VideoCapture(p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read(); cap.release()
        if not ok:
            sys.exit(f"could not read frame {args.frame} of {p}")

    geom = FretboardGeometry(scale_length=args.scale_length, string_spacing=args.string_spacing)
    # canonical positions of the 4 clicked landmarks (fret WIRE x, string y)
    lo, hi = geom.string_y(0), geom.string_y(geom.num_strings - 1)
    x_nut, x_cal = geom.fret_line_x(0), geom.fret_line_x(args.cal_fret)
    canon_click = np.array([[x_nut, lo], [x_nut, hi], [x_cal, lo], [x_cal, hi]], float)

    clicks: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    win = "calibrate — click the 4 points in order (r=reset, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = frame.copy()
        for i, c in enumerate(clicks):
            cv2.circle(disp, c, 6, (0, 255, 0), -1)
            cv2.putText(disp, str(i + 1), (c[0] + 8, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        msg = _PROMPTS[len(clicks)] if len(clicks) < 4 else "done — press ENTER to save, r=reset"
        cv2.putText(disp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            cv2.destroyAllWindows(); sys.exit("cancelled")
        if k == ord("r"):
            clicks.clear()
        if k in (13, 10) and len(clicks) == 4:
            break
    cv2.destroyAllWindows()

    # homography canonical->image from the 4 clicks, then back-project markers
    H = homography(canon_click, np.array(clicks, float))
    aruco = cv2.aruco
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, args.aruco_dict)),
                                   aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        sys.exit("no ArUco markers detected in the calibration frame")
    rig = {}
    for i, c in zip(ids.flatten(), corners):
        canon = image_to_canonical(H, c.reshape(4, 2))
        rig[str(int(i))] = np.round(canon, 2).tolist()
    Path(args.out).write_text(json.dumps(rig, indent=2))
    print(f"detected markers: {sorted(int(i) for i in ids.flatten())}")
    print(f"wrote {args.out} ({len(rig)} markers in canonical fretboard coords)")
    print("next: python scripts/vision_overlay.py <frame> --markers", args.out, "  (verify the grid)")


if __name__ == "__main__":
    main()
