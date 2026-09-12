#!/usr/bin/env python3
"""Diff a detector's output against a GP5 reference, note-by-note.

Converts the GP5 to GuitarSet-style note events (time, pitch, string, fret) via
``load_gp5_notes``, runs the chosen detector on the audio, tempo-aligns the two
(global linear warp, since a notated GP5 is constant-tempo but the audio is a
human performance), then prints an order-preserving alignment marking:

  MATCH  — reference note detected (shows onset drift)
  MISS   — reference note NOT detected (false negative)
  GHOST  — detected note with no reference (false positive)

Usage::

    python scripts/gp5_diff.py --gp5 data/eval/reference/vidtest1.gp5 \
        --audio data/eval/audio/vidtest1.mp3 \
        --detector kong --model-path eval/kong_ft_step4000.pth \
        --onset 0.4 --frame 0.2 --conf 0.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(m: int) -> str:
    return f"{_NAMES[m % 12]}{m // 12 - 1}"


def match_greedy(ref, det, *, time_window: float):
    """Greedy time+pitch matching (chord-correct): each ref note matches the
    nearest unused detected note of equal pitch within the warped time window.
    Returns an aligned, time-sorted list of (ref|None, det|None) rows.
    """
    used = [False] * len(det)
    ref_pair = [None] * len(ref)
    for ri, r in enumerate(ref):
        best, bj = time_window + 1e9, None
        for dj, d in enumerate(det):
            if used[dj] or d["pitch_midi"] != r["pitch_midi"]:
                continue
            dt = abs(d["_w"] - r["start"])
            if dt <= time_window and dt < best:
                best, bj = dt, dj
        if bj is not None:
            used[bj] = True
            ref_pair[ri] = bj
    rows = []
    for ri, r in enumerate(ref):
        rows.append((r["start"], r, det[ref_pair[ri]] if ref_pair[ri] is not None else None))
    for dj, d in enumerate(det):
        if not used[dj]:
            rows.append((d["_w"], None, d))
    rows.sort(key=lambda x: x[0])
    return [(r, d) for _, r, d in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Note-by-note GP5 reference vs detection diff")
    ap.add_argument("--gp5", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--detector", default="kong")
    ap.add_argument("--model-path", default="eval/kong_ft_step4000.pth")
    ap.add_argument("--onset", type=float, default=0.4)
    ap.add_argument("--frame", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--time-window", type=float, default=0.30, help="max warped onset gap to call a match")
    args = ap.parse_args()

    from aitabs.eval.gp5_import import load_gp5_notes
    from aitabs.eval.metrics import estimate_time_warp
    from aitabs.pipeline.audio.pitch import get_detector

    ref_raw, bpm, _ = load_gp5_notes(args.gp5)
    ref = [dict(n) for n in ref_raw]
    det_raw = get_detector(args.detector, model_path=args.model_path).detect(
        args.audio, onset_threshold=args.onset, frame_threshold=args.frame,
        confidence_threshold=args.conf, minimum_note_length_ms=58.0,
        minimum_frequency_hz=82.0, maximum_frequency_hz=1400.0, melodia_trick=False)
    det = [{**dict(n), "string": n.get("string", -1), "fret": n.get("fret", -1)} for n in det_raw]

    # global tempo warp of detection onto the notated reference timeline
    scale, offset = estimate_time_warp(
        [dict(n, string=-1, fret=-1) for n in ref],
        [dict(n, string=-1, fret=-1) for n in det],
    )
    for n in det:
        n["_w"] = n["start"] * scale + offset
    ref.sort(key=lambda n: n["start"])
    det.sort(key=lambda n: n["_w"])

    aligned = match_greedy(ref, det, time_window=args.time_window)

    matches = [(r, d) for r, d in aligned if r and d]
    misses = [r for r, d in aligned if r and not d]
    ghosts = [d for r, d in aligned if d and not r]
    drifts = [d["_w"] - r["start"] for r, d in matches]

    import statistics as st
    print(f"reference: {len(ref)} notes ({args.gp5})")
    print(f"detected:  {len(det)} notes   notated tempo={bpm:.0f}bpm  "
          f"performed~{bpm/scale:.1f}bpm (warp scale={scale:.3f}, offset={offset:+.2f}s)")
    p = len(matches) / len(det) if det else 0
    r = len(matches) / len(ref) if ref else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"MATCH={len(matches)}  MISS={len(misses)}  GHOST={len(ghosts)}   "
          f"pitch P={p:.3f} R={r:.3f} F1={f1:.3f}")
    if drifts:
        print(f"onset drift of matched notes (warped): median={st.median(drifts)*1000:+.0f}ms  "
              f"mean|drift|={st.mean(abs(x) for x in drifts)*1000:.0f}ms  "
              f"max|drift|={max(abs(x) for x in drifts)*1000:.0f}ms")
    from collections import Counter
    mc = Counter(note_name(n["pitch_midi"]) for n in misses)
    gc = Counter(note_name(n["pitch_midi"]) for n in ghosts)
    hi = sum(1 for n in misses if n["pitch_midi"] >= 64)  # >= E4
    dyad = sum(1 for n in misses if any(abs(o["start"] - n["start"]) < 0.05 and o is not n for o in ref))
    print(f"MISS by pitch:  {dict(mc.most_common())}")
    print(f"  ({hi}/{len(misses)} misses are high notes >=E4; {dyad}/{len(misses)} occur in a dyad/chord)")
    print(f"GHOST by pitch: {dict(gc.most_common())}")
    print("\n  REF (time  note  s/f)          DET (warped-time  note  s/f)        status")
    print("  " + "-" * 74)
    for r_, d_ in aligned:
        if r_ and d_:
            drift = (d_["_w"] - r_["start"]) * 1000
            flag = "MATCH" + (f"  drift={drift:+.0f}ms" if abs(drift) > 60 else "")
            print(f"  {r_['start']:6.2f} {note_name(r_['pitch_midi']):<4} s{r_['string']}f{r_['fret']:<2}"
                  f"      {d_['_w']:6.2f} {note_name(d_['pitch_midi']):<4}"
                  f"                {flag}")
        elif r_:
            print(f"  {r_['start']:6.2f} {note_name(r_['pitch_midi']):<4} s{r_['string']}f{r_['fret']:<2}"
                  f"      {'---':>6}                         *** MISS ***")
        else:
            print(f"  {'---':>6}                     {d_['_w']:6.2f} {note_name(d_['pitch_midi']):<4}"
                  f"                *** GHOST ***")


if __name__ == "__main__":
    main()
