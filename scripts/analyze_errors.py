#!/usr/bin/env python3
"""Characterize prediction false positives from dumped note lists (offline).

First dump notes during eval (needs the full pipeline env)::

    python scripts/eval_dataset.py data/eval \
        --config data/eval/test1_best/best_combined.json --no-demucs \
        --dump-notes data/eval_results/notes

Then analyze (no ML deps; runs anywhere)::

    python scripts/analyze_errors.py data/eval_results/notes

For each clip it reports what fraction of the *false positives* are octave
overtones, fifths/12ths, same-pitch duplicates/sustain, other wrong notes, or
isolated spurious detections — so the dominant error type drives the next filter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aitabs.eval.error_analysis import (
    FalsePositiveBreakdown,
    MissBreakdown,
    categorize_false_negatives,
    categorize_false_positives,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="False-positive characterization")
    parser.add_argument("notes_dir", type=Path, help="Dir of *.notes.json from --dump-notes")
    parser.add_argument("--onset-tol", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None, help="JSON report path")
    args = parser.parse_args()

    files = sorted(args.notes_dir.glob("*.notes.json"))
    if not files:
        print(f"No *.notes.json under {args.notes_dir}", file=sys.stderr)
        sys.exit(1)

    fps: list[FalsePositiveBreakdown] = []
    misses: list[MissBreakdown] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        ref, pred = data["reference"], data["prediction"]
        cid = data.get("clip_id", f.stem)
        fps.append(categorize_false_positives(ref, pred, clip_id=cid, onset_tolerance_sec=args.onset_tol))
        misses.append(categorize_false_negatives(ref, pred, clip_id=cid, onset_tolerance_sec=args.onset_tol))

    fp_cats = ["octave", "fifth_or_twelfth", "duplicate_sustain", "other_concurrent", "isolated"]
    print(f"{'clip':26}{'FP':>6}{'oct':>6}{'5th':>6}{'dup':>6}{'other':>7}{'isol':>6}")
    for bd in sorted(fps, key=lambda b: -b.n_false):
        row = "".join(f"{getattr(bd, c):>6}" if c != 'other_concurrent' else f"{getattr(bd, c):>7}" for c in fp_cats)
        print(f"{bd.clip_id[:26]:26}{bd.n_false:>6}{row}")

    miss_cats = ["chord_inner", "masked_by_sustain", "octave_clash", "low_register", "isolated"]
    print(f"\n{'clip':26}{'MISS':>6}{'chord':>7}{'sustn':>7}{'oct':>6}{'bass':>6}{'isol':>6}")
    for bd in sorted(misses, key=lambda b: -b.n_missed):
        row = "".join(f"{getattr(bd, c):>7}" if c in ('chord_inner', 'masked_by_sustain') else f"{getattr(bd, c):>6}" for c in miss_cats)
        print(f"{bd.clip_id[:26]:26}{bd.n_missed:>6}{row}")

    # Aggregates (note-weighted).
    tot_fp = {c: sum(getattr(b, c) for b in fps) for c in fp_cats}
    n_fp = sum(b.n_false for b in fps)
    print("\n=== Aggregate false positives (over-detection) ===")
    for c in fp_cats:
        print(f"  {c:18} {tot_fp[c]:6d}  {(100.0 * tot_fp[c] / n_fp) if n_fp else 0:5.1f}%")
    print(f"  {'TOTAL FP':18} {n_fp:6d}")

    tot_miss = {c: sum(getattr(b, c) for b in misses) for c in miss_cats}
    n_miss = sum(b.n_missed for b in misses)
    print("\n=== Aggregate misses (recall loss) ===")
    for c in miss_cats:
        print(f"  {c:18} {tot_miss[c]:6d}  {(100.0 * tot_miss[c] / n_miss) if n_miss else 0:5.1f}%")
    print(f"  {'TOTAL MISSED':18} {n_miss:6d}")
    print("\n  chord_inner/sustain high -> polyphony recall (detection-side, not post-filtering)")
    print("  isolated high            -> threshold too high / quiet notes")

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "false_positives": {"clips": [b.as_dict() for b in fps], "aggregate": tot_fp},
                    "misses": {"clips": [b.as_dict() for b in misses], "aggregate": tot_miss},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
