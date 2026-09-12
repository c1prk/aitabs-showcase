#!/usr/bin/env python3
"""Compare two GP5 files directly (reference vs prediction export).

Usage::

    python scripts/compare_gp5.py reference.gp5 predicted.gp5
"""

from __future__ import annotations

import argparse
import sys

from aitabs.eval.gp5_import import load_gp5_notes
from aitabs.eval.metrics import compare_note_lists


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reference vs predicted GP5")
    parser.add_argument("reference_gp5")
    parser.add_argument("predicted_gp5")
    parser.add_argument("--onset-tol", type=float, default=0.05)
    args = parser.parse_args()

    ref_notes, ref_bpm, _ = load_gp5_notes(args.reference_gp5)
    pred_notes, pred_bpm, _ = load_gp5_notes(args.predicted_gp5)

    scores = compare_note_lists(
        ref_notes,
        pred_notes,
        clip_id="gp5_compare",
        onset_tolerance_sec=args.onset_tol,
        reference_bpm=ref_bpm,
        estimated_bpm=pred_bpm,
    )
    print(scores.summary_line())
    if scores.tab_f1 < 0.5:
        sys.exit(1)


if __name__ == "__main__":
    main()
