#!/usr/bin/env python3
"""Diagnose rhythmic structure from note onsets (no ML deps; runs anywhere).

When a transcription renders as "nearly all eighth notes" but the ground truth
mixes eighths/sixteenths/dotted figures, this tells you *why* from the onset
data instead of the rendered tab:

* **NOTATION-FLATTEN** — the predicted onsets carry sub-eighth inter-onset
  intervals, so the variety is in the data and notation collapsed it (suspect
  ``snap_dominant_grid`` / ``legato`` / too-coarse a grid).
* **DETECTOR/RECALL** — the predicted onsets are themselves ~eighth-spaced; the
  fast notes were never detected, so no notation change recovers them.

Input is a dumped-notes JSON (``--dump-notes`` output: ``{reference, prediction}``)
or any JSON that is a list of note dicts / has a ``notes`` key. Provide ``--bpm``
(the tempo the tab is notated at; default 96)::

    python scripts/analyze_rhythm.py data/eval_results/notes/vidtest1.notes.json --bpm 96
    python scripts/analyze_rhythm.py path/to/pred_notes.json --bpm 96   # no reference
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# Load the (dependency-free) diagnostic module directly by path so this tool runs
# in a minimal env without triggering aitabs.eval.__init__ (which imports torch /
# pretty_midi / librosa). Falls back to the normal import when those are present.
_MOD = Path(__file__).resolve().parent.parent / "aitabs" / "eval" / "rhythm_structure.py"
_spec = importlib.util.spec_from_file_location("aitabs_rhythm_structure", _MOD)
_rs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _rs  # required so @dataclass can resolve its own module
_spec.loader.exec_module(_rs)
diagnose = _rs.diagnose


def _load_notes(obj) -> tuple[list[dict], list[dict] | None]:
    """Return (prediction, reference|None) from a flexible JSON shape."""
    if isinstance(obj, list):
        return obj, None
    if isinstance(obj, dict):
        if "prediction" in obj:
            return obj["prediction"], obj.get("reference")
        if "notes" in obj:
            return obj["notes"], obj.get("reference")
    raise ValueError("Unrecognized JSON: expected a list, or a dict with "
                     "'prediction'/'notes'.")


def _analyze_file(path: Path, bpm: float, chord_window: float) -> None:
    obj = json.loads(path.read_text())
    pred, ref = _load_notes(obj)
    verdict, p, r = diagnose(pred, ref, bpm, chord_window_sec=chord_window)
    print(f"\n=== {path.name}  (bpm={bpm:g}) ===")
    if r is not None:
        print(f"  reference : {r.summary()}")
    print(f"  prediction: {p.summary()}")
    print(f"  → {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="dumped-notes JSON file or a directory of them")
    ap.add_argument("--bpm", type=float, default=96.0,
                    help="tempo the tab is notated at (default 96)")
    ap.add_argument("--chord-window", type=float, default=0.05,
                    help="seconds within which onsets are one chord (default 0.05)")
    args = ap.parse_args()

    if args.path.is_dir():
        files = sorted(args.path.glob("*.json"))
        if not files:
            print(f"No .json files in {args.path}", file=sys.stderr)
            sys.exit(1)
        for f in files:
            _analyze_file(f, args.bpm, args.chord_window)
    elif args.path.is_file():
        _analyze_file(args.path, args.bpm, args.chord_window)
    else:
        hint = args.path
        if "path/to" in str(args.path):
            hint = "data/eval_results/notes/vidtest1.notes.json"
        print(f"Not found: {args.path}", file=sys.stderr)
        if Path(hint).is_file() and hint != str(args.path):
            print(f"Did you mean: {hint}", file=sys.stderr)
            print(f"  python scripts/analyze_rhythm.py {hint} --bpm {args.bpm:g}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
