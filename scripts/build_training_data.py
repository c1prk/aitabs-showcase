#!/usr/bin/env python3
"""Build a GuitarSet training manifest (solo mic audio + ground-truth notes).

Walks a prepared GuitarSet directory for *_solo* JAMS + matching *_mic.wav pairs
and writes a manifest JSON ready for detector fine-tuning.

Only solo clips are included (comp/chord takes are excluded).

Usage::

    # After running scripts/prepare_guitarset.py --download
    python scripts/build_training_data.py

    # Custom paths
    python scripts/build_training_data.py \\
        --guitarset-dir data/eval/guitarset \\
        --out data/training

Output (data/training/guitarset_manifest.json)::

    {
      "clips": 60,
      "total_notes": 12345,
      "mean_notes_per_clip": 205.75,
      "pairs": [
        {
          "clip_id": "00_BN1-129-Eb_solo",
          "audio_path": "/abs/path/to/00_BN1-129-Eb_solo_mic.wav",
          "bpm": 129.0,
          "notes": [{"start": 0.1, "end": 0.4, "pitch_midi": 52, "confidence": 1.0}, ...]
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aitabs.eval.guitarset_import import load_guitarset_training_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GuitarSet solo training manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--guitarset-dir",
        type=Path,
        default=Path("data/eval/guitarset"),
        help="Prepared GuitarSet dir with annotation/ + audio/ (default: data/eval/guitarset)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/training"),
        help="Output directory for the manifest (default: data/training)",
    )
    args = parser.parse_args()

    if not args.guitarset_dir.is_dir():
        print(
            f"GuitarSet dir not found: {args.guitarset_dir}\n"
            "Run scripts/prepare_guitarset.py --download first.",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs = load_guitarset_training_pairs(args.guitarset_dir)
    if not pairs:
        print(
            f"No solo pairs found under {args.guitarset_dir}.\n"
            "Check that annotation/*_solo*.jams and audio/*_solo*_mic.wav both exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "guitarset_manifest.json"

    total_notes = sum(len(p["notes"]) for p in pairs)
    manifest = {
        "clips": len(pairs),
        "total_notes": total_notes,
        "mean_notes_per_clip": round(total_notes / len(pairs), 2),
        "pairs": pairs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(pairs)} solo clip(s) -> {manifest_path}")
    print(f"Total notes : {total_notes}")
    print(f"Mean / clip : {manifest['mean_notes_per_clip']:.1f}")
    print(
        "\nNext: run A/B eval on detectors, then use this manifest for fine-tuning.\n"
        "  python scripts/eval_dataset.py data/eval/guitarset --guitarset --detector basic_pitch\n"
        "  python scripts/eval_dataset.py data/eval/guitarset --guitarset --detector recall_first"
    )


if __name__ == "__main__":
    main()
