#!/usr/bin/env python3
"""Fine-tune BasicPitch on GuitarSet solo recordings.

Loads the ICASSP-2022 SavedModel, trains on the GuitarSet manifest produced by
scripts/build_training_data.py, and saves a fine-tuned SavedModel compatible
with the existing eval harness.

Usage::

    python scripts/finetune_basicpitch.py
    python scripts/finetune_basicpitch.py --epochs 20 --lr 5e-5
    python scripts/finetune_basicpitch.py --max-clips 10  # smoke test

After training, evaluate with::

    python scripts/eval_dataset.py data/eval/guitarset --guitarset \\
        --detector finetuned --model-path data/models/basicpitch_guitarset_ft
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from aitabs.pipeline.audio.finetune import finetune


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune BasicPitch on GuitarSet solo recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/training/guitarset_manifest.json"),
        help="GuitarSet training manifest (default: data/training/guitarset_manifest.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/models"),
        help="Output dir for fine-tuned model + history (default: data/models)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Adam learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Windows per gradient step (default: 4)"
    )
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable audio augmentation"
    )
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Train all layers (default: freeze first conv)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limit number of training clips (smoke test)",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(
            f"Manifest not found: {args.manifest}\n"
            "Run:  python scripts/build_training_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fine-tuning BasicPitch on {args.manifest}", flush=True)
    print(
        f"  epochs={args.epochs}  lr={args.lr}  batch_size={args.batch_size}"
        f"  augment={not args.no_augment}  freeze_first_conv={not args.no_freeze}",
        flush=True,
    )
    if args.max_clips:
        print(f"  max_clips={args.max_clips} (smoke test)", flush=True)

    history = finetune(
        args.manifest,
        args.out,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        freeze_first_conv=not args.no_freeze,
        augment=not args.no_augment,
        max_clips=args.max_clips,
    )

    losses = history["loss"]
    print(f"\nFinal loss : {losses[-1]:.4f}  (started {losses[0]:.4f})")
    print(f"Model path : {history['model_path']}")
    print(
        f"\nNext:  python scripts/eval_dataset.py data/eval/guitarset --guitarset"
        f" --detector finetuned --model-path {history['model_path']}"
    )


if __name__ == "__main__":
    main()
