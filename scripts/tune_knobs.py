#!/usr/bin/env python3
"""Grid-search pipeline knobs against reference GP5 + audio pairs.

Uses a small random/grid search over key thresholds. Demucs is slow — prefer
pre-separated stems in ``data/eval/stems/<clip_id>.wav`` or ``--no-demucs`` for
quick iteration on solo guitar.

Usage::

    python scripts/tune_knobs.py data/eval --trials 20 --no-demucs
    python scripts/tune_knobs.py data/eval --trials 30 --val-ratio 0.3
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

from aitabs.eval.config import PipelineConfig
from aitabs.eval.dataset import discover_pairs
from aitabs.eval.evaluate import evaluate_pair


def _grid_candidates() -> list[PipelineConfig]:
    """Small grid of knob combinations."""
    conf = [0.35, 0.42, 0.5, 0.55]
    onset = [0.4, 0.5, 0.6]
    merge = [0.10, 0.15, 0.20]
    snap = [False, True]
    chord_grid = [0, 1]
    snap_grid = [False, True]
    librosa_snap = [False, True]

    configs: list[PipelineConfig] = []
    for c, o, m, sg, cg, stg, ls in itertools.product(
        conf, onset, merge, snap_grid, chord_grid, snap, librosa_snap
    ):
        configs.append(
            PipelineConfig(
                confidence_threshold=c,
                onset_threshold=o,
                pitch_merge_sec=m,
                snap_to_grid=sg,
                chord_grid_ticks=cg,
                snap_to_librosa_onsets=ls,
                use_reference_bpm=True,
                export_gp5=False,
            )
        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Search pipeline knobs vs reference GP5")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--trials", type=int, default=24, help="Number of configs to try")
    parser.add_argument("--val-ratio", type=float, default=0.25, help="Hold-out fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-demucs", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: data_dir/../tune_results",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Try entire grid (slow; hundreds of combos)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    pairs = discover_pairs(data_dir)
    if len(pairs) < 2:
        print("Need at least 2 paired clips to tune (train/val split).", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    ids = [p.clip_id for p in pairs]
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_ratio))
    val_ids = set(ids[:n_val])
    train_pairs = [p for p in pairs if p.clip_id not in val_ids]
    val_pairs = [p for p in pairs if p.clip_id in val_ids]

    grid = _grid_candidates()
    if args.full_grid:
        trial_configs = grid
    else:
        rng.shuffle(grid)
        trial_configs = grid[: min(args.trials, len(grid))]

    out_dir = args.output_dir or (data_dir.parent / "tune_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.no_demucs:
        for cfg in trial_configs:
            cfg.use_demucs = False

    print(f"Train clips: {len(train_pairs)}, val clips: {len(val_pairs)}")
    print(f"Trying {len(trial_configs)} config(s)\n")

    best_val_tab = -1.0
    best_config: PipelineConfig | None = None
    leaderboard: list[dict] = []

    for trial_i, config in enumerate(trial_configs):
        train_scores = []
        for pair in train_pairs:
            try:
                train_scores.append(
                    evaluate_pair(
                        pair,
                        config,
                        work_dir=run_dir / f"t{trial_i}" / pair.clip_id,
                    )
                )
            except Exception as exc:
                print(f"  trial {trial_i} {pair.clip_id} failed: {exc}", file=sys.stderr)

        if not train_scores:
            continue

        train_tab_f1 = sum(s.tab_f1 for s in train_scores) / len(train_scores)

        val_scores = []
        for pair in val_pairs:
            try:
                val_scores.append(
                    evaluate_pair(
                        pair,
                        config,
                        work_dir=run_dir / f"t{trial_i}_val" / pair.clip_id,
                    )
                )
            except Exception:
                pass

        val_tab_f1 = (
            sum(s.tab_f1 for s in val_scores) / len(val_scores) if val_scores else 0.0
        )
        val_pitch_f1 = (
            sum(s.pitch_f1 for s in val_scores) / len(val_scores) if val_scores else 0.0
        )

        entry = {
            "trial": trial_i,
            "train_tab_f1": train_tab_f1,
            "val_tab_f1": val_tab_f1,
            "val_pitch_f1": val_pitch_f1,
            "config": config.to_dict(),
        }
        leaderboard.append(entry)
        print(
            f"trial {trial_i:3d}  train tab F1={train_tab_f1:.3f}  "
            f"val tab F1={val_tab_f1:.3f}  conf={config.confidence_threshold}"
        )

        if val_tab_f1 > best_val_tab:
            best_val_tab = val_tab_f1
            best_config = config

    leaderboard.sort(key=lambda x: x["val_tab_f1"], reverse=True)
    (out_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2),
        encoding="utf-8",
    )

    if best_config is None:
        print("No successful trials.", file=sys.stderr)
        sys.exit(1)

    best_path = out_dir / "best_config.json"
    best_config.save_json(best_path)
    print(f"\nBest val tab F1={best_val_tab:.3f}")
    print(f"Saved {best_path}")
    print("\nRe-evaluate on all clips:")
    print(f"  python scripts/eval_dataset.py {data_dir} --config {best_path}")


if __name__ == "__main__":
    main()
