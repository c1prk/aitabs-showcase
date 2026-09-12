#!/usr/bin/env python3
"""Search fingering knobs against cached audio + reference GP5.

Run ``prepare_eval_cache.py`` first so BasicPitch/Demucs are not repeated.

Usage::

    python scripts/prepare_eval_cache.py data/eval --no-demucs
    python scripts/tune_fingering.py data/eval --cache-dir data/eval/cache --trials 40
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

from aitabs.eval.cache import AudioCacheEntry, cache_path_for_clip
from aitabs.eval.config import PipelineConfig
from aitabs.eval.dataset import discover_pairs
from aitabs.eval.evaluate import evaluate_cached_pair
from aitabs.pipeline.mapping.fingering_config import FingeringConfig


def _grid_candidates() -> list[FingeringConfig]:
    mappers = ["viterbi", "greedy"]
    string_cost = [0.75, 1.0, 1.25]
    fret_cost = [0.15, 0.25, 0.35]
    max_span = [3, 4, 5]
    span_base = [3.0, 5.0, 7.0]
    low_fret = [0.05, 0.10, 0.15]

    configs: list[FingeringConfig] = []
    for mapper, sc, fc, ms, sb, lf in itertools.product(
        mappers, string_cost, fret_cost, max_span, span_base, low_fret
    ):
        configs.append(
            FingeringConfig(
                mapper=mapper,
                string_cost=sc,
                fret_cost=fc,
                max_fret_span=ms,
                span_base_penalty=sb,
                low_fret_penalty=lf,
            )
        )
    return configs


def _mean_tab_f1(scores) -> float:
    if not scores:
        return 0.0
    return sum(s.tab_f1 for s in scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune fingering vs reference GP5 (cached audio)")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--pipeline-config", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--onset-tol", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <data_dir>/fingering_tune_results",
    )
    parser.add_argument("--full-grid", action="store_true", help="Exhaustive grid (slow)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    cache_dir = (args.cache_dir or data_dir / "cache").resolve()
    pairs = discover_pairs(data_dir)
    if len(pairs) < 2:
        print("Need at least 2 paired clips.", file=sys.stderr)
        sys.exit(1)

    pipeline_cfg = (
        PipelineConfig.load_json(args.pipeline_config)
        if args.pipeline_config
        else PipelineConfig()
    )

    cached: list[tuple] = []
    for pair in pairs:
        path = cache_path_for_clip(cache_dir, pair.clip_id)
        if not path.is_file():
            print(f"Missing cache for {pair.clip_id}: run prepare_eval_cache.py", file=sys.stderr)
            sys.exit(1)
        cached.append((pair, AudioCacheEntry.load(path)))

    rng = random.Random(args.seed)
    indices = list(range(len(cached)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_ratio))
    val_idx = set(indices[:n_val])

    train_items = [cached[i] for i in range(len(cached)) if i not in val_idx]
    val_items = [cached[i] for i in val_idx]

    grid = _grid_candidates()
    if args.full_grid:
        candidates = grid
    else:
        candidates = rng.sample(grid, min(args.trials, len(grid)))

    out_dir = (args.output_dir or data_dir / "fingering_tune_results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_train_f1 = -1.0
    best_cfg: FingeringConfig | None = None
    results: list[dict] = []

    for i, finger_cfg in enumerate(candidates):
        train_scores = [
            evaluate_cached_pair(
                pair,
                cache,
                pipeline_cfg,
                fingering_config=finger_cfg,
                onset_tolerance_sec=args.onset_tol,
            )
            for pair, cache in train_items
        ]
        train_f1 = _mean_tab_f1(train_scores)
        results.append({"trial": i, "train_tab_f1": train_f1, **finger_cfg.to_dict()})
        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
            best_cfg = finger_cfg
        print(f"[{i + 1}/{len(candidates)}] train tab F1={train_f1:.3f} mapper={finger_cfg.mapper}")

    if best_cfg is None:
        print("No trials completed.", file=sys.stderr)
        sys.exit(1)

    val_scores = [
        evaluate_cached_pair(
            pair,
            cache,
            pipeline_cfg,
            fingering_config=best_cfg,
            onset_tolerance_sec=args.onset_tol,
        )
        for pair, cache in val_items
    ]
    val_f1 = _mean_tab_f1(val_scores)
    val_pitch_f1 = sum(s.pitch_f1 for s in val_scores) / len(val_scores)

    best_path = out_dir / "best_fingering.json"
    best_cfg.save_json(best_path)
    summary = {
        "best_train_tab_f1": best_train_f1,
        "val_tab_f1": val_f1,
        "val_pitch_f1": val_pitch_f1,
        "val_clips": [s.summary_line() for s in val_scores],
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_trials": len(candidates),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "trials.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nBest config → {best_path}")
    print(f"  train tab F1: {best_train_f1:.3f}")
    print(f"  val tab F1:   {val_f1:.3f}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
