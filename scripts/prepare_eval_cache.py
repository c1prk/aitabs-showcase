#!/usr/bin/env python3
"""Precompute filtered notes + tempo per eval clip (for fast fingering tuning).

Usage::

    python scripts/prepare_eval_cache.py data/eval --cache-dir data/eval/cache
    python scripts/prepare_eval_cache.py data/eval --no-demucs --config best_pipeline.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aitabs.eval.cache import cache_path_for_clip
from aitabs.eval.config import PipelineConfig
from aitabs.eval.dataset import discover_pairs
from aitabs.eval.gp5_import import load_gp5_notes
from aitabs.eval.run import audio_stage_to_cache, run_audio_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache audio-stage outputs for eval clips")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Default: <data_dir>/cache",
    )
    parser.add_argument("--config", type=Path, default=None, help="PipelineConfig JSON")
    parser.add_argument("--no-demucs", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    cache_dir = (args.cache_dir or data_dir / "cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig.load_json(args.config) if args.config else PipelineConfig()
    if args.no_demucs:
        cfg.use_demucs = False

    pairs = discover_pairs(data_dir)
    if not pairs:
        print(f"No paired clips under {data_dir}", file=sys.stderr)
        sys.exit(1)

    for pair in pairs:
        out_path = cache_path_for_clip(cache_dir, pair.clip_id)
        if out_path.is_file():
            print(f"skip {pair.clip_id} (cache exists)")
            continue

        ref_bpm = None
        if cfg.use_reference_bpm:
            _, ref_bpm, _ = load_gp5_notes(pair.reference_gp5_path)

        print(f"cache {pair.clip_id} …")
        audio = run_audio_stage(
            pair.audio_path,
            cfg,
            reference_bpm=ref_bpm,
            stem_path=pair.stem_path,
            work_dir=args.work_dir,
        )
        entry = audio_stage_to_cache(
            audio,
            pair.clip_id,
            audio_path=str(pair.audio_path),
        )
        entry.save(out_path)
        print(f"  → {out_path} ({len(entry.filtered_notes)} notes)")

    print(f"Done. Cache dir: {cache_dir}")


if __name__ == "__main__":
    main()
