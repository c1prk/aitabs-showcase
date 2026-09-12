#!/usr/bin/env python3
"""Optimize pipeline + fingering for one eval clip (no train/val split).

Round 1 (default): precision-first coordinate search from defaults.
Round 2 (--from-config): recall + BasicPitch knobs from a prior best JSON.

Usage::

    python scripts/tune_single_clip.py data/eval --clip-id test1 --no-demucs
    python scripts/tune_single_clip.py data/eval --clip-id test1 --no-demucs \\
        --from-config data/eval/test1_best/best_combined.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aitabs.eval.cache import AudioCacheEntry, cache_path_for_clip
from aitabs.eval.config import PipelineConfig
from aitabs.eval.dataset import discover_pairs
from aitabs.eval.evaluate import evaluate_cached_pair, evaluate_pair
from aitabs.eval.gp5_import import load_gp5_notes
from aitabs.eval.run import audio_stage_to_cache, run_audio_stage
from aitabs.pipeline.mapping.fingering_config import FingeringConfig


def _score_line(scores) -> str:
    return (
        f"obj={composite_score(scores):.3f} pitch F1={scores.pitch_f1:.3f} "
        f"tab F1={scores.tab_f1:.3f} pred/ref={scores.pred_note_count}/{scores.ref_note_count}"
    )


def composite_score(scores) -> float:
    """Balance F1 with note-count sanity (target pred ≈ ref)."""
    ref = max(scores.ref_note_count, 1)
    ratio = scores.pred_note_count / ref
    count_pen = min(abs(ratio - 1.0), 0.6) * 0.25
    return 0.55 * scores.pitch_f1 + 0.45 * scores.tab_f1 - count_pen


def _is_better(candidate, best) -> bool:
    cs, bs = composite_score(candidate), composite_score(best)
    if cs > bs + 1e-6:
        return True
    if abs(cs - bs) <= 1e-6 and candidate.pitch_f1 > best.pitch_f1:
        return True
    return False


def _eval_pipeline(pair, config: PipelineConfig, work_dir: Path, onset_tol: float):
    return evaluate_pair(
        pair,
        config,
        work_dir=work_dir,
        onset_tolerance_sec=onset_tol,
    )


def _try_axis(
    pair,
    best_cfg: PipelineConfig,
    best,
    work_root: Path,
    onset_tol: float,
    name: str,
    axis: list[tuple[str, object]],
) -> tuple[PipelineConfig, object]:
    for field, value in axis:
        cfg = PipelineConfig(**{**best_cfg.to_dict(), field: value})
        scores = _eval_pipeline(pair, cfg, work_root / f"{name}_{field}_{value}", onset_tol)
        print(f"  {field}={value!s}  {_score_line(scores)}")
        if _is_better(scores, best):
            best = scores
            best_cfg = cfg
    return best_cfg, best


def _coordinate_audio_search(
    pair,
    base: PipelineConfig,
    work_root: Path,
    onset_tol: float,
    *,
    recall_focus: bool,
) -> tuple[PipelineConfig, dict]:
    best_cfg = base
    best = _eval_pipeline(pair, best_cfg, work_root / "base", onset_tol)
    print(f"  baseline  {_score_line(best)}")

    if recall_focus:
        conf_vals = (0.40, 0.44, 0.48, 0.52, 0.56)
        merge_vals = (0.12, 0.15, 0.18, 0.22, 0.26, 0.30)
        onset_vals = (0.40, 0.45, 0.50, 0.55, 0.60)
        frame_vals = (0.25, 0.30, 0.35, 0.40)
    else:
        conf_vals = (0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72)
        merge_vals = (0.18, 0.22, 0.26, 0.30, 0.35)
        onset_vals = (0.45, 0.55, 0.65)
        frame_vals = (0.25, 0.35, 0.45)

    best_cfg, best = _try_axis(
        pair, best_cfg, best, work_root, onset_tol, "merge",
        [("pitch_merge_sec", v) for v in merge_vals],
    )
    best_cfg, best = _try_axis(
        pair, best_cfg, best, work_root, onset_tol, "conf",
        [("confidence_threshold", v) for v in conf_vals],
    )
    best_cfg, best = _try_axis(
        pair, best_cfg, best, work_root, onset_tol, "onset",
        [("onset_threshold", v) for v in onset_vals],
    )
    best_cfg, best = _try_axis(
        pair, best_cfg, best, work_root, onset_tol, "frame",
        [("frame_threshold", v) for v in frame_vals],
    )
    best_cfg, best = _try_axis(
        pair, best_cfg, best, work_root, onset_tol, "minlen",
        [("minimum_note_length_ms", v) for v in (40.0, 58.0, 80.0, 100.0)],
    )
    for melodia in (True, False):
        cfg = PipelineConfig(**{**best_cfg.to_dict(), "melodia_trick": melodia})
        scores = _eval_pipeline(pair, cfg, work_root / f"melodia_{melodia}", onset_tol)
        print(f"  melodia_trick={melodia}  {_score_line(scores)}")
        if _is_better(scores, best):
            best = scores
            best_cfg = cfg

    for librosa in (False, True):
        cfg = PipelineConfig(**{**best_cfg.to_dict(), "snap_to_librosa_onsets": librosa})
        scores = _eval_pipeline(pair, cfg, work_root / f"librosa_{librosa}", onset_tol)
        print(f"  snap_to_librosa_onsets={librosa}  {_score_line(scores)}")
        if _is_better(scores, best):
            best = scores
            best_cfg = cfg

    if best_cfg.snap_to_librosa_onsets:
        best_cfg, best = _try_axis(
            pair, best_cfg, best, work_root, onset_tol, "delta",
            [("librosa_onset_delta", v) for v in (0.05, 0.07, 0.10)],
        )
        best_cfg, best = _try_axis(
            pair, best_cfg, best, work_root, onset_tol, "snap",
            [("librosa_onset_max_snap_sec", v) for v in (0.03, 0.05, 0.08, 0.12)],
        )

    for chord_bpm in (True, False):
        cfg = PipelineConfig(**{**best_cfg.to_dict(), "chord_group_bpm": chord_bpm})
        scores = _eval_pipeline(pair, cfg, work_root / f"chord_bpm_{chord_bpm}", onset_tol)
        print(f"  chord_group_bpm={chord_bpm}  {_score_line(scores)}")
        if _is_better(scores, best):
            best = scores
            best_cfg = cfg

    return best_cfg, best.__dict__


def _onset_tol_search(pair, config: PipelineConfig, work_dir: Path) -> tuple[float, dict]:
    best_tol = 0.05
    best = _eval_pipeline(pair, config, work_dir / "tol_0.05", 0.05)
    print(f"  onset_tol=0.05  {_score_line(best)}")
    for tol in (0.07, 0.08, 0.10, 0.12):
        scores = _eval_pipeline(pair, config, work_dir / f"tol_{tol}", tol)
        print(f"  onset_tol={tol}  {_score_line(scores)}")
        if _is_better(scores, best):
            best = scores
            best_tol = tol
    return best_tol, best.__dict__


def _fingering_search(
    pair,
    cache: AudioCacheEntry,
    pipeline_cfg: PipelineConfig,
    onset_tol: float,
) -> tuple[FingeringConfig, dict]:
    candidates = [
        FingeringConfig(mapper="viterbi"),
        FingeringConfig(mapper="greedy"),
        FingeringConfig(mapper="viterbi", max_fret_span=3, span_base_penalty=7.0),
        FingeringConfig(mapper="viterbi", max_fret_span=5, span_base_penalty=4.0),
        FingeringConfig(mapper="viterbi", string_cost=0.75, fret_cost=0.35),
        FingeringConfig(mapper="viterbi", string_cost=1.25, fret_cost=0.15),
        FingeringConfig(mapper="viterbi", low_fret_penalty=0.05),
        FingeringConfig(mapper="viterbi", low_fret_penalty=0.20),
        FingeringConfig(mapper="viterbi", open_string_bonus=-0.2),
        FingeringConfig(mapper="viterbi", open_string_bonus=-1.0),
    ]
    best_cfg = FingeringConfig()
    best = evaluate_cached_pair(
        pair, cache, pipeline_cfg, fingering_config=best_cfg, onset_tolerance_sec=onset_tol
    )
    print(f"  finger baseline  {_score_line(best)}")

    for i, finger_cfg in enumerate(candidates):
        scores = evaluate_cached_pair(
            pair,
            cache,
            pipeline_cfg,
            fingering_config=finger_cfg,
            onset_tolerance_sec=onset_tol,
        )
        print(f"  finger trial {i}  {_score_line(scores)}")
        if _is_better(scores, best):
            best = scores
            best_cfg = finger_cfg

    return best_cfg, best.__dict__


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune one clip (coordinate search)")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--clip-id", type=str, default="test1")
    parser.add_argument("--no-demucs", action="store_true")
    parser.add_argument("--onset-tol", type=float, default=0.05)
    parser.add_argument(
        "--from-config",
        type=Path,
        default=None,
        help="Start from prior best JSON (enables recall-focused search)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: data_dir/<clip_id>_best",
    )
    parser.add_argument(
        "--skip-fingering",
        action="store_true",
        help="Only tune audio/BasicPitch knobs",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    pairs = discover_pairs(data_dir)
    pair = next((p for p in pairs if p.clip_id == args.clip_id), None)
    if pair is None:
        print(f"Clip {args.clip_id!r} not found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.output_dir or data_dir / f"{args.clip_id}_best").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = out_dir / ("tune_runs_v2" if args.from_config else "tune_runs")
    work_root.mkdir(parents=True, exist_ok=True)

    if args.from_config and args.from_config.is_file():
        base = PipelineConfig.load_json(args.from_config)
        recall_focus = True
        print(f"Starting from {args.from_config}")
    else:
        base = PipelineConfig(use_demucs=not args.no_demucs, export_gp5=False)
        recall_focus = False
    if args.no_demucs:
        base.use_demucs = False

    print(f"Tuning {pair.clip_id} (audio → fingering)\n--- Audio / BasicPitch ---")
    best_pipeline, audio_scores = _coordinate_audio_search(
        pair, base, work_root, args.onset_tol, recall_focus=recall_focus
    )

    print("\n--- Onset tolerance (eval alignment) ---")
    best_onset_tol, tol_scores = _onset_tol_search(
        pair, best_pipeline, work_root / "onset_tol"
    )
    best_pipeline.save_json(out_dir / "best_pipeline.json")

    summary: dict = {
        "clip_id": pair.clip_id,
        "audio_stage": audio_scores,
        "onset_tol_search": tol_scores,
        "best_onset_tol": best_onset_tol,
        "pipeline": best_pipeline.to_dict(),
        "recall_focus": recall_focus,
    }

    if not args.skip_fingering:
        print("\n--- Fingering (cached audio) ---")
        cache_path = cache_path_for_clip(out_dir / "cache", pair.clip_id)
        print(f"Rebuilding cache → {cache_path}")
        ref_bpm = None
        if best_pipeline.use_reference_bpm:
            _, ref_bpm, _ = load_gp5_notes(pair.reference_gp5_path)
        audio = run_audio_stage(
            pair.audio_path,
            best_pipeline,
            reference_bpm=ref_bpm,
            stem_path=pair.stem_path,
            work_dir=work_root / "cache_build",
        )
        audio_stage_to_cache(audio, pair.clip_id, audio_path=str(pair.audio_path)).save(cache_path)

        cache = AudioCacheEntry.load(cache_path)
        best_finger, finger_scores = _fingering_search(
            pair, cache, best_pipeline, best_onset_tol
        )
        best_finger.save_json(out_dir / "best_fingering.json")
        merged = PipelineConfig(
            **{
                **best_pipeline.to_dict(),
                "fingering_config_path": str(out_dir / "best_fingering.json"),
                "fingering_mapper": best_finger.mapper,
                "export_gp5": True,
            }
        )
        merged.save_json(out_dir / "best_combined.json")
        summary["fingering"] = best_finger.to_dict()
        summary["fingering_stage"] = finger_scores
        summary["combined_config"] = str(out_dir / "best_combined.json")

        final = evaluate_cached_pair(
            pair,
            cache,
            best_pipeline,
            fingering_config=best_finger,
            onset_tolerance_sec=best_onset_tol,
        )
        summary["final"] = final.__dict__
        print(f"\n--- Final (onset_tol={best_onset_tol}) ---\n  {final.summary_line()}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out_dir / 'best_combined.json'}")
    print(
        f"Eval: python scripts/eval_dataset.py {data_dir} "
        f"--config {out_dir / 'best_combined.json'} --no-demucs --onset-tol {best_onset_tol}"
    )


if __name__ == "__main__":
    main()
