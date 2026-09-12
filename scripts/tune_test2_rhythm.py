#!/usr/bin/env python3
"""Rhythm-focused tuning for test2 (8th-grid export + onset collapse).

Usage::

    python scripts/tune_test2_rhythm.py data/eval --no-demucs
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from aitabs.eval.config import PipelineConfig
from aitabs.eval.dataset import discover_pairs
from aitabs.eval.evaluate import evaluate_pair
from aitabs.eval.gp5_import import load_gp5_notes
from aitabs.eval.metrics import rhythm_tick_f1
from aitabs.eval.run import run_pipeline
from aitabs.pipeline.audio.tempo import TempoAnalysis, estimate_tempo
import librosa


def _eval_rhythm(pair, config: PipelineConfig, work_dir: Path, onset_tol: float):
    ref_notes, ref_bpm, _ = load_gp5_notes(pair.reference_gp5_path)
    result = run_pipeline(
        pair.audio_path,
        config,
        reference_bpm=ref_bpm if config.use_reference_bpm else None,
        stem_path=pair.stem_path,
        work_dir=work_dir,
    )
    from aitabs.eval.run import tab_notes_to_eval

    pred_notes = tab_notes_to_eval(result.tab_notes)
    scores = __import__("aitabs.eval.metrics", fromlist=["compare_note_lists"]).compare_note_lists(
        ref_notes,
        pred_notes,
        clip_id=pair.clip_id,
        onset_tolerance_sec=onset_tol,
        reference_bpm=ref_bpm,
        estimated_bpm=result.tempo_analysis.bpm,
    )
    analysis = result.tempo_analysis
    _, _, rhythm_f1 = rhythm_tick_f1(ref_notes, pred_notes, analysis, tick_tolerance=0)
    return scores, rhythm_f1, result.gp5_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rhythm tune for test2")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--no-demucs", action="store_true")
    parser.add_argument("--onset-tol", type=float, default=0.08)
    args = parser.parse_args()

    pair = next((p for p in discover_pairs(args.data_dir) if p.clip_id == "test2"), None)
    if pair is None:
        print("test2 not found", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.data_dir / "test2_best").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = out_dir / "rhythm_tune"
    work_root.mkdir(parents=True, exist_ok=True)

    ticks = [2, 4]
    quantize = [True]
    max_per_tick: list[int | None] = [None, 1]
    snap = [True]
    duration_modes = ["grid_unit", "to_next_onset"]
    chord_grid = [0]
    merge = [0.18, 0.28]
    conf = [0.48, 0.54]
    onset = [0.52, 0.60]

    candidates: list[PipelineConfig] = []
    for tpb, q, mpt, sg, dm, cg, pm, c, o in itertools.product(
        ticks, quantize, max_per_tick, snap, duration_modes, chord_grid, merge, conf, onset
    ):
        candidates.append(
            PipelineConfig(
                use_demucs=not args.no_demucs,
                confidence_threshold=c,
                onset_threshold=o,
                pitch_merge_sec=pm,
                ticks_per_beat=tpb,
                quantize_to_grid=q,
                max_notes_per_grid_tick=mpt,
                snap_to_grid=sg,
                duration_mode=dm,
                chord_grid_ticks=cg,
                chord_group_bpm=False,
                export_gp5=True,
                snap_to_librosa_onsets=True,
                librosa_onset_max_snap_sec=0.08,
            )
        )

    print(f"Trying {len(candidates)} rhythm configs on test2 …\n")
    best_score = -1.0
    best_cfg: PipelineConfig | None = None
    best_summary: dict = {}
    leaderboard: list[dict] = []

    for i, cfg in enumerate(candidates):
        try:
            scores, rhythm_f1, gp5_path = _eval_rhythm(
                pair, cfg, work_root / f"t{i}", args.onset_tol
            )
        except Exception as exc:
            print(f"  [{i}] FAILED: {exc}")
            continue
        obj = 0.45 * rhythm_f1 + 0.35 * scores.pitch_f1 + 0.20 * scores.tab_f1
        row = {
            "trial": i,
            "objective": obj,
            "rhythm_tick_f1": rhythm_f1,
            "pitch_f1": scores.pitch_f1,
            "tab_f1": scores.tab_f1,
            "pred_ref": f"{scores.pred_note_count}/{scores.ref_note_count}",
            "config": cfg.to_dict(),
        }
        leaderboard.append(row)
        print(
            f"  [{i:3d}] obj={obj:.3f} rhythm={rhythm_f1:.3f} "
            f"pitch={scores.pitch_f1:.3f} tab={scores.tab_f1:.3f} "
            f"notes={scores.pred_note_count}/{scores.ref_note_count} "
            f"tpb={cfg.ticks_per_beat} dm={cfg.duration_mode} mpt={cfg.max_notes_per_grid_tick}"
        )
        if obj > best_score:
            best_score = obj
            best_cfg = cfg
            best_summary = {**row, "gp5_path": gp5_path}

    if best_cfg is None:
        print("No successful trials.", file=sys.stderr)
        sys.exit(1)

    best_cfg.save_json(out_dir / "best_combined.json")
    (out_dir / "rhythm_leaderboard.json").write_text(
        json.dumps(sorted(leaderboard, key=lambda x: -x["objective"]), indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    print(f"\nBest objective={best_score:.3f}")
    print(f"Saved {out_dir / 'best_combined.json'}")
    if best_summary.get("gp5_path"):
        print(f"GP5: {best_summary['gp5_path']}")


if __name__ == "__main__":
    main()
