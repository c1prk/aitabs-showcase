#!/usr/bin/env python3
"""Evaluate pipeline vs reference GP5 files for all clips under a data folder.

Example layout::

    data/eval/
      audio/song_01.mp3
      reference/song_01.gp5

Or flat::

    data/eval/song_01.mp3
    data/eval/song_01.gp5

Usage::

    python scripts/eval_dataset.py data/eval
    python scripts/eval_dataset.py data/eval --config eval/results/best_config.json
    python scripts/eval_dataset.py data/eval --no-demucs  # use full mix (faster, worse)
    python scripts/eval_dataset.py data/eval --rhythm    # force GP5 rhythm scoring
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aitabs.eval.config import PipelineConfig
from aitabs.eval.dataset import discover_guitarset_pairs, discover_pairs
from aitabs.eval.evaluate import (
    evaluate_pair,
    evaluate_pair_with_notes,
    evaluate_pair_with_rhythm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-evaluate pipeline vs reference GP5")
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Folder with paired .mp3/.wav and .gp5 files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file from tune_knobs.py (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write per-clip JSON + summary (default: data_dir/../eval_results)",
    )
    parser.add_argument("--no-demucs", action="store_true", help="Skip Demucs separation")
    parser.add_argument(
        "--guitarset",
        action="store_true",
        help="Treat data_dir as GuitarSet (audio + .jams); real acoustic recordings",
    )
    parser.add_argument(
        "--include-comp",
        action="store_true",
        help="With --guitarset, also include 'comp' (chord) takes (default: solo only)",
    )
    parser.add_argument(
        "--guitarset-audio",
        choices=["mic", "mix", "hex"],
        default="mic",
        help="With --guitarset, which audio variant to use (default: mic = clean acoustic)",
    )
    parser.add_argument(
        "--rhythm",
        action="store_true",
        help="Score exported GP5 rhythm/notation (default when export_gp5=True in config)",
    )
    parser.add_argument(
        "--no-rhythm",
        action="store_true",
        help="Skip GP5 rhythm scoring even when export_gp5=True",
    )
    parser.add_argument(
        "--onset-tol",
        type=float,
        default=0.05,
        help="Match tolerance in seconds (default 50ms)",
    )
    parser.add_argument(
        "--detector",
        choices=["basic_pitch", "recall_first", "finetuned", "kong"],
        default=None,
        help="Detector backend (default: basic_pitch)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to fine-tuned SavedModel (finetuned) or Kong checkpoint .pth (kong)",
    )
    parser.add_argument(
        "--onset-threshold", type=float, default=None,
        help="Override onset_threshold (default: from config, 0.5)",
    )
    parser.add_argument(
        "--frame-threshold", type=float, default=None,
        help="Override frame_threshold (default: from config, 0.3)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=None,
        help="Override confidence_threshold (default: from config, 0.42)",
    )
    parser.add_argument(
        "--suppress-harmonics",
        action="store_true",
        help="Enable octave/overtone ghost removal (overrides config; A/B over-detection)",
    )
    parser.add_argument(
        "--harmonic-conf-ratio",
        type=float,
        default=None,
        help="With --suppress-harmonics: max harmonic/fundamental confidence ratio (default 1.0)",
    )
    parser.add_argument(
        "--dump-notes",
        type=Path,
        default=None,
        help="Write per-clip {reference,prediction} note lists here for offline error analysis",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip clips whose per-clip JSON already exists in the output dir and "
             "reuse it for the summary. Lets a killed long eval continue without "
             "redoing finished clips.",
    )
    parser.add_argument(
        "--pitch-merge-sec", type=float, default=None,
        help="Override config.pitch_merge_sec (note-merge width in clean_notes_with_tempo). "
             "Set 0 to disable merging.",
    )
    parser.add_argument(
        "--clip-filter", type=str, default=None,
        help="Only evaluate clips whose clip_id contains this substring (e.g. 'vidtest1' or '00_').",
    )
    parser.add_argument(
        "--limit-clips", type=int, default=None,
        help="Evaluate only the first N pairs (sorted). GuitarSet: first 30 = player 00 (held-out).",
    )
    parser.add_argument(
        "--rhythm-quantizer", type=str, default=None,
        choices=["adaptive", "grid", "fingerstyle_cell", "none"],
        help="Override config.rhythm_quantizer. 'none' keeps raw detected onsets "
             "(use for pure pitch/onset accuracy; the default 'adaptive' snaps onsets "
             "to a beat grid for notation, which hurts pitch-F1 on real performance timing).",
    )
    parser.add_argument(
        "--align", type=str, default="none", choices=["none", "warp", "dtw"],
        help="Reference/prediction time alignment before matching. 'none' = constant "
             "offset (correct for performance-timed refs like GuitarSet). 'warp' = global "
             "linear tempo warp (recommended for notated GP5 refs vs a human performance). "
             "'dtw' = order-preserving pitch match, time-agnostic ceiling.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"Not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.guitarset:
        pairs = discover_guitarset_pairs(
            data_dir,
            solo_only=not args.include_comp,
            audio_kind=args.guitarset_audio,
        )
        expected = "GuitarSet *.jams + matching *_mic.wav audio"
    else:
        pairs = discover_pairs(data_dir)
        expected = "audio/clip.mp3 + reference/clip.gp5 or clip.mp3 + clip.gp5"
    if not pairs:
        print(f"No pairs found under {data_dir}\nExpected e.g. {expected}", file=sys.stderr)
        sys.exit(1)

    if args.clip_filter:
        pairs = [p for p in pairs if args.clip_filter in p.clip_id]
    if args.limit_clips is not None:
        pairs = pairs[: args.limit_clips]
    if not pairs:
        print("No pairs left after --clip-filter/--limit-clips", file=sys.stderr)
        sys.exit(1)

    if args.config and args.config.is_file():
        config = PipelineConfig.load_json(args.config)
    else:
        config = PipelineConfig()

    if args.no_demucs or args.guitarset:
        # GuitarSet 'mic' is already isolated solo guitar — separation only adds artifacts.
        config.use_demucs = False
    if args.detector is not None:
        config.detector = args.detector
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.onset_threshold is not None:
        config.onset_threshold = args.onset_threshold
    if args.frame_threshold is not None:
        config.frame_threshold = args.frame_threshold
    if args.confidence_threshold is not None:
        config.confidence_threshold = args.confidence_threshold
    if args.suppress_harmonics:
        config.suppress_harmonics = True
    if args.harmonic_conf_ratio is not None:
        config.harmonic_conf_ratio = args.harmonic_conf_ratio
    if args.rhythm:
        config.export_gp5 = True
    if args.pitch_merge_sec is not None:
        config.pitch_merge_sec = args.pitch_merge_sec
    if args.rhythm_quantizer is not None:
        config.rhythm_quantizer = args.rhythm_quantizer

    score_rhythm = (config.export_gp5 or args.rhythm) and not args.no_rhythm

    out_dir = args.output_dir or (data_dir.parent / "eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pairs)} clip(s). Config: use_demucs={config.use_demucs}")
    if score_rhythm:
        print("Rhythm scoring: on (exported GP5 vs reference)")
    print(f"Results -> {out_dir}\n")

    all_scores = []
    all_rhythm = []
    for pair in pairs:
        clip_json = out_dir / f"{pair.clip_id}.json"
        if args.resume and clip_json.exists():
            import types
            cached = json.loads(clip_json.read_text(encoding="utf-8"))
            all_scores.append(types.SimpleNamespace(**cached))
            print(f"Evaluating {pair.clip_id} ... (resumed from cache)", flush=True)
            continue
        print(f"Evaluating {pair.clip_id} ...", flush=True)
        try:
            if score_rhythm:
                scores, rscores = evaluate_pair_with_rhythm(
                    pair,
                    config,
                    work_dir=run_dir / pair.clip_id,
                    onset_tolerance_sec=args.onset_tol,
                    align=args.align,
                )
                if rscores is not None:
                    all_rhythm.append(rscores)
            else:
                if args.dump_notes:
                    scores, ref_notes, pred_notes = evaluate_pair_with_notes(
                        pair,
                        config,
                        work_dir=run_dir / pair.clip_id,
                        onset_tolerance_sec=args.onset_tol,
                        align=args.align,
                    )
                    dump_dir = args.dump_notes
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    (dump_dir / f"{pair.clip_id}.notes.json").write_text(
                        json.dumps(
                            {
                                "clip_id": pair.clip_id,
                                "reference": ref_notes,
                                "prediction": pred_notes,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                else:
                    scores = evaluate_pair(
                        pair,
                        config,
                        work_dir=run_dir / pair.clip_id,
                        onset_tolerance_sec=args.onset_tol,
                        align=args.align,
                    )
            all_scores.append(scores)
            print(f"  {scores.summary_line()}")
            if score_rhythm and all_rhythm and all_rhythm[-1].clip_id == pair.clip_id:
                print(f"  {all_rhythm[-1].summary_line()}")

            clip_out = scores.__dict__.copy()
            if score_rhythm and all_rhythm and all_rhythm[-1].clip_id == pair.clip_id:
                clip_out["rhythm"] = all_rhythm[-1].__dict__
            (out_dir / f"{pair.clip_id}.json").write_text(
                json.dumps(clip_out, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)

    if not all_scores:
        sys.exit(1)

    def mean(attr: str) -> float:
        vals = [getattr(s, attr) for s in all_scores]
        return sum(vals) / len(vals)

    summary = {
        "clips": len(all_scores),
        "mean_pitch_f1": mean("pitch_f1"),
        "mean_tab_f1": mean("tab_f1"),
        "mean_pitch_onset_mae_sec": mean("pitch_onset_mae_sec"),
        "config": config.to_dict(),
    }
    if all_rhythm:
        def rmean(attr: str) -> float:
            vals = [getattr(s, attr) for s in all_rhythm]
            return sum(vals) / len(vals)

        summary["mean_metric_position_f1"] = rmean("metric_position_f1")
        summary["mean_note_value_accuracy"] = rmean("note_value_accuracy")
        summary["mean_rest_f1"] = rmean("rest_f1")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if len(all_scores) > 1:
        print("\n--- Mean (all clips; prefer per-clip lines above when configs differ) ---")
    else:
        print("\n--- Mean ---")
    print(f"  pitch F1: {summary['mean_pitch_f1']:.3f}")
    print(f"  tab F1:   {summary['mean_tab_f1']:.3f}")
    print(f"  onset MAE: {summary['mean_pitch_onset_mae_sec']*1000:.1f} ms")
    if all_rhythm:
        print(f"  rhythm pos F1:   {summary['mean_metric_position_f1']:.3f}")
        print(f"  note-value acc:  {summary['mean_note_value_accuracy']:.3f}")
        print(f"  rest F1:         {summary['mean_rest_f1']:.3f}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
