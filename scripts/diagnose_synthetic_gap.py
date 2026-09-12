#!/usr/bin/env python3
"""Track 0 diagnostic: synthetic (detection ceiling) vs real-audio scores.

For each reference ``.gp5`` we render a clean synthetic WAV whose timing is the
reference's own note times, then score the pipeline on it. Because the synthetic
audio shares the reference timing exactly, its score is a **detection ceiling**
with the score-vs-performance timing-mismatch variable removed.

Comparing per clip:

  synth_pF1   ~ how well detection works on clean audio with perfect timing
  real_pF1    ~ what we actually get on the real recording
  gap         = synth_pF1 - real_pF1

A large gap with balanced real-audio note counts (pred/ref ~ 1) points to a
**timing / rubato** problem (bucket B) — the notes are found but the notated GP5
onsets don't match an expressive performance. A small gap (both low) points to a
**detection / timbre** problem (bucket A/C).

Usage::

    python scripts/diagnose_synthetic_gap.py data/eval \
        --config data/eval/test1_best/best_combined.json --no-demucs

Needs the full pipeline environment (BasicPitch, librosa, ...). The render step
alone (numpy/soundfile) can be exercised via aitabs.eval.render_gp5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aitabs.eval.config import PipelineConfig
from aitabs.eval.dataset import EvalPair, discover_pairs
from aitabs.eval.evaluate import evaluate_pair
from aitabs.eval.render_gp5 import render_gp5_to_wav


def _gp5_references(data_dir: Path) -> list[Path]:
    """Reference GP5s only — not pipeline preds under runs/, tune_runs/, etc."""
    pairs = discover_pairs(data_dir)
    if pairs:
        return sorted(p.reference_gp5_path for p in pairs if p.reference_gp5_path is not None)

    refs: list[Path] = []
    ref_dir = data_dir / "reference"
    search_roots = [ref_dir] if ref_dir.is_dir() else [data_dir]
    for root in search_roots:
        for ext in (".gp5", ".gp4", ".gpx"):
            refs.extend(root.glob(f"*{ext}"))
    return sorted(refs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic-vs-real detection gap diagnostic")
    parser.add_argument("data_dir", type=Path, help="Folder with .gp5 references (+ optional real audio)")
    parser.add_argument("--config", type=Path, default=None, help="PipelineConfig JSON")
    parser.add_argument("--no-demucs", action="store_true", help="Skip Demucs (always off for synth)")
    parser.add_argument("--onset-tol", type=float, default=0.05, help="Match tolerance seconds")
    parser.add_argument(
        "--synth-dir",
        type=Path,
        default=None,
        help="Where to write synthetic WAVs (default: data_dir/synthetic)",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON report path")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"Not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig.load_json(args.config) if args.config and args.config.is_file() else PipelineConfig()

    synth_dir = args.synth_dir or (data_dir / "synthetic")
    synth_dir.mkdir(parents=True, exist_ok=True)

    # Real pairs (need real audio); index by clip_id for the gap join.
    real_pairs = {p.clip_id: p for p in discover_pairs(data_dir)}
    refs = _gp5_references(data_dir)
    if not refs:
        print(f"No .gp5 references under {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Synthetic config: identical knobs, but never separate clean synth audio.
    synth_config = PipelineConfig.from_dict(config.to_dict())
    synth_config.use_demucs = False
    if args.no_demucs:
        config.use_demucs = False

    rows = []
    for ref in refs:
        clip_id = ref.stem
        print(f"[{clip_id}] rendering + scoring synthetic ...", flush=True)
        synth_wav = synth_dir / f"{clip_id}.wav"
        try:
            render_gp5_to_wav(ref, synth_wav)
        except Exception as exc:  # noqa: BLE001
            print(f"  render FAILED: {exc}", file=sys.stderr)
            continue

        synth_pair = EvalPair(clip_id=clip_id, audio_path=synth_wav, reference_gp5_path=ref)
        try:
            synth_scores = evaluate_pair(
                synth_pair, synth_config, work_dir=synth_dir / "runs" / clip_id,
                onset_tolerance_sec=args.onset_tol,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  synth eval FAILED: {exc}", file=sys.stderr)
            continue

        real = real_pairs.get(clip_id)
        real_scores = None
        if real is not None:
            print(f"[{clip_id}] scoring real audio ...", flush=True)
            try:
                real_scores = evaluate_pair(
                    real, config, work_dir=data_dir / "_diag_runs" / clip_id,
                    onset_tolerance_sec=args.onset_tol,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  real eval FAILED: {exc}", file=sys.stderr)

        row = {
            "clip_id": clip_id,
            "synth_pitch_f1": synth_scores.pitch_f1,
            "synth_pred_ref": _ratio(synth_scores.pred_note_count, synth_scores.ref_note_count),
            "real_pitch_f1": real_scores.pitch_f1 if real_scores else None,
            "real_pred_ref": _ratio(real_scores.pred_note_count, real_scores.ref_note_count) if real_scores else None,
            "real_offset_sec": real_scores.time_offset_sec if real_scores else None,
            "gap": (synth_scores.pitch_f1 - real_scores.pitch_f1) if real_scores else None,
        }
        rows.append(row)
        _print_row(row)

    rows.sort(key=lambda r: (r["gap"] is None, -(r["gap"] or 0.0)))
    out = args.output or (data_dir.parent / "eval_results" / "synthetic_gap.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clips": rows, "config": config.to_dict()}, indent=2), encoding="utf-8")

    print("\n=== Synthetic vs real (sorted by gap) ===")
    print(f"{'clip':28}{'synth':>7}{'real':>7}{'gap':>7}{'r.pred/ref':>11}{'offset':>8}")
    for r in rows:
        _print_summary(r)
    print(f"\nWrote {out}")
    print("\nReading the gap:")
    print("  big gap + real pred/ref~1  -> TIMING/rubato (bucket B)")
    print("  small gap, both low        -> DETECTION/timbre (bucket A/C)")


def _ratio(pred: int, ref: int) -> float | None:
    return (pred / ref) if ref else None


def _fmt(x: float | None, nd: int = 2) -> str:
    return "  -  " if x is None else f"{x:.{nd}f}"


def _print_row(r: dict) -> None:
    print(
        f"  synth_pF1={_fmt(r['synth_pitch_f1'])} "
        f"real_pF1={_fmt(r['real_pitch_f1'])} gap={_fmt(r['gap'])}"
    )


def _print_summary(r: dict) -> None:
    off = r["real_offset_sec"]
    off_s = "  -  " if off is None else f"{off*1000:+.0f}ms"
    print(
        f"{r['clip_id'][:28]:28}{_fmt(r['synth_pitch_f1']):>7}{_fmt(r['real_pitch_f1']):>7}"
        f"{_fmt(r['gap']):>7}{_fmt(r['real_pred_ref']):>11}{off_s:>8}"
    )


if __name__ == "__main__":
    main()
