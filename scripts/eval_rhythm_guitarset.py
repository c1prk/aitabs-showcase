#!/usr/bin/env python3
"""Score the pipeline's beat/downbeat/tempo/meter estimation against GuitarSet.

This validates the *foundation* of rhythm notation — the metrical grid — on
**real performed audio** with **ground-truth beats** (GuitarSet ``beat_position``
annotations, MIT licensed). Everything the rhythm quantiser does (snapping onsets
to beats, note values, rests, bar lines) is built on this grid, so if beat/
downbeat tracking is off, no downstream notation heuristic can be right.

Unlike the GP5-reference rhythm eval (which scores against a synthetic constant-
tempo notated grid), this measures whether we recover the *actual* performed beat
times — the honest test of `auto_rhythm` on performances.

Metrics (per clip, then aggregate):
  * beat_F      — mir_eval beat F-measure (0.07 s window)
  * downbeat_F  — same, on estimated vs annotated bar downbeats
  * tempo error — octave-folded relative BPM error
  * meter       — estimated bar length (numerator) == annotation

Usage::

    python scripts/eval_rhythm_guitarset.py data/eval/guitarset
    python scripts/eval_rhythm_guitarset.py data/eval/guitarset --refine   # snap beats to onsets
    python scripts/eval_rhythm_guitarset.py data/eval/guitarset --limit 5 --out data/eval_results/rhythm_beats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aitabs.eval.dataset import discover_guitarset_pairs
from aitabs.eval.guitarset_import import load_guitarset_beats
from aitabs.eval.rhythm_metrics import BeatScores, score_beats
from aitabs.pipeline.audio.onset import detect_onsets
from aitabs.pipeline.audio.tempo import estimate_meter, estimate_tempo


def _predicted_downbeats(beat_times, numerator: int, downbeat_beat: int) -> list[float]:
    """Beat times that fall on a bar start given the estimated phase.

    ``downbeat_beat`` is the beat-in-bar index (0-based) of ``beat_times[0]``
    (from :class:`MeterEstimate`). A downbeat is any beat whose bar position is 0.
    """
    if numerator <= 0:
        return list(beat_times)
    first = (numerator - (downbeat_beat % numerator)) % numerator
    return [float(beat_times[i]) for i in range(first, len(beat_times), numerator)]


def evaluate_clip(pair, *, refine: bool) -> BeatScores:
    ref = load_guitarset_beats(pair.reference_jams_path)

    analysis = estimate_tempo(str(pair.audio_path))
    beat_times = list(analysis.beat_times)
    onsets = detect_onsets(str(pair.audio_path))

    if refine and len(beat_times) >= 2 and len(onsets):
        from aitabs.pipeline.audio.tempo import refine_beats_to_onsets

        beat_times = list(refine_beats_to_onsets(beat_times, onsets))

    meter = estimate_meter(onsets, beat_times)
    pred_downbeats = _predicted_downbeats(beat_times, meter.numerator, meter.downbeat_beat)

    return score_beats(
        ref.beat_times,
        beat_times,
        reference_downbeats=ref.downbeat_times,
        predicted_downbeats=pred_downbeats,
        reference_bpm=ref.tempo_bpm,
        predicted_bpm=analysis.bpm,
        reference_numerator=ref.numerator,
        predicted_numerator=meter.numerator,
        clip_id=pair.clip_id,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Score beat/downbeat/tempo vs GuitarSet")
    ap.add_argument("data_dir", type=Path, help="GuitarSet root (audio + .jams)")
    ap.add_argument("--audio-kind", default="mic", help="mic (default) / mix / hex")
    ap.add_argument("--comp", action="store_true", help="Include comp (chord) takes")
    ap.add_argument("--limit", type=int, default=None, help="Only first N clips")
    ap.add_argument(
        "--refine",
        action="store_true",
        help="Snap tracked beats to nearest onsets (refine_beats_to_onsets) before scoring",
    )
    ap.add_argument("--out", type=Path, default=None, help="Write per-clip + summary JSON here")
    args = ap.parse_args()

    pairs = discover_guitarset_pairs(
        args.data_dir, solo_only=not args.comp, audio_kind=args.audio_kind
    )
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("No GuitarSet pairs found", file=sys.stderr)
        raise SystemExit(1)

    print(f"Scoring {len(pairs)} clips (refine={args.refine})\n")
    scores: list[BeatScores] = []
    for pair in pairs:
        s = evaluate_clip(pair, refine=args.refine)
        scores.append(s)
        print("  " + s.summary_line())

    n = len(scores)
    mean_beat = sum(s.beat_f for s in scores) / n
    dbs = [s.downbeat_f for s in scores if s.downbeat_f == s.downbeat_f]  # drop nan
    mean_db = sum(dbs) / len(dbs) if dbs else float("nan")
    mean_tempo_err = sum(s.tempo_pct_err for s in scores) / n
    meter_acc = sum(1 for s in scores if s.meter_correct) / n

    print("\n" + "=" * 60)
    print(f"MEAN beat_F      = {mean_beat:.4f}  ({n} clips)")
    print(f"MEAN downbeat_F  = {mean_db:.4f}")
    print(f"MEAN tempo err   = {mean_tempo_err*100:.2f}%  (octave-folded)")
    print(f"meter correct    = {meter_acc*100:.1f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "refine": args.refine,
            "n_clips": n,
            "mean_beat_f": mean_beat,
            "mean_downbeat_f": mean_db,
            "mean_tempo_pct_err": mean_tempo_err,
            "meter_accuracy": meter_acc,
            "clips": [
                {
                    "clip_id": s.clip_id,
                    "beat_f": s.beat_f,
                    "downbeat_f": s.downbeat_f,
                    "tempo_bpm_ref": s.tempo_bpm_ref,
                    "tempo_bpm_pred": s.tempo_bpm_pred,
                    "tempo_pct_err": s.tempo_pct_err,
                    "meter_correct": s.meter_correct,
                    "n_ref_beats": s.n_ref_beats,
                    "n_pred_beats": s.n_pred_beats,
                }
                for s in scores
            ],
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
