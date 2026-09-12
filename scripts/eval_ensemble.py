#!/usr/bin/env python3
"""A/B the Kong+BasicPitch ensemble against each detector alone on GuitarSet.

Kong inference is read from the npz cache written by ``calibrate_detector.py``
(``--cache-dir``); BasicPitch inference is cached here as per-clip JSON. Both
caches are resumable and this script only runs ``--bp-budget`` new BasicPitch
inferences per invocation, so it finishes inside the environment's job-kill
window — run it repeatedly until every clip is cached, then it prints the full
comparison table (Kong-only, BasicPitch-only, union, intersection, gated).

Example (run a few times until BP cache fills, then it reports)::

    python scripts/eval_ensemble.py --max-clips 16 --bp-budget 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aitabs.eval.guitarset_import import load_guitarset_notes
from aitabs.eval.metrics import compare_note_lists
from aitabs.eval.types import EvalNote
from aitabs.pipeline.audio.ensemble import reconcile_notes
from calibrate_detector import _kong_notes_from_output  # reuse Kong decode


def _kong_cache_path(cache_dir: Path, ckpt: str, wav: str) -> Path:
    mtime = int(os.path.getmtime(ckpt))
    raw = f"{os.path.abspath(ckpt)}|{mtime}|{os.path.abspath(wav)}"
    return cache_dir / f"{hashlib.sha1(raw.encode()).hexdigest()[:16]}.npz"


def _bp_cache_path(cache_dir: Path, wav: str, model_path: str | None = None) -> Path:
    # Stock BP (model_path=None) keeps its original key; a fine-tuned model gets a
    # distinct key so its cache doesn't collide with the stock one.
    src = os.path.abspath(wav) if not model_path else f"{os.path.abspath(wav)}|{model_path}"
    return cache_dir / f"{hashlib.sha1(src.encode()).hexdigest()[:16]}.json"


def _run_basic_pitch(wav: str, onset: float, frame: float, conf: float,
                     model_path: str | None = None) -> list[dict]:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from aitabs.pipeline.audio.basic_pitch_infer import predict_notes

    _mo, _midi, events = predict_notes(
        wav, model_path or str(ICASSP_2022_MODEL_PATH),
        onset_threshold=onset, frame_threshold=frame, minimum_note_length=58,
    )
    out = []
    for start, end, pitch, c, *_ in events:
        if c >= conf:
            out.append({"start": float(start), "end": float(end),
                        "pitch_midi": int(pitch), "confidence": float(c)})
    return out


def _eval_notes(ref_notes, pred_notes) -> tuple[float, float, float]:
    ref = [EvalNote(start=n["start"], end=n["end"], pitch_midi=n["pitch_midi"],
                    string=n.get("string", -1), fret=n.get("fret", -1),
                    confidence=n.get("confidence", 1.0)) for n in ref_notes]
    pred = [EvalNote(start=n["start"], end=n["end"], pitch_midi=n["pitch_midi"],
                     string=-1, fret=-1, confidence=n.get("confidence", 1.0)) for n in pred_notes]
    s = compare_note_lists(ref, pred, onset_tolerance_sec=0.05)
    return s.pitch_precision, s.pitch_recall, s.pitch_f1


def main() -> None:
    ap = argparse.ArgumentParser(description="Kong+BasicPitch ensemble A/B on GuitarSet")
    ap.add_argument("--guitarset-dir", type=Path, default=Path("data/eval/guitarset"))
    ap.add_argument("--kong-ckpt", default="eval/kong2_ft_step4000.pth")
    ap.add_argument("--kong-cache", type=Path, default=Path("data/eval_results/cache_kong2_4000"))
    ap.add_argument("--bp-cache", type=Path, default=Path("data/eval_results/cache_bp_gs"))
    ap.add_argument("--max-clips", type=int, default=16)
    ap.add_argument("--bp-budget", type=int, default=6, help="max new BP inferences this run")
    ap.add_argument("--kong-onset", type=float, default=0.3)
    ap.add_argument("--kong-frame", type=float, default=0.15)
    ap.add_argument("--kong-conf", type=float, default=0.2)
    ap.add_argument("--bp-onset", type=float, default=0.7)  # calibrated on GuitarSet
    ap.add_argument("--bp-frame", type=float, default=0.3)
    ap.add_argument("--bp-conf", type=float, default=0.5)
    ap.add_argument("--bp-model-path", default=None,
                    help="fine-tuned BasicPitch SavedModel dir (default: stock ICASSP-2022)")
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    args.bp_cache.mkdir(parents=True, exist_ok=True)
    anno = args.guitarset_dir / "annotation"
    audio = args.guitarset_dir / "audio"
    jams = sorted(anno.glob("*.jams"))[: args.max_clips]

    clips = []            # (clip_id, ref_notes, kong_notes, bp_notes)
    bp_new = 0
    n_missing_bp = 0
    for jf in jams:
        cid = jf.stem
        wav = audio / f"{cid}_mic.wav"
        if not wav.is_file():
            continue
        kpath = _kong_cache_path(args.kong_cache, args.kong_ckpt, str(wav))
        if not kpath.exists():
            continue  # no Kong inference cached for this clip yet
        with np.load(kpath) as z:
            raw = {k: z[k] for k in z.files}
        kong_notes = _kong_notes_from_output(raw, args.kong_onset, args.kong_frame, args.kong_conf)

        bpath = _bp_cache_path(args.bp_cache, str(wav), args.bp_model_path)
        if bpath.exists():
            bp_notes = json.loads(bpath.read_text())
        elif bp_new < args.bp_budget:
            bp_notes = _run_basic_pitch(str(wav), args.bp_onset, args.bp_frame, args.bp_conf,
                                        args.bp_model_path)
            bpath.write_text(json.dumps(bp_notes))
            bp_new += 1
            print(f"  cached BP for {cid} ({len(bp_notes)} notes)", flush=True)
        else:
            n_missing_bp += 1
            continue

        ref_notes, _ = load_guitarset_notes(jf)
        clips.append((cid, ref_notes, kong_notes, bp_notes))

    print(f"\nScored on {len(clips)} clips "
          f"(BP cached this run: {bp_new}, still missing BP: {n_missing_bp})")
    if n_missing_bp:
        print("  -> re-run to cache the rest before trusting the table.\n")
    if not clips:
        return

    def mean_prf(fn):
        P, R, F = [], [], []
        for _cid, ref, kong, bp in clips:
            p, r, f = _eval_notes(ref, fn(kong, bp))
            P.append(p); R.append(r); F.append(f)
        return np.mean(P), np.mean(R), np.mean(F)

    rows = [
        ("Kong only", lambda k, b: k),
        ("BasicPitch only", lambda k, b: b),
        ("Union", lambda k, b: reconcile_notes(k, b, onset_tolerance_sec=args.tol, mode="union")),
        ("Intersection", lambda k, b: reconcile_notes(k, b, onset_tolerance_sec=args.tol, mode="intersection")),
    ]
    for mc in (0.3, 0.4, 0.5, 0.6, 0.7):
        rows.append((f"Gated (conf>={mc})",
                     lambda k, b, mc=mc: reconcile_notes(
                         k, b, onset_tolerance_sec=args.tol, mode="gated", min_singleton_conf=mc)))

    print(f"{'method':<22} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 46)
    for name, fn in rows:
        p, r, f = mean_prf(fn)
        print(f"{name:<22} {p:>7.4f} {r:>7.4f} {f:>7.4f}")


if __name__ == "__main__":
    main()
