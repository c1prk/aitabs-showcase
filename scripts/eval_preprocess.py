#!/usr/bin/env python3
"""Two-sided A/B for the Tier-1 preprocessing front-end on GuitarSet.

For each clip, run Kong on four variants and score vs ground truth:
  clean, clean+preprocess, degraded, degraded+preprocess.

The two questions:
  * regression check — does preprocessing hurt the already-clean set?
    (clean vs clean+prep should be ~equal)
  * recovery check   — does preprocessing recover accuracy on degraded audio?
    (degraded+prep should beat degraded)

Each (clip, variant) Kong inference is cached to JSON and only ``--budget`` new
inferences run per invocation, so it finishes inside the job-kill window — run
repeatedly until every variant is cached, then it prints the table.

    python scripts/eval_preprocess.py --max-clips 12 --budget 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aitabs.eval.degrade import degrade_audio
from aitabs.eval.guitarset_import import load_guitarset_notes
from aitabs.eval.metrics import compare_note_lists
from aitabs.eval.types import EvalNote
from aitabs.pipeline.audio.preprocess import PreprocessConfig, preprocess_audio
from aitabs.pipeline.audio.pitch import KongDetector

SR = 16000
VARIANTS = ("clean", "clean_prep", "degraded", "degraded_prep", "degraded_prep_drv")
_DRV = PreprocessConfig(dereverb=True)  # Tier-1 + Tier-3 dereverb


def _make_variant(y: np.ndarray, variant: str) -> np.ndarray:
    if variant == "clean":
        return y
    if variant == "clean_prep":
        return preprocess_audio(y, SR)
    if variant == "degraded":
        return degrade_audio(y, SR)
    if variant == "degraded_prep":
        return preprocess_audio(degrade_audio(y, SR), SR)
    if variant == "degraded_prep_drv":
        return preprocess_audio(degrade_audio(y, SR), SR, _DRV)
    raise ValueError(variant)


def _cache_path(cache_dir: Path, ckpt: str, clip_id: str, variant: str) -> Path:
    key = hashlib.sha1(f"{ckpt}|{clip_id}|{variant}".encode()).hexdigest()[:16]
    return cache_dir / f"{key}.json"


def _score(ref_notes, pred_notes) -> tuple[float, float, float]:
    ref = [EvalNote(start=n["start"], end=n["end"], pitch_midi=n["pitch_midi"],
                    string=n.get("string", -1), fret=n.get("fret", -1),
                    confidence=n.get("confidence", 1.0)) for n in ref_notes]
    pred = [EvalNote(start=n["start"], end=n["end"], pitch_midi=n["pitch_midi"],
                     string=-1, fret=-1, confidence=n.get("confidence", 1.0)) for n in pred_notes]
    s = compare_note_lists(ref, pred, onset_tolerance_sec=0.05)
    return s.pitch_precision, s.pitch_recall, s.pitch_f1


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-sided preprocessing A/B on GuitarSet")
    ap.add_argument("--guitarset-dir", type=Path, default=Path("data/eval/guitarset"))
    ap.add_argument("--kong-ckpt", default="eval/kong2_ft_step4000.pth")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/eval_results/cache_prep"))
    ap.add_argument("--max-clips", type=int, default=12)
    ap.add_argument("--budget", type=int, default=8, help="max new Kong inferences this run")
    ap.add_argument("--onset", type=float, default=0.3)
    ap.add_argument("--frame", type=float, default=0.15)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--pcen", action="store_true",
                    help="checkpoint has a trainable PCEN front-end (RESEARCH_PCEN.md)")
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    anno = args.guitarset_dir / "annotation"
    audio = args.guitarset_dir / "audio"
    jams = sorted(anno.glob("*.jams"))[: args.max_clips]

    import librosa
    import soundfile as sf

    det = KongDetector(args.kong_ckpt, pcen=args.pcen)
    budget = args.budget
    tmpdir = Path(tempfile.gettempdir())

    scored: dict[str, list[dict]] = {v: [] for v in VARIANTS}  # variant -> per-clip notes
    complete_clips = 0
    for jf in jams:
        cid = jf.stem
        wav = audio / f"{cid}_mic.wav"
        if not wav.is_file():
            continue
        ref, _ = load_guitarset_notes(jf)
        y = None
        clip_notes: dict[str, list[dict]] = {}
        clip_ok = True
        for v in VARIANTS:
            cp = _cache_path(args.cache_dir, args.kong_ckpt, cid, v)
            if cp.exists():
                clip_notes[v] = json.loads(cp.read_text())
                continue
            if budget <= 0:
                clip_ok = False
                continue
            if y is None:
                y, _ = librosa.load(str(wav), sr=SR, mono=True)
            yv = _make_variant(y, v)
            tmp = tmpdir / f"prep_{cid}_{v}.wav"
            sf.write(str(tmp), yv, SR)
            notes = det.detect(
                str(tmp), onset_threshold=args.onset, frame_threshold=args.frame,
                confidence_threshold=args.conf, minimum_note_length_ms=58.0,
                minimum_frequency_hz=82.0, maximum_frequency_hz=1400.0, melodia_trick=False)
            notes = [{"start": n["start"], "end": n["end"],
                      "pitch_midi": n["pitch_midi"], "confidence": n["confidence"]} for n in notes]
            cp.write_text(json.dumps(notes))
            clip_notes[v] = notes
            budget -= 1
            print(f"  inferred {cid}/{v} ({len(notes)} notes)", flush=True)

        if clip_ok and all(v in clip_notes for v in VARIANTS):
            complete_clips += 1
            for v in VARIANTS:
                scored[v].append({"ref": ref, "pred": clip_notes[v]})

    print(f"\nComplete clips (all 4 variants cached): {complete_clips}"
          f"   (remaining inference budget: {budget})")
    if budget <= 0 and complete_clips < len(jams):
        print("  -> re-run to finish caching the rest.\n")
    if complete_clips == 0:
        return

    print(f"{'variant':<16} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 40)
    for v in VARIANTS:
        rows = scored[v]
        P = [_score(r["ref"], r["pred"]) for r in rows]
        p, r, f = (np.mean([x[i] for x in P]) for i in range(3))
        print(f"{v:<16} {p:>7.4f} {r:>7.4f} {f:>7.4f}")

    def mean_f1(v):
        return float(np.mean([_score(r["ref"], r["pred"])[2] for r in scored[v]]))
    dr = mean_f1("degraded_prep") - mean_f1("degraded")
    cr = mean_f1("clean_prep") - mean_f1("clean")
    drv = mean_f1("degraded_prep_drv") - mean_f1("degraded")
    drv_marg = mean_f1("degraded_prep_drv") - mean_f1("degraded_prep")
    print(f"\nrecovery Tier-1 (degraded_prep - degraded):      {dr:+.4f} F1")
    print(f"recovery Tier-1+drv (degraded_prep_drv - degraded): {drv:+.4f} F1")
    print(f"  dereverb marginal (drv - prep):                {drv_marg:+.4f} F1")
    print(f"regression (clean_prep - clean):                 {cr:+.4f} F1  (want ~0 or +)")


if __name__ == "__main__":
    main()
