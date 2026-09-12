#!/usr/bin/env python3
"""Do the ensemble and preprocessing levers STACK?

For each clip and each audio variant {clean, degraded, degraded_prep}, run BOTH
Kong-alone and the Kong+BasicPitch ensemble (gated), and score vs ground truth.
Answers: on real-world-style (degraded) audio, does ensemble+preprocess beat the
current best (plain Kong + preprocess = 0.914)? And does preprocessing help or
hurt the ensemble?

Kong (torch) + BasicPitch (TF) inference is cached per (clip, variant, detector)
JSON and budgeted per run, so it survives the job-kill window — re-run until every
cell is cached, then it prints the table.

    python scripts/eval_ensemble_preprocess.py --max-clips 6 --budget 16
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
from aitabs.pipeline.audio.preprocess import preprocess_audio
from aitabs.pipeline.audio.pitch import KongDetector, BasicPitchDetector, clean_notes, suppress_octave_harmonics
from aitabs.pipeline.audio.ensemble import reconcile_notes

SR = 16000
VARIANTS = ("clean", "degraded", "degraded_prep")


def _variant(y, v):
    if v == "clean":
        return y
    if v == "degraded":
        return degrade_audio(y, SR)
    if v == "degraded_prep":
        return preprocess_audio(degrade_audio(y, SR), SR)
    raise ValueError(v)


def _cache(cache_dir: Path, det: str, clip: str, v: str) -> Path:
    return cache_dir / f"{hashlib.sha1(f'{det}|{clip}|{v}'.encode()).hexdigest()[:16]}.json"


def _score(ref, pred):
    R = [EvalNote(start=n["start"], end=n["end"], pitch_midi=n["pitch_midi"],
                  string=n.get("string", -1), fret=n.get("fret", -1),
                  confidence=n.get("confidence", 1.0)) for n in ref]
    P = [EvalNote(start=n["start"], end=n["end"], pitch_midi=n["pitch_midi"],
                  string=-1, fret=-1, confidence=n.get("confidence", 1.0)) for n in pred]
    s = compare_note_lists(R, P, onset_tolerance_sec=0.05)
    return s.pitch_precision, s.pitch_recall, s.pitch_f1


def main():
    ap = argparse.ArgumentParser(description="Ensemble x preprocessing stacking bench")
    ap.add_argument("--guitarset-dir", type=Path, default=Path("data/eval/guitarset"))
    ap.add_argument("--kong-ckpt", default="eval/kong2_ft_step4000.pth")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/eval_results/cache_ens_prep"))
    ap.add_argument("--max-clips", type=int, default=6)
    ap.add_argument("--budget", type=int, default=16, help="max new inferences (Kong+BP) this run")
    ap.add_argument("--kong-onset", type=float, default=0.3)
    ap.add_argument("--kong-frame", type=float, default=0.15)
    ap.add_argument("--kong-conf", type=float, default=0.2)
    ap.add_argument("--bp-onset", type=float, default=0.7)   # calibrated stock BP
    ap.add_argument("--bp-frame", type=float, default=0.3)
    ap.add_argument("--bp-conf", type=float, default=0.5)
    ap.add_argument("--gated-conf", type=float, default=0.5)
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.gettempdir())
    anno = args.guitarset_dir / "annotation"
    audio = args.guitarset_dir / "audio"
    jams = sorted(anno.glob("*.jams"))[: args.max_clips]

    kong = KongDetector(args.kong_ckpt)
    bp = BasicPitchDetector()
    budget = args.budget

    def get_notes(det_name, clip_id, v, wav_path):
        cp = _cache(args.cache_dir, det_name, clip_id, v)
        if cp.exists():
            return json.loads(cp.read_text()), False
        nonlocal_flag = True
        if det_name == "kong":
            notes = kong.detect(wav_path, onset_threshold=args.kong_onset, frame_threshold=args.kong_frame,
                                confidence_threshold=args.kong_conf, minimum_note_length_ms=58.0,
                                minimum_frequency_hz=82.0, maximum_frequency_hz=1400.0, melodia_trick=False)
        else:
            raw = bp.detect(wav_path, onset_threshold=args.bp_onset, frame_threshold=args.bp_frame,
                            confidence_threshold=args.bp_conf, minimum_note_length_ms=58.0,
                            minimum_frequency_hz=82.0, maximum_frequency_hz=1400.0, melodia_trick=False)
            notes = suppress_octave_harmonics(clean_notes(raw), max_conf_ratio=1.0)
        notes = [{"start": n["start"], "end": n["end"], "pitch_midi": n["pitch_midi"],
                  "confidence": n["confidence"]} for n in notes]
        cp.write_text(json.dumps(notes))
        return notes, nonlocal_flag

    scored = {v: {"kong": [], "ens": []} for v in VARIANTS}
    complete = 0
    for jf in jams:
        cid = jf.stem
        wav = audio / f"{cid}_mic.wav"
        if not wav.is_file():
            continue
        ref, _ = load_guitarset_notes(jf)
        y = None
        clip_cells = {}
        ok = True
        for v in VARIANTS:
            for det in ("kong", "bp"):
                cp = _cache(args.cache_dir, det, cid, v)
                if cp.exists():
                    clip_cells[(det, v)] = json.loads(cp.read_text())
                    continue
                if budget <= 0:
                    ok = False
                    continue
                if y is None:
                    y, _ = librosa.load(str(wav), sr=SR, mono=True)
                yv = _variant(y, v)
                tmp = tmpdir / f"ep_{cid}_{v}.wav"
                sf.write(str(tmp), yv, SR)
                notes, _new = get_notes(det, cid, v, str(tmp))
                clip_cells[(det, v)] = notes
                budget -= 1
                print(f"  inferred {det}/{cid}/{v} ({len(notes)} notes)", flush=True)
        if ok and all((d, v) in clip_cells for v in VARIANTS for d in ("kong", "bp")):
            complete += 1
            for v in VARIANTS:
                k = clip_cells[("kong", v)]
                b = clip_cells[("bp", v)]
                e = reconcile_notes(k, b, mode="gated", min_singleton_conf=args.gated_conf)
                scored[v]["kong"].append((ref, k))
                scored[v]["ens"].append((ref, e))

    print(f"\nComplete clips: {complete}   (remaining budget: {budget})")
    if budget <= 0 and complete < len(jams):
        print("  -> re-run to finish caching.\n")
    if complete == 0:
        return

    def mean(rows):
        P = [_score(r, p) for r, p in rows]
        return tuple(np.mean([x[i] for x in P]) for i in range(3))

    print(f"\n{'variant':<16} {'detector':<10} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 52)
    for v in VARIANTS:
        for det in ("kong", "ens"):
            p, r, f = mean(scored[v][det])
            name = "Kong" if det == "kong" else "Ensemble"
            print(f"{v:<16} {name:<10} {p:>7.4f} {r:>7.4f} {f:>7.4f}")
        print()


if __name__ == "__main__":
    main()
