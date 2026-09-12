#!/usr/bin/env python3
"""Threshold calibration for BasicPitch or Kong detector.

Caches raw model frame predictions once, then sweeps onset_threshold,
frame_threshold, and confidence_threshold without re-running inference for
each combination.  Reports the grid point with highest mean pitch F1.

Usage::

    # Calibrate BasicPitch fine-tuned model (default)
    python scripts/calibrate_detector.py

    # Calibrate the baseline BasicPitch model for comparison
    python scripts/calibrate_detector.py --model-path baseline

    # Calibrate Kong model
    python scripts/calibrate_detector.py --detector kong

    # Full 180-clip grid (slower)
    python scripts/calibrate_detector.py --max-clips 180
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── grid ─────────────────────────────────────────────────────────────────────
# Fine-tuned Kong is high-recall and calibrates to low onset/frame thresholds;
# the 5-clip and 30-clip sweeps both pinned the optimum near onset=0.2/frame=0.1.
# Ultra-low thresholds (onset/frame=0.05) made the RegressionPostProcessor
# assemble a pathological number of notes (~25 s/combo) for no F1 gain, so the
# grid floors at 0.1.  Range mirrors the BasicPitch baseline grid for a fair,
# apples-to-apples calibration.
_ONSET_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_FRAME_GRID = [0.1, 0.2, 0.3]
_CONF_GRID = [0.2, 0.42, 0.5]
_MIN_NOTE_LEN_MS = 58.0


def _min_note_frames(ms: float) -> int:
    """Convert minimum note length in ms to annotation frames."""
    from basic_pitch.inference import FFT_HOP, AUDIO_SAMPLE_RATE
    return max(1, int(ms / 1000 * AUDIO_SAMPLE_RATE / FFT_HOP))


def _run_inference_cached(audio_path: str, model_path: str) -> dict:
    """Return raw model output dict (onset, note, contour arrays)."""
    from basic_pitch.inference import run_inference
    return run_inference(audio_path, model_path)


def _notes_from_output(
    raw_output: dict,
    onset_thresh: float,
    frame_thresh: float,
    confidence_thresh: float,
    min_freq: float | None = 82.0,
    max_freq: float | None = 1400.0,
    melodia_trick: bool = True,
) -> list[dict]:
    """Apply thresholds to cached model output and return NoteEvent list."""
    from basic_pitch.note_creation import model_output_to_notes
    _midi, note_events = model_output_to_notes(
        raw_output,
        onset_thresh=onset_thresh,
        frame_thresh=frame_thresh,
        min_note_len=_min_note_frames(_MIN_NOTE_LEN_MS),
        min_freq=min_freq,
        max_freq=max_freq,
        melodia_trick=melodia_trick,
    )
    return [
        {"start": float(s), "end": float(e), "pitch_midi": int(p), "confidence": float(c)}
        for s, e, p, c, *_ in note_events
        if c >= confidence_thresh
    ]


def _pitch_f1(ref_notes: list[dict], pred_notes: list[dict], tol: float = 0.05) -> tuple[float, float, float]:
    """Pitch-only P/R/F1 with onset tolerance.  Ignores string/fret."""
    from aitabs.eval.metrics import EvalNote, compare_note_lists
    ref = [EvalNote(start=n["start"], end=n["end"], pitch_midi=n["pitch_midi"],
                    string=n.get("string", -1), fret=n.get("fret", -1),
                    confidence=n.get("confidence", 1.0)) for n in ref_notes]
    pred = [EvalNote(start=n["start"], end=n["end"], pitch_midi=n["pitch_midi"],
                     string=-1, fret=-1, confidence=n.get("confidence", 1.0)) for n in pred_notes]
    scores = compare_note_lists(ref, pred, onset_tolerance_sec=tol)
    return scores.pitch_precision, scores.pitch_recall, scores.pitch_f1


_KONG_TRANSCRIPTOR_CACHE: dict = {}

# Disk cache for raw Kong inference output.  Inference is the expensive part of
# calibration (~30 s/clip on CPU); the threshold sweep is cheap.  Background jobs
# in this environment get killed frequently, so persist each clip's output_dict
# to disk keyed by (checkpoint, clip, checkpoint mtime).  A restart then reloads
# finished clips instead of re-running inference.  Set via --cache-dir.
_KONG_DISK_CACHE_DIR: "Path | None" = None
# Set via --pcen: the checkpoint has a trainable PCEN front-end, which must be
# attached before loading (RESEARCH_PCEN.md).
_KONG_PCEN: bool = False


def _kong_cache_key(audio_path: str, checkpoint_path: str) -> "Path | None":
    if _KONG_DISK_CACHE_DIR is None:
        return None
    import hashlib
    import os
    try:
        mtime = int(os.path.getmtime(checkpoint_path))
    except OSError:
        mtime = 0
    raw = f"{os.path.abspath(checkpoint_path)}|{mtime}|{os.path.abspath(audio_path)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return _KONG_DISK_CACHE_DIR / f"{digest}.npz"


def _run_kong_inference_cached(audio_path: str, checkpoint_path: str) -> dict:
    """Return Kong raw output_dict (onset/offset/frame/velocity arrays).

    The PianoTranscription model is loaded once per checkpoint path and cached
    at module level to avoid repeated 165 MB checkpoint loads.  When a disk cache
    dir is configured (--cache-dir), the per-clip output is also persisted so a
    killed-then-restarted calibration skips already-computed inference.
    """
    import numpy as np
    import torch
    import soundfile as sf
    import librosa
    from piano_transcription_inference import PianoTranscription
    from piano_transcription_inference.config import sample_rate as KONG_SR
    from piano_transcription_inference.inference import forward

    cache_path = _kong_cache_key(audio_path, checkpoint_path)
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as data:
            return {k: data[k] for k in data.files}

    y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != KONG_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=KONG_SR)

    if checkpoint_path not in _KONG_TRANSCRIPTOR_CACHE:
        if _KONG_PCEN:
            # Build on the base pretrained model, attach PCEN, then load the PCEN
            # weights — keys only match once PCEN is attached (RESEARCH_PCEN.md).
            from aitabs.pipeline.audio.pcen import attach_pcen_frontend
            tr = PianoTranscription(model_type="Note_pedal", checkpoint_path=None,
                                    device=torch.device("cpu"))
            attach_pcen_frontend(tr.model.note_model)
            st = torch.load(checkpoint_path, map_location="cpu")
            tr.model.load_state_dict(st["model"])
            tr.model.eval()
            _KONG_TRANSCRIPTOR_CACHE[checkpoint_path] = tr
        else:
            _KONG_TRANSCRIPTOR_CACHE[checkpoint_path] = PianoTranscription(
                model_type="Note_pedal",
                checkpoint_path=checkpoint_path,
                device=torch.device("cpu"),
            )
    transcriptor = _KONG_TRANSCRIPTOR_CACHE[checkpoint_path]

    audio = y[None, :]
    audio_len = audio.shape[1]
    seg = transcriptor.segment_samples
    pad = int(np.ceil(audio_len / seg)) * seg - audio_len
    audio = np.concatenate((audio, np.zeros((1, pad))), axis=1)
    segments = transcriptor.enframe(audio, seg)
    output_dict = forward(transcriptor.model, segments, batch_size=1)
    for key in output_dict.keys():
        output_dict[key] = transcriptor.deframe(output_dict[key])[:audio_len]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".npz.tmp")
        # Pass an open handle so np.savez writes to exactly this path (a path
        # string not ending in ".npz" would get ".npz" appended, breaking the
        # rename).  Then atomically replace so a kill mid-write can't leave a
        # partial .npz that a later run would treat as valid.
        with open(tmp, "wb") as fh:
            np.savez(fh, **output_dict)
        tmp.replace(cache_path)
    return output_dict


def _kong_notes_from_output(
    raw_output: dict,
    onset_thresh: float,
    frame_thresh: float,
    confidence_thresh: float,
    min_midi: int = 40,
    max_midi: int = 88,
    min_len_sec: float = 0.058,
) -> list[dict]:
    """Apply thresholds to cached Kong output_dict and return note list."""
    from piano_transcription_inference.utilities import RegressionPostProcessor
    pp = RegressionPostProcessor(
        frames_per_second=100,
        classes_num=88,
        onset_threshold=onset_thresh,
        offset_threshold=0.3,
        frame_threshold=frame_thresh,
        pedal_offset_threshold=0.2,
    )
    note_events, _ = pp.output_dict_to_midi_events(raw_output)
    return [
        {"start": float(ev["onset_time"]), "end": float(ev["offset_time"]),
         "pitch_midi": int(ev["midi_note"]), "confidence": float(ev["velocity"]) / 127.0}
        for ev in note_events
        if float(ev["velocity"]) / 127.0 >= confidence_thresh
        and min_midi <= int(ev["midi_note"]) <= max_midi
        and float(ev["offset_time"]) - float(ev["onset_time"]) >= min_len_sec
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold calibration for BasicPitch or Kong detector")
    parser.add_argument(
        "--detector",
        choices=["basicpitch", "kong"],
        default="basicpitch",
        help="Which detector to calibrate (default: basicpitch)",
    )
    parser.add_argument(
        "--model-path",
        default="data/models/basicpitch_guitarset_ft",
        help="Path to SavedModel or 'baseline' (basicpitch), or Kong .pth checkpoint path (kong)",
    )
    parser.add_argument(
        "--guitarset-dir",
        type=Path,
        default=Path("data/eval/guitarset"),
        help="Prepared GuitarSet directory (default: data/eval/guitarset)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=30,
        help="Clips to use for calibration (default: 30)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write calibration JSON to this path (optional)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Persist per-clip Kong inference output here so a killed-then-"
             "restarted calibration skips finished clips (kong detector only)",
    )
    parser.add_argument(
        "--pcen",
        action="store_true",
        help="Kong checkpoint has a trainable PCEN front-end (RESEARCH_PCEN.md)",
    )
    args = parser.parse_args()

    if args.cache_dir is not None:
        global _KONG_DISK_CACHE_DIR
        _KONG_DISK_CACHE_DIR = args.cache_dir
    if args.pcen:
        global _KONG_PCEN
        _KONG_PCEN = args.pcen

    # ── Resolve model path / label ───────────────────────────────────────────
    is_kong = args.detector == "kong"
    if is_kong:
        from pathlib import Path as _P
        kong_ckpt = args.model_path if args.model_path != "data/models/basicpitch_guitarset_ft" else None
        if kong_ckpt is None:
            kong_ckpt = str(_P.home() / "piano_transcription_inference_data"
                            / "note_F1=0.9677_pedal_F1=0.9186.pth")
        model_path = kong_ckpt
        label = "kong"
    elif args.model_path == "baseline":
        from basic_pitch import ICASSP_2022_MODEL_PATH
        model_path = str(ICASSP_2022_MODEL_PATH)
        label = "baseline"
    else:
        model_path = args.model_path
        label = Path(model_path).name

    # ── Load GuitarSet ground truth ──────────────────────────────────────────
    from aitabs.eval.guitarset_import import load_guitarset_notes
    anno_dir = args.guitarset_dir / "annotation"
    audio_dir = args.guitarset_dir / "audio"
    jams_files = sorted(anno_dir.glob("*.jams"))
    if not jams_files:
        print(f"No .jams files under {anno_dir}", file=sys.stderr)
        sys.exit(1)
    if args.max_clips:
        jams_files = jams_files[: args.max_clips]

    clips = []
    for jf in jams_files:
        clip_id = jf.stem
        wav = audio_dir / f"{clip_id}_mic.wav"
        if not wav.is_file():
            continue
        ref_notes, _ = load_guitarset_notes(jf)
        clips.append({"clip_id": clip_id, "audio_path": str(wav), "ref_notes": ref_notes})

    print(f"Calibrating '{label}' on {len(clips)} clips", flush=True)
    print(f"Grid: {len(_ONSET_GRID)} onset x {len(_FRAME_GRID)} frame x {len(_CONF_GRID)} conf"
          f" = {len(_ONSET_GRID)*len(_FRAME_GRID)*len(_CONF_GRID)} combinations", flush=True)

    # ── Phase 1: cache raw model output for every clip ───────────────────────
    print("Running model inference (once per clip) ...", flush=True)
    cached: list[dict] = []
    for i, clip in enumerate(clips):
        if is_kong:
            raw = _run_kong_inference_cached(clip["audio_path"], model_path)
        else:
            raw = _run_inference_cached(clip["audio_path"], model_path)
        cached.append({"ref_notes": clip["ref_notes"], "raw_output": raw})
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(clips)} clips cached", flush=True)
    print(f"  {len(clips)}/{len(clips)} clips cached", flush=True)

    # ── Phase 2: sweep thresholds ────────────────────────────────────────────
    # The sweep (grid x clips postprocessing) can outlast the ~15-min job-kill
    # window in this environment, so persist per-combo results to a sidecar and
    # skip already-scored combos on restart.  Inference is npz-cached (phase 1),
    # so a resumed run reaches here in seconds and finishes only the remaining
    # combos.
    print("Sweeping thresholds ...", flush=True)
    best_f1 = -1.0
    best_combo: tuple[float, float, float] = (0.5, 0.3, 0.42)
    results: list[dict] = []
    done: set[tuple[float, float, float]] = set()

    progress_path = args.out.with_suffix(".partial.json") if args.out else None
    if progress_path is not None and progress_path.exists():
        try:
            prior = json.loads(progress_path.read_text(encoding="utf-8"))
            results = prior.get("grid", [])
            for r in results:
                key = (r["onset"], r["frame"], r["conf"])
                done.add(key)
                if r["pitch_f1"] > best_f1:
                    best_f1 = r["pitch_f1"]
                    best_combo = key
            print(f"  resumed: {len(done)} combos already scored, best so far F1={best_f1:.4f}", flush=True)
        except (json.JSONDecodeError, KeyError):
            results, done = [], set()

    combos = list(itertools.product(_ONSET_GRID, _FRAME_GRID, _CONF_GRID))
    for idx, (ot, ft, ct) in enumerate(combos):
        if (ot, ft, ct) in done:
            continue
        f1s = []
        for c in cached:
            if is_kong:
                pred = _kong_notes_from_output(c["raw_output"], ot, ft, ct)
            else:
                pred = _notes_from_output(c["raw_output"], ot, ft, ct)
            _, _, f1 = _pitch_f1(c["ref_notes"], pred)
            f1s.append(f1)
        mean_f1 = float(np.mean(f1s))
        results.append({"onset": ot, "frame": ft, "conf": ct, "pitch_f1": round(mean_f1, 4)})
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_combo = (ot, ft, ct)
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(combos)} done  best so far: F1={best_f1:.4f}", flush=True)
            if progress_path is not None:
                tmp = progress_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps({"grid": results}), encoding="utf-8")
                tmp.replace(progress_path)

    if progress_path is not None and progress_path.exists():
        progress_path.unlink()  # sweep finished cleanly; drop the sidecar

    ot, ft, ct = best_combo
    print(f"\n=== Best thresholds for '{label}' ===")
    print(f"  onset_threshold    = {ot}")
    print(f"  frame_threshold    = {ft}")
    print(f"  confidence_threshold = {ct}")
    print(f"  pitch F1 (cal set) = {best_f1:.4f}")
    print(f"\nTop 5 combos:")
    top5 = sorted(results, key=lambda r: -r["pitch_f1"])[:5]
    for r in top5:
        print(f"  onset={r['onset']}  frame={r['frame']}  conf={r['conf']}  F1={r['pitch_f1']:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"model": label, "best": {"onset": ot, "frame": ft, "conf": ct,
                                                  "pitch_f1": round(best_f1, 4)},
                        "grid": results}, indent=2),
            encoding="utf-8",
        )
        print(f"\nCalibration results -> {args.out}")

    detector_flag = f"--detector {args.detector}"
    if is_kong:
        print(f"\nTo eval with best thresholds:")
        print(f"  python scripts/eval_dataset.py data/eval/guitarset --guitarset"
              f" {detector_flag}"
              f" --onset-threshold {ot} --frame-threshold {ft} --confidence-threshold {ct}")
    else:
        print(f"\nTo eval with best thresholds:")
        print(f"  python scripts/eval_dataset.py data/eval/guitarset --guitarset"
              f" --detector finetuned --model-path {args.model_path}"
              f" --onset-threshold {ot} --frame-threshold {ft} --confidence-threshold {ct}")


if __name__ == "__main__":
    main()
