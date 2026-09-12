#!/usr/bin/env python3
"""Self-contained BasicPitch fine-tune for Kaggle (or Colab) GPU.

Fine-tunes the ICASSP-2022 BasicPitch SavedModel on GuitarSet (MIT) +
Guitar-TECHS (CC BY 4.0). Standalone — no `aitabs`/repo import — so it runs in a
fresh Kaggle notebook (see KAGGLE_GPU.md). Meant to run **alongside** the Kong
PCEN retrain on Colab (KAGGLE lets a linked Colab Pro account use separate GPU
hours), so both heavy jobs run at once without competing.

Fixes over the repo's earlier GuitarSet-only fine-tune (which stalled at 0.808):
  * Gaussian-blurred onset targets (a one-hot onset at ~86 fps under-trains the
    onset head → hurts precision).
  * held-out validation split by clip + keep-best-epoch (stops overfit of ~150
    clips).
  * overlapping windows (stride = --stride-frac of a window → more data).
  * Guitar-TECHS mixed in, domain-balanced 50/50 per epoch at the clip level.

Loss supervises the note + onset heads (BCE, onset upweighted); the contour
layers upstream still receive gradients through the note head. Saves a SavedModel
that `basic_pitch.inference.predict` / `--detector finetuned` load unchanged.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from pathlib import Path

import numpy as np

# ── BasicPitch model I/O constants (serving_default signature) ─────────────────
_SR = 22_050
_WINDOW_SAMPLES = 43_844
_WINDOW_FRAMES = 172
_FPS = _SR * _WINDOW_FRAMES / _WINDOW_SAMPLES   # ≈ 86.47 fps
_MIDI_OFFSET = 21                                # A0 = output bin 0
_N_PITCHES = 88
_ONSET_WEIGHT = 5.0


# ── data loading (inlined so the script is standalone) ─────────────────────────

def _iter_jams_observations(data):
    if isinstance(data, list):
        return [o for o in data if isinstance(o, dict)]
    if isinstance(data, dict) and "value" in data:
        t, d, v = data.get("time", []), data.get("duration", []), data.get("value", [])
        return [{"time": t[i] if i < len(t) else 0.0,
                 "duration": d[i] if i < len(d) else 0.0, "value": v[i]}
                for i in range(len(v))]
    return []


def load_guitarset_clips(root):
    """[{audio_path, notes:[{start,end,pitch_midi}]}] for GuitarSet _solo clips."""
    root = Path(root)
    anno, audio = root / "annotation", root / "audio"
    clips = []
    for jp in sorted(anno.glob("*.jams")):
        if "_solo" not in jp.stem:
            continue
        wav = audio / f"{jp.stem}_mic.wav"
        if not wav.is_file():
            continue
        jam = json.loads(jp.read_text(encoding="utf-8"))
        notes = []
        for ann in jam.get("annotations", []):
            if ann.get("namespace") != "note_midi":
                continue
            for o in _iter_jams_observations(ann.get("data")):
                try:
                    s = float(o["time"]); dur = float(o.get("duration") or 0.0)
                    p = int(round(float(o["value"])))
                except (KeyError, TypeError, ValueError):
                    continue
                notes.append({"start": s, "end": s + dur, "pitch_midi": p})
        if notes:
            clips.append({"audio_path": str(wav), "notes": notes})
    return clips


def load_guitar_techs_clips(root, audio_kind="directinput"):
    """[{audio_path, notes}] from Guitar-TECHS per-string MIDI + DI audio."""
    import pretty_midi

    root = Path(root)
    clips = []
    for midi_path in sorted(root.glob("*/midi/*.mid")):
        content = midi_path.parent.parent
        wav = next(iter((content / "audio" / audio_kind).glob("*.wav")), None)
        if wav is None or not wav.is_file():
            continue
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = []
        for inst in pm.instruments[:6]:
            for n in inst.notes:
                notes.append({"start": float(n.start), "end": float(n.end),
                              "pitch_midi": int(n.pitch)})
        if notes:
            clips.append({"audio_path": str(wav), "notes": notes})
    return clips


def _load_audio(path):
    import librosa
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != _SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_SR)
    return y.astype(np.float32)


# ── target construction (with the fixes) ───────────────────────────────────────

def _window_labels(notes, t_start, onset_blur):
    """note_gt, onset_gt for one window. Onset is Gaussian-blurred in time."""
    note_gt = np.zeros((_WINDOW_FRAMES, _N_PITCHES), dtype=np.float32)
    onset_gt = np.zeros((_WINDOW_FRAMES, _N_PITCHES), dtype=np.float32)
    t_end = t_start + _WINDOW_SAMPLES / _SR
    for n in notes:
        if n["end"] <= t_start or n["start"] >= t_end:
            continue
        b = int(round(n["pitch_midi"])) - _MIDI_OFFSET
        if not (0 <= b < _N_PITCHES):
            continue
        ns = max(n["start"], t_start) - t_start
        ne = min(n["end"], t_end) - t_start
        fs = max(0, min(int(round(ns * _FPS)), _WINDOW_FRAMES - 1))
        fe = max(0, min(int(round(ne * _FPS)), _WINDOW_FRAMES))
        if fe > fs:
            note_gt[fs:fe, b] = 1.0
        if n["start"] >= t_start and fs < _WINDOW_FRAMES:
            onset_gt[fs, b] = 1.0
    if onset_blur > 0:
        from scipy.ndimage import gaussian_filter1d
        onset_gt = gaussian_filter1d(onset_gt, sigma=onset_blur, axis=0)
        peak = onset_gt.max()
        if peak > 0:
            onset_gt = onset_gt / peak            # keep the onset frame at ~1.0
    return note_gt, onset_gt


def _clip_windows(audio_path, notes, stride_frac, onset_blur, augment_rng=None):
    y = _load_audio(audio_path)
    if augment_rng is not None:
        db = augment_rng.uniform(-3, 3)
        y = y * (10.0 ** (db / 20.0))
        snr = augment_rng.uniform(20, 40)
        p = float(np.mean(y ** 2)) or 1e-9
        y = y + augment_rng.standard_normal(len(y)).astype(np.float32) * np.sqrt(p / 10 ** (snr / 10))
    stride = max(1, int(_WINDOW_SAMPLES * stride_frac))
    out = []
    pos = 0
    while pos + _WINDOW_SAMPLES <= len(y):
        chunk = y[pos:pos + _WINDOW_SAMPLES, None]
        ng, og = _window_labels(notes, pos / _SR, onset_blur)
        out.append((chunk, ng, og))
        pos += stride
    return out


# ── training ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Self-contained BasicPitch fine-tune (Kaggle/Colab GPU)")
    ap.add_argument("--guitarset-dir", required=True)
    ap.add_argument("--guitar-techs-dir", default=None)
    ap.add_argument("--out", default="basicpitch_guitar_ft")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--stride-frac", type=float, default=0.5, help="window overlap (0.5 = 50%)")
    ap.add_argument("--onset-blur", type=float, default=1.5, help="Gaussian sigma (frames) on onset target")
    ap.add_argument("--no-freeze", action="store_true", help="train the first conv too")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import tensorflow as tf
    from basic_pitch import ICASSP_2022_MODEL_PATH

    print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    rng = np.random.default_rng(args.seed)

    gs = load_guitarset_clips(args.guitarset_dir)
    gt = load_guitar_techs_clips(args.guitar_techs_dir) if args.guitar_techs_dir else []
    if args.max_clips:
        gs, gt = gs[:args.max_clips], gt[:args.max_clips]
    print(f"GuitarSet clips: {len(gs)}   Guitar-TECHS clips: {len(gt)}", flush=True)
    if not gs and not gt:
        raise SystemExit("No clips found — check --guitarset-dir / --guitar-techs-dir layout")

    def split(clips):
        idx = rng.permutation(len(clips))
        n_val = int(len(clips) * args.val_frac)
        val = [clips[i] for i in idx[:n_val]]
        train = [clips[i] for i in idx[n_val:]]
        return train, val
    gs_tr, gs_val = split(gs)
    gt_tr, gt_val = split(gt) if gt else ([], [])
    val_clips = gs_val + gt_val
    print(f"train: {len(gs_tr)} GS + {len(gt_tr)} GT   val: {len(val_clips)}", flush=True)

    model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
    keep, skip = ("kernel", "bias", "gamma", "beta"), ("moving",)
    skip_layer = () if args.no_freeze else ("conv2d_1",)
    t_vars = [v for v in model.variables
              if any(k in v.name for k in keep)
              and not any(s in v.name for s in skip)
              and not any(s in v.name for s in skip_layer)]
    print(f"Trainable tensors: {len(t_vars)}", flush=True)
    opt = tf.keras.optimizers.Adam(args.lr)
    bce = tf.keras.losses.BinaryCrossentropy()

    def loss_on(audio_b, note_b, onset_b, train):
        with tf.GradientTape() as tape:
            for v in t_vars:
                tape.watch(v)
            out = model(audio_b)
            loss = bce(note_b, out["note"]) + _ONSET_WEIGHT * bce(onset_b, out["onset"])
        if train:
            grads = tape.gradient(loss, t_vars)
            gv = [(tf.clip_by_norm(g, 1.0), v) for g, v in zip(grads, t_vars) if g is not None]
            opt.apply_gradients(gv)
        return float(loss)

    def run_batches(window_iter, train):
        ba, bn, bo = [], [], []
        tot, nb = 0.0, 0
        for chunk, ng, og in window_iter:
            ba.append(chunk); bn.append(ng); bo.append(og)
            if len(ba) >= args.batch:
                tot += loss_on(tf.constant(np.stack(ba), tf.float32),
                               tf.constant(np.stack(bn), tf.float32),
                               tf.constant(np.stack(bo), tf.float32), train)
                nb += 1; ba, bn, bo = [], [], []
        if ba:
            tot += loss_on(tf.constant(np.stack(ba), tf.float32),
                           tf.constant(np.stack(bn), tf.float32),
                           tf.constant(np.stack(bo), tf.float32), train)
            nb += 1
        return tot / max(1, nb)

    def domain_balanced_order():
        """Interleave GS and GT clips 1:1 per epoch (cycle the smaller pool)."""
        a = list(gs_tr); rng.shuffle(a)
        if not gt_tr:
            return a
        b = list(gt_tr); rng.shuffle(b)
        pairs = zip(a, itertools.cycle(b)) if len(a) >= len(b) else zip(itertools.cycle(a), b)
        order = []
        for x, y in pairs:
            order += [x, y]
        return order

    def windows_for(clips, augment):
        for c in clips:
            aug = rng if (augment and not args.no_augment) else None
            ws = _clip_windows(c["audio_path"], c["notes"], args.stride_frac, args.onset_blur, aug)
            order = rng.permutation(len(ws))
            for i in order:
                yield ws[int(i)]

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_batches(windows_for(domain_balanced_order(), augment=True), train=True)
        val_loss = run_batches(windows_for(val_clips, augment=False), train=False) if val_clips else tr_loss
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            tf.saved_model.save(model, str(out_dir), signatures=model.signatures)
            tag = "  *best -> saved"
        print(f"epoch {epoch}/{args.epochs}  train={tr_loss:.4f}  val={val_loss:.4f}{tag}", flush=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nDone. Best val={best_val:.4f}. Model -> {out_dir}", flush=True)
    print("Download the SavedModel dir, then locally:")
    print(f"  python scripts/calibrate_detector.py --detector finetuned --model-path {out_dir} --max-clips 30")
    print(f"  python scripts/eval_ensemble.py --max-clips 16 --bp-budget 0   # (point BP at the FT model)")


if __name__ == "__main__":
    main()
