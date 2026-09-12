"""Fine-tune a BasicPitch SavedModel on GuitarSet solo data.

BasicPitch (ICASSP 2022) processes fixed 43 844-sample windows of 22 050 Hz
mono audio and returns per-window onset / note predictions at 172 frames × 88
MIDI bins.  We fine-tune on GuitarSet solo mic recordings using binary
cross-entropy loss on the note and onset heads.

Usage via CLI::

    python scripts/finetune_basicpitch.py

The fine-tuned model is saved as a SavedModel compatible with
``basic_pitch.inference.predict``, so the existing eval harness works unchanged
by pointing ``--detector finetuned`` at the saved path.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── model I/O constants (from serving_default signature inspection) ────────────
_SR: int = 22_050
_WINDOW_SAMPLES: int = 43_844       # samples per window (shape (None, 43844, 1))
_WINDOW_FRAMES: int = 172           # output frames per window
_FPS: float = _SR * _WINDOW_FRAMES / _WINDOW_SAMPLES   # ≈ 86.47 frames/s
_MIDI_OFFSET: int = 21              # A0 — bin 0 of the 88-pitch output
_N_PITCHES: int = 88                # A0 … C8

_ONSET_WEIGHT: float = 5.0          # onset frames are sparse → upweight loss


# ── internal helpers ──────────────────────────────────────────────────────────

def _trainable_vars(model, *, freeze_first_conv: bool = True):
    """Kernel / bias / BN-scale vars, optionally skipping the first conv."""
    keep = ("kernel", "bias", "gamma", "beta")
    skip = ("moving",)
    skip_layer = ("conv2d_1",) if freeze_first_conv else ()
    return [
        v for v in model.variables
        if any(t in v.name for t in keep)
        and not any(t in v.name for t in skip)
        and not any(t in v.name for t in skip_layer)
    ]


def _load_audio(audio_path: str, sr: int = _SR) -> np.ndarray:
    import soundfile as sf
    import librosa
    y, file_sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def _window_labels(
    notes: list[dict],
    t_start: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary note / onset labels for one _WINDOW_SAMPLES window.

    Returns:
        note_gt  : float32 (_WINDOW_FRAMES, _N_PITCHES) — 1 while note is active
        onset_gt : float32 (_WINDOW_FRAMES, _N_PITCHES) — 1 only at first frame
    """
    note_gt = np.zeros((_WINDOW_FRAMES, _N_PITCHES), dtype=np.float32)
    onset_gt = np.zeros((_WINDOW_FRAMES, _N_PITCHES), dtype=np.float32)
    t_end = t_start + _WINDOW_SAMPLES / _SR

    for n in notes:
        if n["end"] <= t_start or n["start"] >= t_end:
            continue
        bin_idx = int(round(n["pitch_midi"])) - _MIDI_OFFSET
        if not (0 <= bin_idx < _N_PITCHES):
            continue
        # Relative times within window → frame indices
        ns = max(n["start"], t_start) - t_start
        ne = min(n["end"], t_end) - t_start
        fs = max(0, min(int(round(ns * _FPS)), _WINDOW_FRAMES - 1))
        fe = max(0, min(int(round(ne * _FPS)), _WINDOW_FRAMES))
        if fe > fs:
            note_gt[fs:fe, bin_idx] = 1.0
        if n["start"] >= t_start and fs < _WINDOW_FRAMES:
            onset_gt[fs, bin_idx] = 1.0

    return note_gt, onset_gt


def _iter_windows(
    y: np.ndarray,
    notes: list[dict],
    stride: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (audio_window, note_gt, onset_gt) triples across a clip."""
    if stride is None:
        stride = _WINDOW_SAMPLES
    windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    pos = 0
    while pos + _WINDOW_SAMPLES <= len(y):
        chunk = y[pos : pos + _WINDOW_SAMPLES, np.newaxis]   # (W, 1)
        note_gt, onset_gt = _window_labels(notes, pos / _SR)
        windows.append((chunk, note_gt, onset_gt))
        pos += stride
    return windows


# ── public API ────────────────────────────────────────────────────────────────

def finetune(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 4,
    freeze_first_conv: bool = True,
    augment: bool = True,
    max_clips: int | None = None,
) -> dict:
    """Fine-tune BasicPitch on the GuitarSet training manifest.

    Args:
        manifest_path: path to ``guitarset_manifest.json``
        out_dir: directory to write the fine-tuned SavedModel and history JSON
        epochs: full passes over the windowed dataset
        lr: Adam learning rate
        batch_size: windows per gradient step
        freeze_first_conv: freeze ``conv2d_1`` (keep audio front-end stable)
        augment: apply gain jitter + noise + high-pass to each clip
        max_clips: cap number of clips (smoke test / quick iteration)

    Returns:
        dict with keys ``loss`` (list, one entry per epoch), ``clips``,
        ``windows``, and ``model_path`` (str).
    """
    import tensorflow as tf
    from aitabs.pipeline.audio.augment import augment as aug_fn
    from basic_pitch import ICASSP_2022_MODEL_PATH

    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = manifest["pairs"]
    if max_clips is not None:
        pairs = pairs[:max_clips]
    log.info("Fine-tuning on %d clips", len(pairs))

    log.info("Loading model ...")
    model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
    t_vars = _trainable_vars(model, freeze_first_conv=freeze_first_conv)
    log.info("Trainable tensors: %d (freeze_first_conv=%s)", len(t_vars), freeze_first_conv)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    bce = tf.keras.losses.BinaryCrossentropy()

    rng = np.random.default_rng(42)

    # Count windows without pre-loading to report progress
    n_windows = sum(
        max(0, (len(_load_audio(p["audio_path"])) - _WINDOW_SAMPLES) // _WINDOW_SAMPLES + 1)
        for p in pairs
    )
    log.info("~%d windows from %d clips", n_windows, len(pairs))
    print(f"  {len(pairs)} clips -> ~{n_windows} windows per epoch", flush=True)

    history: dict = {"loss": [], "clips": len(pairs), "windows": n_windows}

    for epoch in range(1, epochs + 1):
        # Stream clips in a shuffled order; build batches on the fly
        clip_order = np.random.permutation(len(pairs))
        batch_audio: list[np.ndarray] = []
        batch_note: list[np.ndarray] = []
        batch_onset: list[np.ndarray] = []
        epoch_loss = 0.0
        n_batches = 0

        def _flush() -> None:
            nonlocal epoch_loss, n_batches
            audio_b = tf.constant(np.stack(batch_audio), dtype=tf.float32)
            note_b = tf.constant(np.stack(batch_note), dtype=tf.float32)
            onset_b = tf.constant(np.stack(batch_onset), dtype=tf.float32)
            with tf.GradientTape() as tape:
                for v in t_vars:
                    tape.watch(v)
                out = model(audio_b)
                loss = bce(note_b, out["note"]) + _ONSET_WEIGHT * bce(onset_b, out["onset"])
            grads = tape.gradient(loss, t_vars)
            gv = [
                (tf.clip_by_norm(g, 1.0), v)
                for g, v in zip(grads, t_vars)
                if g is not None
            ]
            optimizer.apply_gradients(gv)
            epoch_loss += float(loss)
            n_batches += 1
            batch_audio.clear()
            batch_note.clear()
            batch_onset.clear()

        for clip_idx in clip_order:
            pair = pairs[int(clip_idx)]
            y = _load_audio(pair["audio_path"])
            if augment:
                y = aug_fn(y, _SR, rng=rng)
            windows = _iter_windows(y, pair["notes"])
            # Shuffle windows within this clip
            win_order = np.random.permutation(len(windows))
            for wi in win_order:
                audio_w, note_gt, onset_gt = windows[int(wi)]
                batch_audio.append(audio_w)
                batch_note.append(note_gt)
                batch_onset.append(onset_gt)
                if len(batch_audio) >= batch_size:
                    _flush()

        if batch_audio:  # leftover partial batch
            _flush()

        mean_loss = epoch_loss / max(1, n_batches)
        history["loss"].append(mean_loss)
        msg = f"  epoch {epoch}/{epochs}  loss={mean_loss:.4f}"
        log.info(msg)
        print(msg, flush=True)

        # Checkpoint after every epoch
        ckpt_path = out_dir / "basicpitch_guitarset_ft"
        tf.saved_model.save(model, str(ckpt_path), signatures=model.signatures)
        history["model_path"] = str(ckpt_path)
        (out_dir / "finetune_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        log.info("Checkpoint -> %s", ckpt_path)

    ft_path = out_dir / "basicpitch_guitarset_ft"
    log.info("Done. Final model -> %s", ft_path)
    print(f"Final model -> {ft_path}", flush=True)
    return history
