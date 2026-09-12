#!/usr/bin/env python3
"""Fine-tune the Kong (2021) piano transcription model on GuitarSet.

Methodology follows Riley et al. (ICASSP 2024):
  - lr=1e-5, Adam, batch=4 (via gradient accumulation), lr_decay=0.9/10K steps
  - Full model fine-tuning (no layer freezing)
  - GuitarSet split: player 00 for validation, players 01-05 for training
  - Augmentation: pitch shift +-2 semitones, gain jitter, noise, soft-clip distortion

Expected outcome after 10K steps (~15h on CPU):
  - Pitch F1 on GuitarSet: ~87-91% (vs calibrated BasicPitch 81%)
  - Fine-tuning is required — zero-shot Kong on guitar is NOT better than BasicPitch

Usage::

    python scripts/finetune_kong.py
    python scripts/finetune_kong.py --steps 5000 --save-every 500
    python scripts/finetune_kong.py --checkpoint-dir data/models/kong_guitarset_ft --no-augment
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SR = 16000        # Kong model sample rate
_FPS = 100         # Kong frames per second (10ms resolution)
_CLASSES = 88      # A0 (MIDI 21) to C8 (MIDI 108)
_MIDI_MIN = 21     # MIDI 21 = A0

# Guitar MIDI range E2-E6
_GUITAR_MIDI_MIN = 40
_GUITAR_MIDI_MAX = 88

# Training hyperparameters (Riley et al.)
_ONSET_WEIGHT = 5.0
_OFFSET_WEIGHT = 2.0


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def notes_to_roll(
    notes: list[dict],
    n_frames: int,
    fps: int = _FPS,
    onset_sigma: float = 1.0,
) -> dict[str, np.ndarray]:
    """Convert note dicts to frame-level training targets.

    Args:
        notes: list of {start, end, pitch_midi} dicts (GuitarSet JAMS output)
        n_frames: number of output frames
        fps: frames per second
        onset_sigma: Gaussian sigma for onset/offset smoothing (frames)

    Returns:
        dict with keys: frame, onset, offset, velocity (all np.ndarray of shape (n_frames, 88))
    """
    frame = np.zeros((n_frames, _CLASSES), dtype=np.float32)
    onset = np.zeros((n_frames, _CLASSES), dtype=np.float32)
    offset = np.zeros((n_frames, _CLASSES), dtype=np.float32)
    velocity = np.zeros((n_frames, _CLASSES), dtype=np.float32)

    for n in notes:
        midi = int(round(n["pitch_midi"]))
        if not (_GUITAR_MIDI_MIN <= midi <= _GUITAR_MIDI_MAX):
            continue
        cls = midi - _MIDI_MIN
        onset_frame = int(round(float(n["start"]) * fps))
        offset_frame = int(round(float(n["end"]) * fps))
        onset_frame = max(0, min(onset_frame, n_frames - 1))
        offset_frame = max(onset_frame + 1, min(offset_frame, n_frames))

        # Frame: 1 during note duration
        frame[onset_frame:offset_frame, cls] = 1.0

        # Onset/offset: Gaussian kernel (σ=1 frame by default)
        for delta in range(-3, 4):
            t = onset_frame + delta
            if 0 <= t < n_frames:
                onset[t, cls] = max(onset[t, cls], np.exp(-0.5 * (delta / onset_sigma) ** 2))
            t = offset_frame + delta
            if 0 <= t < n_frames:
                offset[t, cls] = max(offset[t, cls], np.exp(-0.5 * (delta / onset_sigma) ** 2))

        # Velocity: fixed (GuitarSet has no MIDI velocity)
        velocity[onset_frame:offset_frame, cls] = 0.63

    return {"frame": frame, "onset": onset, "offset": offset, "velocity": velocity}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_clips(guitarset_dir: Path, split: str) -> list[dict]:
    """Load GuitarSet mic clips for the given split.

    split: "train" = players 01-05, "val" = player 00
    """
    from aitabs.eval.guitarset_import import load_guitarset_notes
    import soundfile as sf
    import librosa

    audio_dir = guitarset_dir / "audio"
    anno_dir = guitarset_dir / "annotation"

    clips = []
    for jams_path in sorted(anno_dir.glob("*.jams")):
        player_id = jams_path.stem.split("_")[0]
        if split == "val" and player_id != "00":
            continue
        if split == "train" and player_id == "00":
            continue
        wav = audio_dir / f"{jams_path.stem}_mic.wav"
        if not wav.exists():
            continue
        ref_notes, _ = load_guitarset_notes(jams_path)
        clips.append({"path": str(wav), "notes": ref_notes, "stem": jams_path.stem})

    return clips


def make_segments(clip: dict, seg_sec: float = 3.0, overlap: float = 0.5) -> list[dict]:
    """Chop a clip into overlapping segments with their note labels."""
    import soundfile as sf
    import librosa

    y, sr = sf.read(clip["path"], dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != _SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_SR)

    seg_samples = int(seg_sec * _SR)
    hop_samples = int(seg_samples * (1 - overlap))
    n_frames = int(seg_sec * _FPS)

    segments = []
    start_sample = 0
    while start_sample + seg_samples <= len(y):
        start_sec = start_sample / _SR
        end_sec = start_sec + seg_sec
        # Filter notes to this window
        seg_notes = [
            {**n, "start": n["start"] - start_sec, "end": n["end"] - start_sec}
            for n in clip["notes"]
            if n["end"] > start_sec and n["start"] < end_sec
        ]
        audio_seg = y[start_sample: start_sample + seg_samples]
        roll = notes_to_roll(seg_notes, n_frames)
        segments.append({"audio": audio_seg, "roll": roll, "notes": seg_notes})
        start_sample += hop_samples

    return segments


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_loss(
    output: dict,
    target: dict[str, np.ndarray],
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """BCE loss on frame + onset + offset outputs.

    Trims target to match model output length (Kong may output T+1 frames due
    to how it pads audio internally).

    ``pos_weight`` upweights the positive (note-present) class within each BCE
    term.  The targets are dominated by zeros — across 88 pitch classes only a
    few are active per frame — so plain BCE is minimised by predicting ~0
    everywhere: it nails the silent majority (loss -> ~0.02) while leaving the
    model underconfident on real notes (muted outputs, calibration optimum
    pinned at the grid floor, F1 stuck ~0.81 on both train and held-out).
    Weighting positives counteracts that imbalance so real notes are pushed
    toward high probability.  pos_weight=1.0 reproduces the original behaviour.
    """
    def _bce(pred: torch.Tensor, label: np.ndarray) -> torch.Tensor:
        # Kong model outputs are already sigmoid-activated probabilities [0,1].
        # Do NOT apply .sigmoid() again — double-sigmoid collapses all outputs
        # to ~0.5 and kills gradients. Clamp for numerical stability.
        n_frames = min(pred.shape[1], label.shape[0])
        p = pred[:, :n_frames].clamp(1e-7, 1 - 1e-7)   # (1, T, 88)
        t = torch.from_numpy(label[:n_frames]).unsqueeze(0).to(pred.device)
        if pos_weight == 1.0:
            return F.binary_cross_entropy(p, t)
        # Per-element weight = 1 + (pos_weight-1)*t: scales the loss by pos_weight
        # where the target is fully positive (t=1), by 1 where fully negative
        # (t=0), and smoothly in between for the Gaussian onset/offset targets.
        w = 1.0 + (pos_weight - 1.0) * t
        return F.binary_cross_entropy(p, t, weight=w)

    loss = _bce(output["frame_output"], target["frame"])
    loss = loss + _ONSET_WEIGHT * _bce(output["reg_onset_output"], target["onset"])
    loss = loss + _OFFSET_WEIGHT * _bce(output["reg_offset_output"], target["offset"])
    return loss


def _nested_state(model) -> dict:
    """Convert flat Note_pedal state_dict to piano_transcription_inference format.

    PianoTranscription expects {'note_model': {stripped_keys}, 'pedal_model': {stripped_keys}}
    but PyTorch's default state_dict() returns flat keys like 'note_model.layer.weight'.
    """
    flat = model.state_dict()
    nested: dict = {"note_model": {}, "pedal_model": {}}
    for k, v in flat.items():
        for prefix in ("note_model", "pedal_model"):
            if k.startswith(prefix + "."):
                nested[prefix][k[len(prefix) + 1:]] = v
                break
    return nested


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Kong model on GuitarSet")
    parser.add_argument("--guitarset-dir", type=Path, default=Path("data/eval/guitarset"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/models/kong_guitarset_ft"))
    parser.add_argument("--steps", type=int, default=5000, help="Optimizer update steps (~6s/step on CPU: 1000=1.7h, 5000=8.7h)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-decay-every", type=int, default=10000)
    parser.add_argument("--lr-decay-factor", type=float, default=0.9)
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps (effective batch size)")
    parser.add_argument("--segment-sec", type=float, default=3.0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation (baseline comparison)")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a fine-tuned checkpoint (.pth)")
    parser.add_argument("--resume-step", type=int, default=0, help="Step count to resume from (for logging)")
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="Positive-class weight in the BCE loss to counter the "
                             "zero-dominated targets (fixes muted outputs / underfit). "
                             "1.0 = original behaviour; try 5-10.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from piano_transcription_inference.models import Note_pedal
    import piano_transcription_inference.config as cfg

    model = Note_pedal(frames_per_second=cfg.frames_per_second, classes_num=cfg.classes_num)

    if args.resume:
        print(f"Resuming from: {args.resume}")
        state = torch.load(str(args.resume), map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
    else:
        from pathlib import Path as _P
        ckpt_path = str(_P.home() / "piano_transcription_inference_data"
                        / "note_F1=0.9677_pedal_F1=0.9186.pth")
        print(f"Loading Kong checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Load GuitarSet clips ──────────────────────────────────────────────────
    print("Loading GuitarSet clips ...")
    train_clips = load_clips(args.guitarset_dir, "train")
    val_clips = load_clips(args.guitarset_dir, "val")
    print(f"  train: {len(train_clips)} clips  val: {len(val_clips)} clips")

    print("Pre-loading training segments ...")
    train_segs = []
    for c in train_clips:
        train_segs.extend(make_segments(c, seg_sec=args.segment_sec))
    print(f"  {len(train_segs)} training segments")

    # ── Training loop ─────────────────────────────────────────────────────────
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.checkpoint_dir / "training_log.jsonl"

    step = args.resume_step
    accum_loss = 0.0
    optimizer.zero_grad()

    print(f"Training for {args.steps} steps (accum_steps={args.accum_steps}, "
          f"lr={args.lr}, pos_weight={args.pos_weight}) ...")
    indices = list(range(len(train_segs)))
    random.shuffle(indices)
    seg_idx = 0

    use_augment = not args.no_augment
    if use_augment:
        from aitabs.pipeline.audio.augment import augment_kong
        print("Augmentation: pitch shift +-2st, gain, noise, distortion (30%)")
    else:
        print("Augmentation: disabled")

    while step < args.steps:
        if seg_idx >= len(indices):
            random.shuffle(indices)
            seg_idx = 0

        seg = train_segs[indices[seg_idx]]
        seg_idx += 1

        audio = seg["audio"]
        roll = seg["roll"]

        if use_augment:
            aug_audio, aug_notes = augment_kong(
                audio, _SR, seg["notes"], rng=np.random.default_rng()
            )
            n_frames = int(aug_audio.shape[0] * _FPS / _SR)
            roll = notes_to_roll(aug_notes, n_frames)
            audio = aug_audio

        x = torch.from_numpy(audio[None]).to(device)  # (1, samples)
        out = model.note_model(x)
        loss = compute_loss(out, roll, pos_weight=args.pos_weight)
        (loss / args.accum_steps).backward()
        accum_loss += loss.item() / args.accum_steps

        if (seg_idx % args.accum_steps) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            # Learning rate decay
            if step % args.lr_decay_every == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] *= args.lr_decay_factor
                print(f"  lr -> {optimizer.param_groups[0]['lr']:.2e}")

            if step % 100 == 0:
                print(f"  step {step}/{args.steps}  loss={accum_loss/100:.4f}", flush=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "loss": round(accum_loss / 100, 4)}) + "\n")
                accum_loss = 0.0

            if step % args.save_every == 0 or step == args.steps:
                ckpt_out = args.checkpoint_dir / f"kong_ft_step{step}.pth"
                torch.save({"step": step, "model": _nested_state(model)}, ckpt_out)
                print(f"  Saved -> {ckpt_out}")

    # Save final checkpoint in piano_transcription_inference format
    final_path = args.checkpoint_dir / "kong_ft_final.pth"
    torch.save({"model": _nested_state(model)}, final_path)
    print(f"\nFinal checkpoint -> {final_path}")
    print(f"Use with: --detector kong --model-path {final_path}")


if __name__ == "__main__":
    main()
