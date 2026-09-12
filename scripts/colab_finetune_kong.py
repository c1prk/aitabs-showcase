#!/usr/bin/env python3
"""Self-contained GPU fine-tuning of the Kong (2021) CRNN on GuitarSet.

Designed to run in Google Colab (or any CUDA box) WITHOUT the aitabs package or
the private repo — everything needed is inlined. Mirrors the recipe validated on
CPU in ``scripts/finetune_kong.py`` but adds:

  * real mini-batching (segments are fixed 3 s, so they stack trivially) — the
    big GPU win over the CPU batch=1 loop;
  * ``--pos-weight`` positive-class BCE weighting, the fix for the zero-dominated
    targets that made plain BCE underfit (loss collapsed to ~0.02, muted outputs,
    held-out F1 stuck ~0.81). Validated on CPU: pos_weight=5 reached 0.830 at
    step 1000 vs 0.813 for the plain 5000-step model;
  * augmentation OFF by default — we are UNDERfitting, so heavy pitch-shift aug
    hurts here (and it is the per-step CPU bottleneck). Turn on with --augment.

Checkpoints are saved in ``piano_transcription_inference`` nested format
({'note_model': {...}, 'pedal_model': {...}}) so they load straight into
``PianoTranscription`` for calibration/eval back on the local machine.

Colab quickstart (see the notebook cells in the handoff message)::

    !pip -q install piano_transcription_inference librosa soundfile
    !python colab_finetune_kong.py \
        --guitarset-dir /content/guitarset \
        --checkpoint-dir /content/drive/MyDrive/kong_posw5_gpu \
        --pos-weight 5 --batch 8 --steps 50000 --save-every 2000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Trainable PCEN front-end (inlined from aitabs/pipeline/audio/pcen.py so this
#    script stays self-contained on Colab; see RESEARCH_PCEN.md) ────────────────

def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class TrainablePCEN(nn.Module):
    """Per-channel energy normalization with learnable alpha/delta/r (per band)."""

    def __init__(self, num_bands, *, s=0.025, alpha=0.8, delta=2.0, r=0.5,
                 eps=1e-6, trainable=True):
        super().__init__()
        self.eps = float(eps)
        self.s = float(s)
        init = lambda v: torch.full((num_bands,), _inv_softplus(v), dtype=torch.float32)
        self._log_alpha = nn.Parameter(init(alpha), requires_grad=trainable)
        self._log_delta = nn.Parameter(init(delta), requires_grad=trainable)
        self._log_r = nn.Parameter(init(r), requires_grad=trainable)

    def _smooth(self, E):
        s = self.s
        m = E[:, :, 0, :]
        outs = [m]
        for t in range(1, E.shape[2]):
            m = (1.0 - s) * m + s * E[:, :, t, :]
            outs.append(m)
        return torch.stack(outs, dim=2)

    def forward(self, E):
        E = torch.clamp(E, min=0.0)
        alpha = F.softplus(self._log_alpha)
        delta = F.softplus(self._log_delta)
        r = F.softplus(self._log_r)
        M = self._smooth(E)
        smooth = (self.eps + M) ** alpha
        return (E / smooth + delta) ** r - delta ** r


class MelPCEN(nn.Module):
    """Mel projection (reusing Kong's frozen melW) then trainable PCEN."""

    def __init__(self, orig_logmel, **pcen_kwargs):
        super().__init__()
        melW = orig_logmel.melW.detach().clone()
        self.register_buffer("melW", melW)
        self.pcen = TrainablePCEN(melW.shape[-1], **pcen_kwargs)

    def forward(self, power_spectrogram):
        mel = torch.matmul(power_spectrogram, self.melW)
        return self.pcen(mel)


def attach_pcen_frontend(note_model, **pcen_kwargs):
    """Swap note_model.logmel_extractor for a trainable PCEN front-end."""
    mel_pcen = MelPCEN(note_model.logmel_extractor, **pcen_kwargs)
    note_model.logmel_extractor = mel_pcen
    return mel_pcen.pcen


_SR = 16000        # Kong model sample rate
_FPS = 100         # Kong frames per second (10 ms)
_CLASSES = 88      # A0 (MIDI 21) .. C8 (MIDI 108)
_MIDI_MIN = 21     # MIDI 21 = A0
_GUITAR_MIDI_MIN = 40   # E2
_GUITAR_MIDI_MAX = 88   # E6-ish
_ONSET_WEIGHT = 5.0
_OFFSET_WEIGHT = 2.0


# ---------------------------------------------------------------------------
# GuitarSet JAMS parsing (self-contained — training only needs start/end/pitch)
# ---------------------------------------------------------------------------

def _iter_observations(data: object) -> list[dict]:
    if isinstance(data, list):
        return [o for o in data if isinstance(o, dict)]
    if isinstance(data, dict) and "value" in data:
        times = data.get("time", [])
        durs = data.get("duration", [])
        vals = data.get("value", [])
        return [
            {"time": times[i] if i < len(times) else 0.0,
             "duration": durs[i] if i < len(durs) else 0.0,
             "value": v}
            for i, v in enumerate(vals)
        ]
    return []


def load_jams_notes(jams_path: Path) -> list[dict]:
    """Return [{start, end, pitch_midi}] from a GuitarSet JAMS file."""
    jam = json.loads(jams_path.read_text(encoding="utf-8"))
    notes: list[dict] = []
    for ann in jam.get("annotations", []):
        if ann.get("namespace") != "note_midi":
            continue
        for obs in _iter_observations(ann.get("data")):
            try:
                start = float(obs["time"])
                dur = float(obs.get("duration") or 0.0)
                pitch = int(round(float(obs["value"])))
            except (KeyError, TypeError, ValueError):
                continue
            notes.append({"start": start, "end": start + dur, "pitch_midi": pitch})
    notes.sort(key=lambda n: (n["start"], n["pitch_midi"]))
    return notes


# ---------------------------------------------------------------------------
# Label generation (identical semantics to scripts/finetune_kong.py)
# ---------------------------------------------------------------------------

def notes_to_roll(notes: list[dict], n_frames: int, fps: int = _FPS,
                  onset_sigma: float = 1.0) -> dict[str, np.ndarray]:
    frame = np.zeros((n_frames, _CLASSES), dtype=np.float32)
    onset = np.zeros((n_frames, _CLASSES), dtype=np.float32)
    offset = np.zeros((n_frames, _CLASSES), dtype=np.float32)
    velocity = np.zeros((n_frames, _CLASSES), dtype=np.float32)
    for n in notes:
        midi = int(round(n["pitch_midi"]))
        if not (_GUITAR_MIDI_MIN <= midi <= _GUITAR_MIDI_MAX):
            continue
        cls = midi - _MIDI_MIN
        on_f = max(0, min(int(round(n["start"] * fps)), n_frames - 1))
        off_f = max(on_f + 1, min(int(round(n["end"] * fps)), n_frames))
        frame[on_f:off_f, cls] = 1.0
        for d in range(-3, 4):
            t = on_f + d
            if 0 <= t < n_frames:
                onset[t, cls] = max(onset[t, cls], np.exp(-0.5 * (d / onset_sigma) ** 2))
            t = off_f + d
            if 0 <= t < n_frames:
                offset[t, cls] = max(offset[t, cls], np.exp(-0.5 * (d / onset_sigma) ** 2))
        velocity[on_f:off_f, cls] = 0.63
    return {"frame": frame, "onset": onset, "offset": offset, "velocity": velocity}


# ---------------------------------------------------------------------------
# Dataset -> fixed-length segments (batchable)
# ---------------------------------------------------------------------------

def build_segments(guitarset_dir: Path, seg_sec: float, overlap: float
                   ) -> tuple[list[np.ndarray], list[dict], int]:
    """Load train clips (players 01-05; player 00 held out) and chop to segments.

    Returns (audio_segments, roll_segments, n_train_clips).
    """
    import soundfile as sf
    import librosa

    anno_dir = guitarset_dir / "annotation"
    audio_dir = guitarset_dir / "audio"
    seg_samples = int(seg_sec * _SR)
    hop = int(seg_samples * (1 - overlap))
    n_frames = int(seg_sec * _FPS)

    audio_segs: list[np.ndarray] = []
    roll_segs: list[dict] = []
    n_clips = 0
    for jams_path in sorted(anno_dir.glob("*.jams")):
        if jams_path.stem.split("_")[0] == "00":   # player 00 = validation, skip
            continue
        if "_solo" not in jams_path.stem:           # match the validated recipe (solo only)
            continue
        wav = audio_dir / f"{jams_path.stem}_mic.wav"
        if not wav.exists():
            continue
        notes = load_jams_notes(jams_path)
        y, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != _SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=_SR)
        n_clips += 1
        start = 0
        while start + seg_samples <= len(y):
            s0 = start / _SR
            seg_notes = [
                {**n, "start": n["start"] - s0, "end": n["end"] - s0}
                for n in notes if n["end"] > s0 and n["start"] < s0 + seg_sec
            ]
            audio_segs.append(y[start:start + seg_samples].copy())
            roll_segs.append(notes_to_roll(seg_notes, n_frames))
            start += hop
    return audio_segs, roll_segs, n_clips


def build_guitar_techs_segments(gt_root: Path, seg_sec: float, overlap: float
                                ) -> tuple[list[np.ndarray], list[dict], int]:
    """Chop Guitar-TECHS (electric, CC BY 4.0) DI audio + 6-track MIDI to segments.

    Each */midi/*.mid has 6 per-string tracks; we take pitch+onset (roll only needs
    pitch). Paired with */audio/directinput/*.wav. Same segment format as GuitarSet.
    """
    import soundfile as sf
    import librosa
    import pretty_midi

    seg_samples = int(seg_sec * _SR)
    hop = int(seg_samples * (1 - overlap))
    n_frames = int(seg_sec * _FPS)
    audio_segs: list[np.ndarray] = []
    roll_segs: list[dict] = []
    n_clips = 0
    for midi_path in sorted(gt_root.glob("*/midi/*.mid")):
        di = midi_path.parent.parent / "audio" / "directinput"
        wav = next(iter(di.glob("*.wav")), None)
        if wav is None:
            continue
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = [{"start": float(n.start), "end": float(n.end), "pitch_midi": int(n.pitch)}
                 for inst in pm.instruments[:6] for n in inst.notes]
        notes.sort(key=lambda x: x["start"])
        y, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != _SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=_SR)
        n_clips += 1
        start = 0
        while start + seg_samples <= len(y):
            s0 = start / _SR
            seg_notes = [{**n, "start": n["start"] - s0, "end": n["end"] - s0}
                         for n in notes if n["end"] > s0 and n["start"] < s0 + seg_sec]
            audio_segs.append(y[start:start + seg_samples].copy())
            roll_segs.append(notes_to_roll(seg_notes, n_frames))
            start += hop
    return audio_segs, roll_segs, n_clips


# ---------------------------------------------------------------------------
# Loss (batched, positive-weighted)
# ---------------------------------------------------------------------------

def compute_loss(output: dict, target: dict, pos_weight: float) -> torch.Tensor:
    def _bce(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        n = min(pred.shape[1], tgt.shape[1])
        p = pred[:, :n].clamp(1e-7, 1 - 1e-7)
        t = tgt[:, :n]
        if pos_weight == 1.0:
            return F.binary_cross_entropy(p, t)
        w = 1.0 + (pos_weight - 1.0) * t   # upweight positive (note-present) class
        return F.binary_cross_entropy(p, t, weight=w)

    loss = _bce(output["frame_output"], target["frame"])
    loss = loss + _ONSET_WEIGHT * _bce(output["reg_onset_output"], target["onset"])
    loss = loss + _OFFSET_WEIGHT * _bce(output["reg_offset_output"], target["offset"])
    return loss


def _nested_state(model) -> dict:
    """Flat Note_pedal state_dict -> piano_transcription_inference nested format."""
    flat = model.state_dict()
    nested: dict = {"note_model": {}, "pedal_model": {}}
    for k, v in flat.items():
        for prefix in ("note_model", "pedal_model"):
            if k.startswith(prefix + "."):
                nested[prefix][k[len(prefix) + 1:]] = v
                break
    return nested


# ---------------------------------------------------------------------------
# Cheap GPU-side augmentation (gain + noise only; no per-step librosa)
# ---------------------------------------------------------------------------

def augment_batch(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """x: (B, samples). Light gain jitter (+-3 dB) + white noise (SNR 25-45 dB)."""
    B = x.shape[0]
    db = (torch.rand(B, 1, generator=rng, device=x.device) * 6.0 - 3.0)
    x = x * (10.0 ** (db / 20.0))
    sig = x.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-9)
    snr = torch.rand(B, 1, generator=rng, device=x.device) * 20.0 + 25.0
    noise_std = (sig / (10.0 ** (snr / 10.0))).sqrt()
    x = x + torch.randn(x.shape, generator=rng, device=x.device) * noise_std
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU Kong fine-tune on GuitarSet")
    ap.add_argument("--guitarset-dir", type=Path, required=True,
                    help="Dir with annotation/*.jams and audio/*_mic.wav")
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--pretrained", type=Path, default=None,
                    help="Kong checkpoint .pth (downloaded from Zenodo). If omitted, "
                         "looks in ~/piano_transcription_inference_data/")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--segment-sec", type=float, default=3.0)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--augment", action="store_true", help="Enable light gain+noise aug")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--resume-step", type=int, default=0)
    ap.add_argument("--guitar-techs-dir", type=Path, default=None,
                    help="Optional Guitar-TECHS root (electric) to MIX into training")
    ap.add_argument("--pcen", action="store_true",
                    help="Attach trainable PCEN front-end (replaces log-mel). See RESEARCH_PCEN.md")
    ap.add_argument("--pcen-warmup", type=int, default=1500,
                    help="steps to keep the CRNN frozen while PCEN+bn0 adapt (distribution shift)")
    ap.add_argument("--pcen-fe-lr", type=float, default=1e-3, help="LR for PCEN + bn0")
    ap.add_argument("--pcen-crnn-lr", type=float, default=3e-5, help="LR for the CRNN under PCEN")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    from piano_transcription_inference.models import Note_pedal
    import piano_transcription_inference.config as cfg
    model = Note_pedal(frames_per_second=cfg.frames_per_second, classes_num=cfg.classes_num)

    if args.resume:
        print(f"Resuming from {args.resume}")
        st = torch.load(str(args.resume), map_location="cpu")
        # Note_pedal.load_state_dict expects NESTED {'note_model':{}, 'pedal_model':{}}.
        m = st["model"]
        if "note_model" in m and isinstance(m["note_model"], dict):
            model.load_state_dict(m, strict=False)          # already nested
        else:                                               # flat -> nest it
            nested = {"note_model": {}, "pedal_model": {}}
            for k, v in m.items():
                for pfx in ("note_model", "pedal_model"):
                    if k.startswith(pfx + "."):
                        nested[pfx][k[len(pfx) + 1:]] = v
                        break
            model.load_state_dict(nested, strict=False)
    else:
        ckpt = args.pretrained or (Path.home() / "piano_transcription_inference_data"
                                   / "note_F1=0.9677_pedal_F1=0.9186.pth")
        print(f"Loading pretrained: {ckpt}")
        st = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(st["model"], strict=False)

    model = model.to(device).train()

    # Optional trainable PCEN front-end (swap log-mel -> learnable per-channel AGC).
    # The pretrained CRNN expects log-mel, so PCEN+bn0 get a higher LR and the CRNN
    # is frozen for --pcen-warmup steps to absorb the distribution shift first.
    frozen_crnn = None
    if args.pcen:
        pcen = attach_pcen_frontend(model.note_model)  # inlined above (self-contained)
        model = model.to(device)
        fe_params = list(pcen.parameters()) + list(model.note_model.bn0.parameters())
        fe_ids = {id(p) for p in fe_params}
        crnn_params = [p for p in model.parameters() if id(p) not in fe_ids]
        optimizer = torch.optim.Adam([
            {"params": fe_params, "lr": args.pcen_fe_lr},
            {"params": crnn_params, "lr": args.pcen_crnn_lr},
        ])
        if args.pcen_warmup > 0:
            for p in crnn_params:
                p.requires_grad_(False)
            frozen_crnn = crnn_params
        print(f"PCEN attached: fe_lr={args.pcen_fe_lr} crnn_lr={args.pcen_crnn_lr} "
              f"warmup={args.pcen_warmup}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Building training segments (players 01-05) ...")
    audio_segs, roll_segs, n_clips = build_segments(
        args.guitarset_dir, args.segment_sec, overlap=0.5)
    print(f"  GuitarSet: {n_clips} clips -> {len(audio_segs)} segments")
    n_guitarset = len(audio_segs)                       # GuitarSet segs are [0:n_guitarset]
    if args.guitar_techs_dir:
        gt_a, gt_r, gt_n = build_guitar_techs_segments(
            args.guitar_techs_dir, args.segment_sec, overlap=0.5)
        print(f"  Guitar-TECHS: {gt_n} clips -> {len(gt_a)} segments")
        audio_segs += gt_a
        roll_segs += gt_r
    print(f"  TOTAL: {len(audio_segs)} training segments")
    if not audio_segs:
        sys.exit("No segments — check --guitarset-dir layout (annotation/, audio/*_mic.wav)")

    # Stack into tensors once (kept on CPU; moved per-batch to GPU).
    audio_all = torch.from_numpy(np.stack(audio_segs))                 # (N, samples)
    frame_all = torch.from_numpy(np.stack([r["frame"] for r in roll_segs]))
    onset_all = torch.from_numpy(np.stack([r["onset"] for r in roll_segs]))
    offset_all = torch.from_numpy(np.stack([r["offset"] for r in roll_segs]))
    N = audio_all.shape[0]

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.checkpoint_dir / "training_log.jsonl"
    aug_rng = torch.Generator(device=device); aug_rng.manual_seed(args.seed)

    # Domain-balanced sampling: draw half the batch from GuitarSet (acoustic) and
    # half from Guitar-TECHS (electric) so neither domain dominates regardless of
    # segment counts — prevents the model tilting toward whichever set is larger.
    balance = bool(args.guitar_techs_dir) and 0 < n_guitarset < N
    if balance:
        print(f"  domain-balanced sampling: 50/50 acoustic[0:{n_guitarset}] vs "
              f"electric[{n_guitarset}:{N}]")

    print(f"Training {args.steps} steps  batch={args.batch}  lr={args.lr}  "
          f"pos_weight={args.pos_weight}  augment={args.augment}  balance={balance}")
    step = args.resume_step
    run_loss = 0.0
    while step < args.steps:
        if frozen_crnn is not None and step >= args.pcen_warmup:
            for p in frozen_crnn:
                p.requires_grad_(True)
            frozen_crnn = None
            print(f"[step {step}] PCEN warmup done — CRNN unfrozen", flush=True)
        if balance:
            h = args.batch // 2
            idx = torch.cat([torch.randint(0, n_guitarset, (h,)),
                             torch.randint(n_guitarset, N, (args.batch - h,))])
        else:
            idx = torch.randint(0, N, (args.batch,))
        x = audio_all[idx].to(device)
        if args.augment:
            x = augment_batch(x, aug_rng)
        target = {
            "frame": frame_all[idx].to(device),
            "onset": onset_all[idx].to(device),
            "offset": offset_all[idx].to(device),
        }
        out = model.note_model(x)
        loss = compute_loss(out, target, args.pos_weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        run_loss += loss.item()

        if step % args.log_every == 0:
            avg = run_loss / args.log_every
            run_loss = 0.0
            print(f"  step {step}/{args.steps}  loss={avg:.4f}", flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "loss": round(avg, 4)}) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            out_path = args.checkpoint_dir / f"kong_ft_step{step}.pth"
            torch.save({"step": step, "model": _nested_state(model)}, out_path)
            print(f"  saved -> {out_path}", flush=True)

    final = args.checkpoint_dir / "kong_ft_final.pth"
    torch.save({"model": _nested_state(model)}, final)
    print(f"\nFinal -> {final}\nCalibrate locally: "
          f"python scripts/calibrate_detector.py --detector kong "
          f"--model-path {final} --max-clips 30 --cache-dir data/eval_results/cache_gpu")


if __name__ == "__main__":
    main()
