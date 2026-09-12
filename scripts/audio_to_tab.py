#!/usr/bin/env python3
"""Audio -> notation (MusicXML) using the current best pipeline:
Kong detection -> ghost suppression -> fingering -> audio-beat rhythm
(librosa beat_track + 16th grid) -> music21 MusicXML (treble-8vb, key sig).

Usage::

    python scripts/audio_to_tab.py data/eval/audio/clip.mp3 --out out.musicxml
    python scripts/audio_to_tab.py clip.mp3 --start 60 --end 135   # a segment
"""
from __future__ import annotations
import argparse, sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Audio -> MusicXML tab (current pipeline)")
    ap.add_argument("audio")
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--detector", default="kong",
                    help="'kong' (default) or 'ensemble' (Kong+BasicPitch reconciled)")
    ap.add_argument("--ensemble-mode", default="auto",
                    choices=["auto", "augment", "intersection", "union", "gated"],
                    help="reconciliation mode when --detector ensemble. 'auto' (default) "
                         "picks union with --preprocess (messy input -> inclusive BP) else "
                         "augment (clean input -> selective BP), per the config sweep")
    ap.add_argument("--ensemble-min-conf", type=float, default=0.7,
                    help="min BasicPitch singleton confidence for augment/gated modes")
    ap.add_argument("--model-path", default="eval/kong_ft_step4000.pth")
    ap.add_argument("--onset", type=float, default=0.1)
    ap.add_argument("--frame", type=float, default=0.05)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--ghost-ioi", type=float, default=0.2)
    ap.add_argument("--grid", type=int, default=4, help="subdivisions per beat (4=16th, 8=32nd)")
    ap.add_argument("--preprocess", action="store_true",
                    help="Tier-1 front-end (HPF + loudness normalize) before detection; "
                         "recovers recall on reverberant/quiet/noisy real-world audio")
    ap.add_argument("--pcen", action="store_true",
                    help="--model-path checkpoint has a trainable PCEN front-end")
    args = ap.parse_args()

    import librosa, soundfile as sf, music21 as m21
    from collections import Counter
    from aitabs.pipeline.audio.pitch import get_detector
    from aitabs.pipeline.audio.tempo import suppress_repeat_ghosts
    from aitabs.pipeline.mapping.registry import apply_fingering
    from aitabs.pipeline.mapping.fingering_config import FingeringConfig

    # load (segment) and write a temp wav for the detector
    y, sr = librosa.load(args.audio, sr=16000, offset=args.start,
                         duration=(args.end - args.start) if args.end else None)
    if args.preprocess:
        from aitabs.pipeline.audio.preprocess import preprocess_audio
        y = preprocess_audio(y, sr)
    tmp = Path(tempfile.gettempdir()) / "a2t_seg.wav"
    sf.write(str(tmp), y, sr)

    if args.detector == "ensemble":
        from aitabs.pipeline.audio.ensemble import EnsembleDetector
        # Adaptive rule (config sweep): preprocessed/messy input -> union (inclusive
        # BP recovers degraded recall, 0.918); clean input -> augment (selective BP,
        # best clean 0.921). Explicit --ensemble-mode overrides.
        mode = ("union" if args.preprocess else "augment") if args.ensemble_mode == "auto" \
            else args.ensemble_mode
        print(f"ensemble mode: {mode}" + ("  (auto: preprocess->union)" if args.ensemble_mode == "auto" and args.preprocess else ""))
        detector = EnsembleDetector(args.model_path, mode=mode,
                                    min_singleton_conf=args.ensemble_min_conf,
                                    kong_pcen=args.pcen)
    else:
        detector = get_detector(args.detector, model_path=args.model_path, pcen=args.pcen)
    det = detector.detect(
        str(tmp), onset_threshold=args.onset, frame_threshold=args.frame,
        confidence_threshold=args.conf, minimum_note_length_ms=58.0,
        minimum_frequency_hz=82.0, maximum_frequency_hz=1400.0, melodia_trick=False)
    det = sorted(suppress_repeat_ghosts([dict(n) for n in det], args.ghost_ioi),
                 key=lambda n: n["start"])
    if not det:
        sys.exit("No notes detected in this segment.")

    # audio beats (quarter grid), extended to cover the segment
    dur = librosa.get_duration(y=y, sr=sr)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    ibi = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    be = list(beats)
    while be and be[0] - ibi > -0.3 * ibi:
        be.insert(0, be[0] - ibi)
    while be and be[-1] + ibi < dur:
        be.append(be[-1] + ibi)
    be = np.array(be); idx = np.arange(len(be), dtype=float)
    bpm = int(round(60 / ibi))
    on = np.array([n["start"] for n in det])
    q = np.round(np.interp(on, be, idx) * args.grid) / args.grid
    q0 = q.min(); qpos = q - q0

    events: dict[float, list[int]] = {}
    for i, p in enumerate(qpos):
        events.setdefault(round(float(p), 3), []).append(int(det[i]["pitch_midi"]))
    offs = sorted(events)

    part = m21.stream.Part()
    part.insert(0, m21.instrument.AcousticGuitar())
    part.insert(0, m21.clef.Treble8vbClef())
    part.insert(0, m21.tempo.MetronomeMark(number=bpm))
    part.insert(0, m21.meter.TimeSignature("4/4"))
    vh = Counter()
    for k, off in enumerate(offs):
        d = (offs[k + 1] - off) if k + 1 < len(offs) else 1.0
        d = float(np.clip(round(d * args.grid) / args.grid, 1.0 / args.grid, 4.0))
        vh[d] += 1
        pits = sorted(set(events[off]))
        el = m21.note.Note(pits[0]) if len(pits) == 1 else m21.chord.Chord(pits)
        el.quarterLength = d
        part.insert(float(off), el)
    try:
        key = part.analyze("key")
        part.insert(0, m21.key.KeySignature(key.sharps))
        keyname = f"{key.tonic.name} {key.mode}"
    except Exception:
        keyname = "?"
    part.makeNotation(inPlace=True)
    for m in part.getElementsByClass("Measure"):
        m.clef = m21.clef.Treble8vbClef()

    out = args.out or (os.path.splitext(args.audio)[0] + ".musicxml")
    part.write("musicxml", out)
    print(f"notes={len(det)} attacks={len(offs)} tempo~{bpm}bpm key={keyname}")
    print("note values (1=qtr .5=8th .25=16th):", dict(sorted(vh.items())))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
