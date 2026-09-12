#!/usr/bin/env python3
"""Pre-flight check for new tabbed recordings before running eval.

For each reference ``.gp5`` (+ matching audio) it prints a one-line profile and
flags problems, so a session of fresh recordings is validated *before* you spend
time on a full eval run:

- confirms each ``.gp5`` has a matching audio file,
- reads tempo(s), time signature, tuning, capo, note count, duration,
- characterises difficulty: mean polyphony, % let-ring overlap, fastest IOI,
- labels each clip mono / melody+bass / chordal so you can see at a glance
  whether the set is **diverse** (each clip should stress a different case).

Standalone: needs only PyGuitarPro (no ML stack).

    python scripts/check_recordings.py data/eval
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import guitarpro
from guitarpro.models import BeatStatus, Duration

_TPQ = Duration.quarterTime
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aif", ".aiff")


def _find_refs(root: Path) -> list[Path]:
    refs: list[Path] = []
    for d in ([root / "reference"] if (root / "reference").is_dir() else [root]):
        for ext in (".gp5", ".gp4", ".gpx"):
            refs.extend(d.glob(f"*{ext}"))
    return sorted(refs)


def _find_audio(ref: Path, root: Path) -> Path | None:
    for d in {ref.parent, root, root / "audio"}:
        for ext in _AUDIO_EXTS:
            cand = d / f"{ref.stem}{ext}"
            if cand.is_file():
                return cand
    return None


def _profile(ref: Path) -> dict:
    song = guitarpro.parse(str(ref))
    track = song.tracks[0]
    header_bpm = float(song.tempo) if song.tempo else 120.0
    ts = song.measureHeaders[0].timeSignature
    tuning = [s.value for s in track.strings]
    capo = int(getattr(track, "offset", 0) or 0)

    notes: list[tuple[float, float, int]] = []  # (start_sec, end_sec, pitch)
    elapsed_sec = 0.0
    bpm = header_bpm
    tempos = {round(header_bpm)}
    for measure in track.measures:
        if not measure.voices:
            continue
        for beat in measure.voices[0].beats:
            mtc = getattr(getattr(beat, "effect", None), "mixTableChange", None)
            if mtc is not None and mtc.tempo is not None:
                bpm = float(mtc.tempo.value)
                tempos.add(round(bpm))
            dur_sec = beat.duration.time * 60.0 / (bpm * _TPQ)
            if beat.status != BeatStatus.rest and beat.notes:
                for n in beat.notes:
                    notes.append((elapsed_sec, elapsed_sec + dur_sec, int(n.value)))
            elapsed_sec += dur_sec

    n = len(notes)
    duration = elapsed_sec
    tempo_list = sorted(tempos)
    if len(tempo_list) <= 2:
        tempo_str = "/".join(str(t) for t in tempo_list)
    else:
        tempo_str = f"{tempo_list[0]}-{tempo_list[-1]}({len(tempo_list)})"
    onsets = sorted({round(s, 4) for s, _, _ in notes})
    iois = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    # polyphony per onset
    by_onset: dict[float, int] = {}
    for s, _, _ in notes:
        by_onset[round(s, 3)] = by_onset.get(round(s, 3), 0) + 1
    poly_mean = st.mean(by_onset.values()) if by_onset else 0.0
    # let-ring overlap: a note still sounding when a later, different note starts
    overlap = 0
    ns = sorted(notes)
    for i, (s, e, p) in enumerate(ns):
        for j in range(i + 1, min(i + 12, n)):
            if ns[j][0] >= e:
                break
            if ns[j][2] != p:
                overlap += 1
                break
    overlap_pct = 100.0 * overlap / n if n else 0.0
    fastest_ioi_ms = min(iois) * 1000 if iois else 0.0

    if poly_mean < 1.2:
        kind = "monophonic"
    elif poly_mean < 2.2:
        kind = "melody+bass"
    else:
        kind = "chordal"

    return {
        "title": song.title or ref.stem,
        "ts": f"{ts.numerator}/{int(round(3840 / ts.denominator.time)) if ts.denominator.time else 4}",
        "tempos": tempo_str,
        "tuning": tuning,
        "capo": capo,
        "notes": n,
        "dur": duration,
        "poly": poly_mean,
        "overlap_pct": overlap_pct,
        "fast_ioi": fastest_ioi_ms,
        "kind": kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate tabbed recordings before eval")
    ap.add_argument("data_dir", type=Path, nargs="?", default=Path("data/eval"))
    args = ap.parse_args()

    root = args.data_dir.resolve()
    refs = _find_refs(root)
    if not refs:
        print(f"No .gp5 references under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"{'clip':24}{'audio':>6}{'ts':>5}{'tempo':>8}{'capo':>5}{'notes':>6}"
          f"{'dur':>6}{'poly':>6}{'olap%':>6}{'fIOI':>6}  kind")
    kinds: list[str] = []
    warnings: list[str] = []
    std_tuning = [64, 59, 55, 50, 45, 40]
    for ref in refs:
        audio = _find_audio(ref, root)
        try:
            p = _profile(ref)
        except Exception as exc:  # noqa: BLE001
            print(f"{ref.stem[:24]:24}  PARSE ERROR: {exc}", file=sys.stderr)
            continue
        kinds.append(p["kind"])
        a = "ok" if audio else "MISS"
        print(f"{ref.stem[:24]:24}{a:>6}{p['ts']:>5}{p['tempos']:>8}{p['capo']:>5}"
              f"{p['notes']:>6}{p['dur']:>6.0f}{p['poly']:>6.2f}{p['overlap_pct']:>6.0f}"
              f"{p['fast_ioi']:>6.0f}  {p['kind']}")
        if not audio:
            warnings.append(f"{ref.stem}: no matching audio file ({', '.join(_AUDIO_EXTS)})")
        if p["dur"] < 8:
            warnings.append(f"{ref.stem}: very short ({p['dur']:.0f}s)")
        if p["notes"] == 0:
            warnings.append(f"{ref.stem}: zero notes parsed (wrong track/voice?)")
        if p["tuning"] != std_tuning and p["capo"] == 0:
            warnings.append(f"{ref.stem}: non-standard tuning {p['tuning']} — confirm it's intended")

    print("\nDiversity:", {k: kinds.count(k) for k in sorted(set(kinds))})
    if len(set(kinds)) < min(3, len(kinds)):
        print("  ⚠ clips are similar — vary mono / fingerpicking / chordal / distorted / capo for a richer test")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("\nAll clips look well-formed. Run:")
        print(f"  python scripts/eval_dataset.py {args.data_dir} --no-demucs --rhythm")
    print("\nColumns: poly=mean notes/onset, olap%=let-ring overlap (recall difficulty), "
          "fIOI=fastest inter-onset (ms)")


if __name__ == "__main__":
    main()
