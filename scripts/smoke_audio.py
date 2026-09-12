#!/usr/bin/env python3
"""Run Demucs + BasicPitch on one file — quick Phase 2 sanity check."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from aitabs.pipeline.audio.onset import detect_onsets
from aitabs.pipeline.audio.pitch import detect_notes
from aitabs.pipeline.audio.separate import separate_guitar


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/smoke_audio.py <audio-or-video-path>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        stems = Path(tmp) / "stems"
        print("Separating guitar (Demucs)...")
        guitar_wav = separate_guitar(str(input_path), str(stems))
        print(f"  → {guitar_wav}")

        print("Detecting notes (BasicPitch)...")
        notes = detect_notes(guitar_wav)
        print(f"  → {len(notes)} notes")

        print("Detecting onsets (librosa)...")
        onsets = detect_onsets(guitar_wav)
        print(f"  → {len(onsets)} onsets")

    for note in notes[:10]:
        print(
            f"  {note['start']:.2f}s–{note['end']:.2f}s  "
            f"MIDI {note['pitch_midi']}  conf={note['confidence']:.2f}"
        )
    if len(notes) > 10:
        print(f"  ... and {len(notes) - 10} more")


if __name__ == "__main__":
    main()
