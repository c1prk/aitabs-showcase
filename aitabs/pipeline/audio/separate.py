"""Demucs source separation — isolate guitar from mixed audio."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def separate_guitar(input_path: str, output_dir: str) -> str:
    """Run Demucs htdemucs to split audio into guitar ('other') vs everything else.

    Uses the --two-stems flag so only two files are produced:
      <output_dir>/htdemucs/<track_name>/other.wav   ← guitar lives here
      <output_dir>/htdemucs/<track_name>/no_other.wav

    Args:
        input_path: Path to the source audio or video file.
        output_dir: Destination folder for separated stems.

    Returns:
        Absolute path to the isolated guitar stem WAV.

    Raises:
        RuntimeError: If Demucs exits with a non-zero code.

    TODO (you):
        - Experiment with htdemucs vs htdemucs_6s (6-stem model) to see which
          puts guitar cleanly in the 'guitar' stem vs the 'other' stem.
        - For mixed recordings with bass guitar, the 'bass' stem may bleed in.
          Consider a second-pass separation or post-filter.
        - If GPU is available (Modal), pass `--device cuda` for 10× speed.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "-m", "demucs",
        "--two-stems", "other",
        "-o", output_dir,
        input_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{result.stderr}")

    track_name = Path(input_path).stem
    stem_path = Path(output_dir) / "htdemucs" / track_name / "other.wav"

    if not stem_path.exists():
        raise FileNotFoundError(f"Expected stem at {stem_path} but not found.")

    return str(stem_path)
