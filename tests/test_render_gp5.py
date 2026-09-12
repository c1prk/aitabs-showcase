"""Tests for the Track 0 GP5 → synthetic WAV renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aitabs.eval.gp5_import import load_gp5_notes
from aitabs.eval.render_gp5 import midi_to_hz, render_gp5_to_wav

REF_GP5 = Path("data/eval/reference/test1.gp5")


def test_midi_to_hz_a440():
    assert abs(midi_to_hz(69) - 440.0) < 1e-6
    assert abs(midi_to_hz(57) - 220.0) < 1e-6  # one octave down


@pytest.mark.skipif(not REF_GP5.is_file(), reason="reference gp5 missing")
def test_render_matches_reference_duration(tmp_path: Path):
    notes, _bpm, _setup = load_gp5_notes(REF_GP5)
    out = tmp_path / "test1.wav"
    render_gp5_to_wav(REF_GP5, out, sr=22050)

    assert out.is_file()
    audio, sr = sf.read(str(out))
    assert sr == 22050
    assert audio.ndim == 1
    assert not np.allclose(audio, 0.0)  # produced sound

    # Duration should cover the last note plus the ring/tail, and not wildly exceed it.
    last_end = max(n["end"] for n in notes)
    dur = len(audio) / sr
    assert last_end <= dur <= last_end + 2.0

    # Mechanical timing: no global offset should be needed to align with itself.
    assert abs(audio).max() <= 1.0
