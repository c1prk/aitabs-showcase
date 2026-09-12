"""Tests for quiet BasicPitch wrapper."""

from __future__ import annotations

import io
import sys

from aitabs.pipeline.audio.basic_pitch_infer import _FilteredStream, quiet_basic_pitch_output


def test_filtered_stream_drops_coreml_noise():
    buf = io.StringIO()
    stream = _FilteredStream(buf)
    stream.write("isfinite: True\nshape: (1, 2, 3)\ndtype: float32\n")
    stream.write("Predicting MIDI for foo.wav\n")
    stream.write("\n\n\n")
    stream.write("real output\n")
    assert buf.getvalue() == "real output\n"


def test_quiet_context_restores_stdout():
    old = sys.stdout
    with quiet_basic_pitch_output():
        print("isfinite: True")
        print("kept")
    buf = sys.stdout
    assert buf is old
