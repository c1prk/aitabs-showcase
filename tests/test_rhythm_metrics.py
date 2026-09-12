"""Tests for rhythm/notation accuracy metrics (aitabs.eval.rhythm_metrics)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aitabs.eval.gp5_import import load_gp5_notes
from aitabs.eval.rhythm_metrics import (
    compare_rhythm,
    read_notes_and_rests,
)
from aitabs.export.guitarpro import export_gp5

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "eval" / "reference"
TEST1 = REF_DIR / "test1.gp5"
TEST2 = REF_DIR / "test2.gp5"

pytestmark = pytest.mark.skipif(
    not TEST1.exists(), reason="reference gp5 fixtures not available"
)

needs_test2 = pytest.mark.skipif(
    not TEST2.is_file(), reason="test2.gp5 reference not available"
)


def test_read_notes_normalizes_first_attack_to_zero():
    notes, _ = read_notes_and_rests(TEST1)
    assert notes
    assert notes[0].tick == 0


def test_read_notes_durations_positive():
    notes, _ = read_notes_and_rests(TEST1)
    assert all(n.duration > 0 for n in notes)


def test_identical_file_scores_perfectly():
    s = compare_rhythm(TEST1, TEST1, clip_id="self")
    assert s.metric_position_f1 == pytest.approx(1.0)
    assert s.note_value_accuracy == pytest.approx(1.0)
    assert s.rest_f1 == pytest.approx(1.0)


def test_note_value_mode_recovers_rests_on_test1():
    """The headline fix: to_next_onset emits no rests; note_value recovers them."""
    notes, bpm, _ = load_gp5_notes(TEST1)
    tempo = int(round(bpm))

    out_legacy = Path("/tmp/rm_test1_legacy.gp5")
    out_nv = Path("/tmp/rm_test1_nv.gp5")
    export_gp5(notes, out_legacy, tempo=tempo, duration_mode="to_next_onset")
    export_gp5(notes, out_nv, tempo=tempo, duration_mode="note_value")

    s_legacy = compare_rhythm(TEST1, out_legacy)
    s_nv = compare_rhythm(TEST1, out_nv)

    # Legacy stretches every note to the next onset -> no rests at all.
    assert s_legacy.rest_f1 == pytest.approx(0.0)
    # note_value recovers them.
    assert s_nv.rest_f1 > 0.8
    # ...without wrecking onset positions.
    assert s_nv.metric_position_f1 > 0.95


@needs_test2
def test_note_value_no_regression_on_continuous_clip():
    """test2 is continuous fingerstyle (no rests): note_value must not regress."""
    notes, bpm, _ = load_gp5_notes(TEST2)
    tempo = int(round(bpm))
    out_nv = Path("/tmp/rm_test2_nv.gp5")
    export_gp5(notes, out_nv, tempo=tempo, duration_mode="note_value")
    s = compare_rhythm(TEST2, out_nv)
    assert s.metric_position_f1 > 0.95
    assert s.note_value_accuracy > 0.9


def test_summary_line_formats():
    s = compare_rhythm(TEST1, TEST1, clip_id="test1")
    line = s.summary_line()
    assert "test1" in line
    assert "pos F1=" in line
    assert "rest_F1=" in line
