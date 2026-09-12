"""Tests for GuitarSet beat-grid loading and beat-tracking scoring.

Pure logic only — no audio/PortAudio/model deps. The audio-estimation path
(estimate_tempo/estimate_meter) is exercised by the CLI, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

from aitabs.eval.guitarset_import import load_guitarset_beats
from aitabs.eval.rhythm_metrics import _octave_folded_pct_err, score_beats


def _beat_obs(time: float, position: int, num_beats: int = 4, beat_units: int = 4) -> dict:
    return {
        "time": time,
        "duration": 0.0,
        "value": {
            "position": position,
            "measure": 1 + (position == 1),
            "num_beats": num_beats,
            "beat_units": beat_units,
        },
        "confidence": None,
    }


def _write_jams(path: Path, beats: list[dict], tempo_bpm: float) -> None:
    path.write_text(
        json.dumps(
            {
                "annotations": [
                    {"namespace": "beat_position", "data": beats},
                    {
                        "namespace": "tempo",
                        "data": [{"time": 0.0, "duration": 0.0, "value": tempo_bpm}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_load_beats_extracts_grid_and_downbeats(tmp_path: Path) -> None:
    # 2 bars of 4/4 at 120 bpm -> beat every 0.5 s, downbeats at 0.0 and 2.0.
    beats = [
        _beat_obs(0.0, 1),
        _beat_obs(0.5, 2),
        _beat_obs(1.0, 3),
        _beat_obs(1.5, 4),
        _beat_obs(2.0, 1),
        _beat_obs(2.5, 2),
        _beat_obs(3.0, 3),
        _beat_obs(3.5, 4),
    ]
    jams = tmp_path / "clip.jams"
    _write_jams(jams, beats, tempo_bpm=120.0)

    ann = load_guitarset_beats(jams)
    assert ann.beat_times == (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    assert ann.downbeat_times == (0.0, 2.0)
    assert (ann.numerator, ann.denominator) == (4, 4)
    assert ann.tempo_bpm == 120.0


def test_load_beats_sorts_unordered_input(tmp_path: Path) -> None:
    beats = [_beat_obs(1.0, 3), _beat_obs(0.0, 1), _beat_obs(0.5, 2)]
    jams = tmp_path / "clip.jams"
    _write_jams(jams, beats, tempo_bpm=120.0)
    ann = load_guitarset_beats(jams)
    assert ann.beat_times == (0.0, 0.5, 1.0)
    assert ann.downbeat_times == (0.0,)


def test_score_beats_perfect_match() -> None:
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    dbs = [0.0, 2.0]
    s = score_beats(
        beats, beats,
        reference_downbeats=dbs, predicted_downbeats=dbs,
        reference_bpm=120.0, predicted_bpm=120.0,
        reference_numerator=4, predicted_numerator=4,
    )
    assert s.beat_f == 1.0
    assert s.downbeat_f == 1.0
    assert s.tempo_pct_err == 0.0
    assert s.meter_correct is True


def test_score_beats_phase_shifted_downbeats_fail() -> None:
    """Beats can match while downbeats are on the wrong metrical position."""
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    ref_db = [0.0, 2.0]
    pred_db = [0.5, 2.5]  # off by one beat -> no downbeat within tolerance
    s = score_beats(
        beats, beats,
        reference_downbeats=ref_db, predicted_downbeats=pred_db,
        reference_bpm=120.0, predicted_bpm=120.0,
        reference_numerator=4, predicted_numerator=4,
    )
    assert s.beat_f == 1.0
    assert s.downbeat_f == 0.0


def test_octave_folded_tempo_error() -> None:
    # Double / half tempo folds onto the reference (phase is what matters).
    assert _octave_folded_pct_err(240.0, 120.0) == 0.0
    assert _octave_folded_pct_err(60.0, 120.0) == 0.0
    assert abs(_octave_folded_pct_err(132.0, 120.0) - 0.1) < 1e-9


def test_score_beats_missing_downbeats_returns_nan() -> None:
    beats = [0.0, 0.5, 1.0]
    s = score_beats(beats, beats, reference_bpm=120.0, predicted_bpm=120.0)
    assert s.downbeat_f != s.downbeat_f  # nan when downbeats not provided
