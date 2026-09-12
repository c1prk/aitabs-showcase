"""Tests for the GuitarSet JAMS importer."""

from __future__ import annotations

import json
from pathlib import Path

from aitabs.eval.guitarset_import import load_guitarset_notes, load_guitarset_training_pairs
from aitabs.pipeline.mapping.guitar import fret_to_midi


def _note_midi_anno(string_idx: int, observations: list[dict]) -> dict:
    return {
        "namespace": "note_midi",
        "annotation_metadata": {"data_source": str(string_idx)},
        "sandbox": {},
        "data": observations,
    }


def _write_jams(path: Path, annotations: list[dict]) -> None:
    path.write_text(json.dumps({"annotations": annotations}), encoding="utf-8")


def test_load_guitarset_notes_basic(tmp_path: Path) -> None:
    jams = tmp_path / "00_BN1-129-Eb_solo.jams"
    _write_jams(
        jams,
        [
            {
                "namespace": "tempo",
                "data": [{"time": 0.0, "duration": 0.0, "value": 96.0, "confidence": 1.0}],
            },
            # low E (string 0, open=40): fret 0
            _note_midi_anno(0, [{"time": 0.5, "duration": 0.25, "value": 40.0, "confidence": None}]),
            # high e (string 5, open=64): value 67 → fret 3
            _note_midi_anno(5, [{"time": 1.0, "duration": 0.5, "value": 67.0, "confidence": None}]),
        ],
    )

    notes, bpm = load_guitarset_notes(jams)

    assert bpm == 96.0
    assert len(notes) == 2
    # sorted by start
    assert notes[0]["start"] == 0.5
    assert notes[0]["pitch_midi"] == 40
    assert notes[0]["string"] == 0
    assert notes[0]["fret"] == 0
    assert notes[0]["end"] == 0.75

    assert notes[1]["pitch_midi"] == 67
    assert notes[1]["string"] == 5
    assert notes[1]["fret"] == 3
    assert notes[1]["end"] == 1.5

    # string/fret must reconstruct the pitch
    for n in notes:
        assert fret_to_midi(n["string"], n["fret"]) == n["pitch_midi"]


def test_missing_tempo_defaults_to_120(tmp_path: Path) -> None:
    jams = tmp_path / "x_solo.jams"
    _write_jams(
        jams,
        [_note_midi_anno(2, [{"time": 0.0, "duration": 0.1, "value": 55.0, "confidence": None}])],
    )
    notes, bpm = load_guitarset_notes(jams)
    assert bpm == 120.0
    assert len(notes) == 1
    # string 2 (D, open=50), value 55 → fret 5
    assert notes[0]["string"] == 2
    assert notes[0]["fret"] == 5


def test_fractional_pitch_is_rounded(tmp_path: Path) -> None:
    jams = tmp_path / "y_solo.jams"
    _write_jams(
        jams,
        [_note_midi_anno(0, [{"time": 0.0, "duration": 0.2, "value": 52.4, "confidence": None}])],
    )
    notes, _ = load_guitarset_notes(jams)
    assert notes[0]["pitch_midi"] == 52
    # string 0 (open=40) → fret 12
    assert notes[0]["fret"] == 12


# ---------------------------------------------------------------------------
# load_guitarset_training_pairs
# ---------------------------------------------------------------------------

def _make_training_dir(root: Path, clips: list[tuple[str, list[dict]]]) -> Path:
    """Create a minimal GuitarSet directory layout for training pair tests."""
    anno_dir = root / "annotation"
    audio_dir = root / "audio"
    anno_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    for clip_id, annotations in clips:
        (anno_dir / f"{clip_id}.jams").write_text(
            json.dumps({"annotations": annotations}), encoding="utf-8"
        )
        # Create a dummy audio file so is_file() returns True
        (audio_dir / f"{clip_id}_mic.wav").write_bytes(b"RIFF")
    return root


def test_training_pairs_solo_only(tmp_path: Path) -> None:
    clips = [
        ("00_BN1-129-Eb_solo", [_note_midi_anno(0, [{"time": 0.0, "duration": 0.3, "value": 40.0, "confidence": None}])]),
        ("00_BN1-129-Eb_comp", [_note_midi_anno(0, [{"time": 0.0, "duration": 0.3, "value": 40.0, "confidence": None}])]),
    ]
    _make_training_dir(tmp_path, clips)
    pairs = load_guitarset_training_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0]["clip_id"] == "00_BN1-129-Eb_solo"


def test_training_pairs_no_string_or_fret(tmp_path: Path) -> None:
    clips = [
        ("01_solo", [_note_midi_anno(2, [{"time": 0.1, "duration": 0.2, "value": 55.0, "confidence": None}])]),
    ]
    _make_training_dir(tmp_path, clips)
    pairs = load_guitarset_training_pairs(tmp_path)
    note = pairs[0]["notes"][0]
    assert "string" not in note
    assert "fret" not in note
    assert note["pitch_midi"] == 55
    assert note["confidence"] == 1.0


def test_training_pairs_skips_missing_audio(tmp_path: Path) -> None:
    anno_dir = tmp_path / "annotation"
    anno_dir.mkdir(parents=True)
    (tmp_path / "audio").mkdir()
    # JAMS exists but no matching _mic.wav
    (anno_dir / "02_solo.jams").write_text(
        json.dumps({"annotations": [_note_midi_anno(0, [{"time": 0.0, "duration": 0.1, "value": 40.0, "confidence": None}])]}),
        encoding="utf-8",
    )
    pairs = load_guitarset_training_pairs(tmp_path)
    assert pairs == []


def test_training_pairs_note_sort_order(tmp_path: Path) -> None:
    clips = [
        ("03_solo", [
            _note_midi_anno(0, [
                {"time": 0.5, "duration": 0.1, "value": 45.0, "confidence": None},
                {"time": 0.1, "duration": 0.2, "value": 40.0, "confidence": None},
            ]),
        ]),
    ]
    _make_training_dir(tmp_path, clips)
    pairs = load_guitarset_training_pairs(tmp_path)
    starts = [n["start"] for n in pairs[0]["notes"]]
    assert starts == sorted(starts)


def test_out_of_range_string_falls_back_to_valid_position(tmp_path: Path) -> None:
    # value 64 on string 0 (open 40) would be fret 24 (> MAX_FRETS=22) → fall back
    jams = tmp_path / "z_solo.jams"
    _write_jams(
        jams,
        [_note_midi_anno(0, [{"time": 0.0, "duration": 0.2, "value": 64.0, "confidence": None}])],
    )
    notes, _ = load_guitarset_notes(jams)
    assert notes[0]["pitch_midi"] == 64
    # pitch is preserved via a playable position
    assert fret_to_midi(notes[0]["string"], notes[0]["fret"]) == 64
    assert 0 <= notes[0]["fret"] <= 22
