"""Tests for capo and alternate tuning from GP5 metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from aitabs.eval.gp5_import import load_guitar_setup, load_gp5_notes
from aitabs.pipeline.mapping.guitar import (
    GuitarSetup,
    STANDARD_TUNING,
    midi_to_positions,
)
from aitabs.pipeline.mapping.fingering_config import FingeringConfig
from aitabs.pipeline.mapping.registry import apply_fingering

REF = Path(__file__).resolve().parents[1] / "data" / "eval" / "reference"


@pytest.mark.parametrize(
    "name,capo,tuning",
    [
        ("test1.gp5", 0, STANDARD_TUNING),
        ("test7 (Let Her Go).gp5", 7, STANDARD_TUNING),
        ("test8 (River Flows in You).gp5", 2, STANDARD_TUNING),
        ("test3 (Claude de Lune).gp5", 0, [39, 44, 49, 54, 58, 63]),
    ],
)
def test_load_guitar_setup_from_reference(name: str, capo: int, tuning: list[int]):
    path = REF / name
    if not path.is_file():
        pytest.skip(f"missing {name}")
    setup = load_guitar_setup(path)
    assert setup.capo == capo
    assert list(setup.tuning) == tuning


def test_capo_raises_concert_pitch_on_import():
    path = REF / "test7 (Let Her Go).gp5"
    if not path.is_file():
        pytest.skip("test7 reference missing")
    notes, _, setup = load_gp5_notes(path)
    assert setup.capo == 7
    open_high_e = setup.written_to_concert_midi(5, 0)
    assert open_high_e == 64 + 7
    assert any(n["fret"] == 0 and n["pitch_midi"] == open_high_e for n in notes)


def test_capo_fingering_uses_written_frets():
    setup = GuitarSetup(capo=7)
    # Open high e with capo 7 sounds at MIDI 71; written tab shows fret 0.
    positions = midi_to_positions(71, setup=setup)
    assert (5, 0) in positions


def test_down_tuning_fingering():
    setup = GuitarSetup(tuning=tuple([39, 44, 49, 54, 58, 63]))
    positions = midi_to_positions(63, setup=setup)  # open high string
    assert (5, 0) in positions


def test_fingering_config_capo_applied():
    notes = [{"start": 0.0, "end": 0.5, "pitch_midi": 71, "confidence": 1.0}]
    cfg = FingeringConfig(tuning=list(STANDARD_TUNING), capo=7)
    tab = apply_fingering(notes, cfg)
    assert len(tab) == 1
    assert tab[0]["string"] == 5
    assert tab[0]["fret"] == 0
