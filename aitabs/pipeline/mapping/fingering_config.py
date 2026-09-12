"""Tunable parameters for string/fret mapping."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class FingeringConfig:
    """Knobs for ``map_sequence`` / fingering mappers."""

    mapper: str = "viterbi"
    string_cost: float = 1.0
    fret_cost: float = 0.25
    open_string_bonus: float = -0.5
    max_fret_span: int = 4
    span_base_penalty: float = 5.0
    span_extra_penalty: float = 2.0
    low_fret_penalty: float = 0.10
    max_states_per_chord: int = 200
    default_epsilon_sec: float = 0.04
    use_bpm_chord_groups: bool = True
    ticks_per_beat: int = 4
    tuning: list[int] | None = None  # open-string MIDI low→high; None = standard EADGBE
    capo: int = 0  # GP5 track.offset — written frets are relative to this fret

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FingeringConfig:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> FingeringConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_FINGERING_CONFIG = FingeringConfig()
