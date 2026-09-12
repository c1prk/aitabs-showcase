"""Pluggable fingering mappers for eval and tuning."""

from __future__ import annotations

from collections.abc import Callable

from .fingering_config import FingeringConfig
from .fingering import TabNote, map_sequence_greedy, map_sequence_viterbi

FingeringMapper = Callable[..., list[TabNote]]

FINGERING_MAPPERS: dict[str, FingeringMapper] = {
    "viterbi": map_sequence_viterbi,
    "greedy": map_sequence_greedy,
}


def get_mapper(name: str) -> FingeringMapper:
    if name not in FINGERING_MAPPERS:
        known = ", ".join(sorted(FINGERING_MAPPERS))
        raise ValueError(f"Unknown fingering mapper {name!r}. Choose from: {known}")
    return FINGERING_MAPPERS[name]


def apply_fingering(
    notes: list[dict],
    config: FingeringConfig,
    *,
    bpm: float | None = None,
    chord_group_epsilon_sec: float | None = None,
) -> list[TabNote]:
    """Run the configured fingering mapper."""
    mapper = get_mapper(config.mapper)
    return mapper(
        notes,
        config=config,
        bpm=bpm,
        ticks_per_beat=config.ticks_per_beat,
        chord_group_epsilon_sec=chord_group_epsilon_sec,
    )
