"""Shared types for tab export."""

from __future__ import annotations

from typing import TypedDict


class ExportNote(TypedDict):
    start: float
    end: float
    pitch_midi: int
    string: int
    fret: int
    confidence: float


def aitabs_string_to_gp(string_index: int) -> int:
    """Map AITabs string index (0=low E) to Guitar Pro string number (1=high e)."""
    return 6 - string_index
