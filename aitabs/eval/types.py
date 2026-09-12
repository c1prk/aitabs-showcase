"""Shared note types for evaluation."""

from __future__ import annotations

from typing import TypedDict


class EvalNote(TypedDict):
    """One note event for metrics (reference or prediction)."""

    start: float
    end: float
    pitch_midi: int
    string: int
    fret: int
    confidence: float
