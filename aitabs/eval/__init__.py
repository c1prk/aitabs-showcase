"""Evaluation: GP5 ground truth vs pipeline output."""

from .config import PipelineConfig
from .dataset import discover_pairs, discover_guitarset_pairs, EvalPair
from .error_analysis import (
    FalsePositiveBreakdown,
    MissBreakdown,
    categorize_false_negatives,
    categorize_false_positives,
)
from .evaluate import (
    evaluate_pair,
    evaluate_pair_with_notes,
    evaluate_pair_with_rhythm,
    load_reference_notes,
)
from .gp5_import import load_gp5_notes, load_guitar_setup
from .guitarset_import import load_guitarset_notes
from .render_gp5 import render_gp5_to_wav
from .metrics import ClipScores, compare_note_lists, estimate_time_offset, rhythm_tick_f1
from .rhythm_metrics import RhythmScores, compare_rhythm, read_notes_and_rests
from .run import PipelineResult, run_pipeline, tab_notes_to_eval

__all__ = [
    "PipelineConfig",
    "EvalPair",
    "discover_pairs",
    "discover_guitarset_pairs",
    "load_gp5_notes",
    "load_guitar_setup",
    "load_guitarset_notes",
    "load_reference_notes",
    "render_gp5_to_wav",
    "ClipScores",
    "RhythmScores",
    "compare_note_lists",
    "compare_rhythm",
    "read_notes_and_rests",
    "estimate_time_offset",
    "rhythm_tick_f1",
    "PipelineResult",
    "run_pipeline",
    "tab_notes_to_eval",
    "evaluate_pair",
    "evaluate_pair_with_notes",
    "evaluate_pair_with_rhythm",
    "categorize_false_positives",
    "categorize_false_negatives",
    "FalsePositiveBreakdown",
    "MissBreakdown",
]
