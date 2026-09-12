"""Score one clip: reference GP5 vs pipeline prediction."""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .dataset import EvalPair
from .gp5_import import load_gp5_notes
from .guitarset_import import load_guitarset_notes
from .metrics import ClipScores, compare_note_lists
from .types import EvalNote
from aitabs.pipeline.mapping.fingering_config import FingeringConfig
from aitabs.pipeline.mapping.registry import apply_fingering

from .cache import AudioCacheEntry
from .run import run_fingering_stage, run_pipeline, tab_notes_to_eval, _load_guitar_setup


def load_reference_notes(pair: EvalPair) -> tuple[list[EvalNote], float]:
    """Load reference notes + BPM, dispatching on the pair's reference type."""
    if pair.reference_jams_path is not None:
        return load_guitarset_notes(pair.reference_jams_path)
    if pair.reference_gp5_path is not None:
        ref_path = Path(pair.reference_gp5_path)
        if ref_path.suffix.lower() == ".gp":
            from .gp_import import load_gp_notes

            notes, bpm, _ = load_gp_notes(ref_path)
            return notes, bpm
        notes, bpm, _ = load_gp5_notes(pair.reference_gp5_path)
        return notes, bpm
    raise ValueError(f"EvalPair {pair.clip_id!r} has no reference annotation")


def evaluate_pair(
    pair: EvalPair,
    config: PipelineConfig,
    *,
    work_dir: str | Path | None = None,
    onset_tolerance_sec: float = 0.05,
    align: str = "none",
) -> ClipScores:
    """Run pipeline on audio and compare tab output to reference (GP5 or JAMS)."""
    scores, _ref, _pred = evaluate_pair_with_notes(
        pair, config, work_dir=work_dir, onset_tolerance_sec=onset_tolerance_sec, align=align,
    )
    return scores


def evaluate_pair_with_notes(
    pair: EvalPair,
    config: PipelineConfig,
    *,
    work_dir: str | Path | None = None,
    onset_tolerance_sec: float = 0.05,
    align: str = "none",
) -> tuple[ClipScores, list[EvalNote], list[EvalNote]]:
    """Like :func:`evaluate_pair` but also return (reference, prediction) notes.

    Used for error analysis / note dumping without re-running the pipeline.
    """
    ref_notes, ref_bpm = load_reference_notes(pair)

    result = run_pipeline(
        pair.audio_path,
        config,
        reference_bpm=ref_bpm if config.use_reference_bpm else None,
        reference_gp5_path=pair.reference_gp5_path,
        stem_path=pair.stem_path,
        work_dir=work_dir,
    )

    pred_notes = tab_notes_to_eval(result.tab_notes)

    scores = compare_note_lists(
        ref_notes,
        pred_notes,
        clip_id=pair.clip_id,
        onset_tolerance_sec=onset_tolerance_sec,
        reference_bpm=ref_bpm,
        estimated_bpm=result.tempo_analysis.bpm,
        align=align,
    )
    return scores, ref_notes, pred_notes


def evaluate_pair_with_rhythm(
    pair: EvalPair,
    config: PipelineConfig,
    *,
    work_dir: str | Path | None = None,
    onset_tolerance_sec: float = 0.05,
    align: str = "none",
):
    """Like :func:`evaluate_pair` but also score the *exported* GP5 rhythm.

    Returns ``(ClipScores, RhythmScores | None)``. The rhythm score is computed
    by comparing the predicted GP5 file (notated tab the player actually reads)
    against the reference GP5 in score time, and is ``None`` when export is off.
    Requires ``config.export_gp5=True``.
    """
    from .rhythm_metrics import RhythmScores, compare_rhythm

    ref_notes, ref_bpm = load_reference_notes(pair)

    result = run_pipeline(
        pair.audio_path,
        config,
        reference_bpm=ref_bpm if config.use_reference_bpm else None,
        reference_gp5_path=pair.reference_gp5_path,
        stem_path=pair.stem_path,
        work_dir=work_dir,
    )

    pred_notes = tab_notes_to_eval(result.tab_notes)
    note_scores = compare_note_lists(
        ref_notes,
        pred_notes,
        clip_id=pair.clip_id,
        onset_tolerance_sec=onset_tolerance_sec,
        reference_bpm=ref_bpm,
        estimated_bpm=result.tempo_analysis.bpm,
        align=align,
    )

    rhythm_scores: RhythmScores | None = None
    if (
        pair.reference_gp5_path is not None
        and result.gp5_path
        and Path(result.gp5_path).is_file()
    ):
        rhythm_scores = compare_rhythm(
            pair.reference_gp5_path,
            result.gp5_path,
            clip_id=pair.clip_id,
        )
    return note_scores, rhythm_scores


def evaluate_cached_pair(
    pair: EvalPair,
    cache: AudioCacheEntry,
    pipeline_config: PipelineConfig,
    *,
    fingering_config: FingeringConfig | None = None,
    onset_tolerance_sec: float = 0.05,
) -> ClipScores:
    """Score fingering on precomputed filtered notes (no audio re-run)."""
    ref_notes, ref_bpm = load_reference_notes(pair)

    if fingering_config is not None:
        bpm = cache.tempo_analysis.bpm if pipeline_config.chord_group_bpm else None
        tab_notes = apply_fingering(cache.filtered_notes, fingering_config, bpm=bpm)
    else:
        tab_notes = run_fingering_stage(
            cache.filtered_notes,
            cache.tempo_analysis,
            pipeline_config,
            guitar_setup=_load_guitar_setup(pair.reference_gp5_path, pipeline_config),
        )

    pred_notes = tab_notes_to_eval(tab_notes)
    return compare_note_lists(
        ref_notes,
        pred_notes,
        clip_id=pair.clip_id,
        onset_tolerance_sec=onset_tolerance_sec,
        reference_bpm=ref_bpm,
        estimated_bpm=cache.tempo_analysis.bpm,
    )
