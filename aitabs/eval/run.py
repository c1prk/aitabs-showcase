"""Run the Phase-2 pipeline with a tunable config."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from aitabs.pipeline.audio.onset import detect_onsets
from aitabs.pipeline.audio.pitch import NoteEvent, suppress_octave_harmonics
from aitabs.pipeline.audio.rhythm import (
    quantize_notes_adaptive,
    quantize_notes_fingerstyle_cell,
    quantize_notes_to_grid,
)
from aitabs.pipeline.audio.tempo import TempoAnalysis, clean_notes_with_tempo, estimate_tempo
from aitabs.pipeline.mapping.fingering_config import FingeringConfig
from aitabs.pipeline.mapping.guitar import GuitarSetup
from aitabs.pipeline.mapping.registry import apply_fingering
from aitabs.export.guitarpro import export_gp5

from .cache import AudioCacheEntry
from .config import PipelineConfig
from .types import EvalNote


@dataclass
class AudioStageResult:
    """Outputs after audio + tempo (before fingering)."""

    guitar_wav: str
    filtered_notes: list[NoteEvent]
    tempo_analysis: TempoAnalysis
    reference_bpm: float | None


@dataclass
class PipelineResult:
    """Outputs from one pipeline run."""

    guitar_wav: str
    filtered_notes: list[NoteEvent]
    tab_notes: list[dict]
    tempo_analysis: TempoAnalysis
    reference_bpm: float | None
    gp5_path: str | None


def _load_fingering_config(cfg: PipelineConfig, guitar_setup: GuitarSetup | None = None) -> FingeringConfig:
    if cfg.fingering_config_path:
        finger = FingeringConfig.load_json(cfg.fingering_config_path)
    else:
        finger = FingeringConfig(mapper=cfg.fingering_mapper)
    if guitar_setup is not None and cfg.use_reference_guitar_setup:
        finger = FingeringConfig(
            **{
                **finger.to_dict(),
                "tuning": list(guitar_setup.tuning),
                "capo": guitar_setup.capo,
            }
        )
    return finger


def _load_guitar_setup(reference_gp5_path: str | Path | None, cfg: PipelineConfig) -> GuitarSetup | None:
    if not cfg.use_reference_guitar_setup or not reference_gp5_path:
        return None
    path = Path(reference_gp5_path)
    if not path.is_file():
        return None
    if path.suffix.lower() == ".gp":
        from .gp_import import load_gp_notes

        _, _, setup = load_gp_notes(path)
        return setup
    from .gp5_import import load_guitar_setup

    return load_guitar_setup(path)


def _fingering_kwargs(cfg: PipelineConfig, tempo: TempoAnalysis) -> dict:
    out: dict = {}
    if cfg.chord_group_bpm:
        out["bpm"] = tempo.bpm
    return out


def _resolve_rhythm_quantizer(cfg: PipelineConfig) -> str:
    """Map config (including legacy flags) to a quantizer mode."""
    if cfg.use_fingerstyle_cell:
        return "fingerstyle_cell"
    if cfg.quantize_to_grid:
        return "grid"
    return cfg.rhythm_quantizer or "adaptive"


def _apply_rhythm_quantization(
    cfg: PipelineConfig,
    notes: list[NoteEvent],
    analysis: TempoAnalysis,
) -> tuple[list[NoteEvent], TempoAnalysis]:
    mode = _resolve_rhythm_quantizer(cfg)
    if mode == "none":
        return notes, analysis
    if mode == "fingerstyle_cell":
        return quantize_notes_fingerstyle_cell(notes, analysis), analysis
    if mode == "grid":
        return (
            quantize_notes_to_grid(
                notes,
                analysis,
                max_notes_per_tick=cfg.max_notes_per_grid_tick,
            ),
            analysis,
        )
    if mode == "adaptive":
        return quantize_notes_adaptive(
            notes,
            analysis,
            merge_frac=cfg.rhythm_merge_frac,
        )
    return notes, analysis


def run_fingering_stage(
    filtered_notes: list[NoteEvent],
    tempo_analysis: TempoAnalysis,
    config: PipelineConfig,
    *,
    guitar_setup: GuitarSetup | None = None,
) -> list[dict]:
    """Map filtered notes to tab positions using configured fingering."""
    finger_cfg = _load_fingering_config(config, guitar_setup)
    return apply_fingering(
        filtered_notes,
        finger_cfg,
        **_fingering_kwargs(config, tempo_analysis),
    )


def run_audio_stage(
    audio_path: str | Path,
    config: PipelineConfig,
    *,
    reference_bpm: float | None = None,
    stem_path: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> AudioStageResult:
    """Demucs → BasicPitch → tempo → note cleaning (no fingering)."""
    from aitabs.pipeline.audio.pitch import get_detector
    from aitabs.pipeline.audio.separate import separate_guitar

    audio_path = Path(audio_path)
    cfg = config

    if stem_path is not None and Path(stem_path).is_file():
        guitar_wav = str(Path(stem_path).resolve())
    elif cfg.use_demucs:
        out_dir = Path(cfg.demucs_output_dir or work_dir or tempfile.mkdtemp(prefix="aitabs_demucs_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        guitar_wav = separate_guitar(str(audio_path), str(out_dir))
    else:
        guitar_wav = str(audio_path.resolve())

    import librosa

    y_stem, sr_stem = librosa.load(guitar_wav, sr=22050, mono=True)
    if cfg.peak_normalize:
        y_norm = _normalize_peak(y_stem)
        if work_dir:
            norm_path = Path(work_dir) / f"{audio_path.stem}_norm.wav"
            norm_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(norm_path), y_norm, sr_stem)
            predict_path = str(norm_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                predict_path = tmp.name
                sf.write(predict_path, y_norm, sr_stem)
    else:
        predict_path = guitar_wav
        y_norm = y_stem

    raw = get_detector(
        getattr(cfg, "detector", "basic_pitch"),
        model_path=getattr(cfg, "model_path", None),
    ).detect(
        predict_path,
        onset_threshold=cfg.onset_threshold,
        frame_threshold=cfg.frame_threshold,
        confidence_threshold=cfg.confidence_threshold,
        minimum_note_length_ms=cfg.minimum_note_length_ms,
        minimum_frequency_hz=cfg.minimum_frequency_hz,
        maximum_frequency_hz=cfg.maximum_frequency_hz,
        melodia_trick=cfg.melodia_trick,
    )

    if cfg.suppress_repeat_ghosts:
        from aitabs.pipeline.audio.tempo import suppress_repeat_ghosts
        raw = suppress_repeat_ghosts(raw, cfg.repeat_ghost_min_ioi_sec)

    if cfg.snap_to_librosa_onsets:
        onsets = detect_onsets(
            predict_path,
            delta=cfg.librosa_onset_delta,
        )
        raw = _snap_notes_to_librosa(
            raw,
            onsets,
            max_snap_sec=cfg.librosa_onset_max_snap_sec,
        )

    tempo_analysis = estimate_tempo(
        y=y_norm,
        sr=sr_stem,
        hop_length=cfg.tempo_hop_length,
        ticks_per_beat=cfg.ticks_per_beat,
    )

    if cfg.use_reference_bpm and reference_bpm is not None and reference_bpm > 0:
        bpm = float(reference_bpm)
        tempo_analysis = TempoAnalysis(
            bpm=bpm,
            beat_times=tempo_analysis.beat_times,
            beat_period=60.0 / bpm,
            origin=tempo_analysis.origin,
            ticks_per_beat=cfg.ticks_per_beat,
            confidence=tempo_analysis.confidence,
        )

    if cfg.suppress_harmonics:
        raw = suppress_octave_harmonics(
            raw,
            onset_window_sec=cfg.harmonic_onset_window_sec,
            max_conf_ratio=cfg.harmonic_conf_ratio,
        )
    filtered = clean_notes_with_tempo(
        raw,
        tempo_analysis,
        pitch_merge_sec=cfg.pitch_merge_sec,
    )
    filtered, tempo_analysis = _apply_rhythm_quantization(cfg, filtered, tempo_analysis)

    # Beat-phase de-jitter: align the tracked beat grid to actual note onsets so
    # steady playing quantizes cleanly (variable tempo_map is built from these
    # beats downstream). No-op for metronomic input (onsets already on beats).
    if getattr(cfg, "refine_beats_to_onsets", False) and len(tempo_analysis.beat_times) >= 2 and filtered:
        from aitabs.pipeline.audio.tempo import refine_beats_to_onsets

        refined_beats = refine_beats_to_onsets(
            tempo_analysis.beat_times,
            [float(n["start"]) for n in filtered],
            window_frac=cfg.beat_refine_window_frac,
        )
        tempo_analysis = TempoAnalysis(
            bpm=tempo_analysis.bpm,
            beat_times=refined_beats,
            beat_period=tempo_analysis.beat_period,
            origin=float(refined_beats[0]) if len(refined_beats) else tempo_analysis.origin,
            ticks_per_beat=tempo_analysis.ticks_per_beat,
            confidence=tempo_analysis.confidence,
            phase_template=tempo_analysis.phase_template,
        )

    return AudioStageResult(
        guitar_wav=guitar_wav,
        filtered_notes=filtered,
        tempo_analysis=tempo_analysis,
        reference_bpm=reference_bpm,
    )


def _normalize_peak(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y)))
    if peak <= 0:
        return y
    return y / peak


def _snap_notes_to_librosa(
    notes: list[NoteEvent],
    onset_times: np.ndarray,
    *,
    max_snap_sec: float,
) -> list[NoteEvent]:
    if len(onset_times) == 0:
        return notes
    onsets = np.asarray(onset_times, dtype=float)
    out: list[NoteEvent] = []
    for note in notes:
        start = float(note["start"])
        idx = int(np.argmin(np.abs(onsets - start)))
        snapped = float(onsets[idx])
        if abs(snapped - start) <= max_snap_sec:
            start = snapped
        out.append(
            NoteEvent(
                start=start,
                end=float(note["end"]),
                pitch_midi=int(note["pitch_midi"]),
                confidence=float(note["confidence"]),
            )
        )
    return out


def _tab_to_eval(tab_notes: list[dict]) -> list[EvalNote]:
    return [
        EvalNote(
            start=float(n["start"]),
            end=float(n["end"]),
            pitch_midi=int(n["pitch_midi"]),
            string=int(n["string"]),
            fret=int(n["fret"]),
            confidence=float(n.get("confidence", 1.0)),
        )
        for n in tab_notes
    ]


def audio_stage_to_cache(entry: AudioStageResult, clip_id: str, audio_path: str | None = None) -> AudioCacheEntry:
    return AudioCacheEntry(
        clip_id=clip_id,
        filtered_notes=entry.filtered_notes,
        tempo_analysis=entry.tempo_analysis,
        reference_bpm=entry.reference_bpm,
        audio_path=audio_path,
    )


def _resolve_meter(
    cfg: PipelineConfig,
    reference_gp5_path: str | Path | None,
    audio: "AudioStageResult | None" = None,
):
    """Resolve (time_signature, tempo_map, lead_in_beats) for notation.

    Priority: explicit config override → reference GP5 (eval) → estimate from audio
    → default 4/4 / constant tempo. The tempo map honours mid-song tempo changes;
    ``lead_in_beats`` anchors the first downbeat (pickup) when estimated.
    """
    time_signature: tuple[int, int] = tuple(cfg.time_signature) if cfg.time_signature else (4, 4)
    tempo_map = None
    lead_in_beats = 0

    has_ref = reference_gp5_path is not None and Path(reference_gp5_path).is_file()
    if has_ref:
        from .gp5_import import load_gp5_tempo_map, load_gp5_time_signature

        if cfg.time_signature is None and cfg.use_reference_meter:
            try:
                time_signature = load_gp5_time_signature(reference_gp5_path)
            except Exception:  # noqa: BLE001 - fall back to default
                pass
        if cfg.use_reference_tempo_map:
            try:
                tempo_map = load_gp5_tempo_map(reference_gp5_path)
            except Exception:  # noqa: BLE001
                tempo_map = None
        return time_signature, tempo_map, lead_in_beats

    # No reference: estimate from audio (license-clean, numpy/librosa only).
    if audio is not None:
        from aitabs.pipeline.audio.tempo import TempoMap, estimate_meter

        beat_times = getattr(audio.tempo_analysis, "beat_times", None)
        onsets = [float(n["start"]) for n in audio.filtered_notes]
        if cfg.time_signature is None and cfg.estimate_meter and beat_times is not None and len(beat_times) >= 5:
            est = estimate_meter(onsets, beat_times)
            time_signature = est.time_signature
            lead_in_beats = est.lead_in_beats
        if cfg.estimate_tempo_map and beat_times is not None and len(beat_times) >= 2:
            tempo_map = TempoMap.from_beat_times(beat_times)

    return time_signature, tempo_map, lead_in_beats


def run_pipeline(
    audio_path: str | Path,
    config: PipelineConfig,
    *,
    reference_bpm: float | None = None,
    reference_gp5_path: str | Path | None = None,
    stem_path: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> PipelineResult:
    """Run Demucs → BasicPitch → tempo → fingering → optional GP5 export."""
    audio_path = Path(audio_path)
    cfg = config
    guitar_setup = _load_guitar_setup(reference_gp5_path, cfg)

    audio = run_audio_stage(
        audio_path,
        cfg,
        reference_bpm=reference_bpm,
        stem_path=stem_path,
        work_dir=work_dir,
    )
    tab_notes = run_fingering_stage(
        audio.filtered_notes,
        audio.tempo_analysis,
        cfg,
        guitar_setup=guitar_setup,
    )

    gp5_out: str | None = None
    if cfg.export_gp5 and work_dir:
        gp5_out = str(Path(work_dir) / f"{audio_path.stem}_pred.gp5")
        if cfg.use_reference_rhythm and reference_gp5_path and Path(reference_gp5_path).is_file():
            from aitabs.export.guitarpro_types import ExportNote
            from aitabs.export.rhythm_scaffold import export_gp5_with_reference_rhythm

            export_notes: list[ExportNote] = [
                ExportNote(
                    start=float(n["start"]),
                    end=float(n["end"]),
                    pitch_midi=int(n["pitch_midi"]),
                    string=int(n["string"]),
                    fret=int(n["fret"]),
                    confidence=float(n.get("confidence", 1.0)),
                )
                for n in tab_notes
            ]
            export_gp5_with_reference_rhythm(
                export_notes,
                reference_gp5_path,
                gp5_out,
                slot_snap_sec=cfg.reference_rhythm_snap_sec,
            )
        else:
            snap = cfg.snap_to_grid and not audio.tempo_analysis.uses_adaptive_template
            duration = cfg.duration_mode
            if audio.tempo_analysis.uses_adaptive_template and duration in ("grid_unit", "to_next_onset"):
                duration = "note_value"
            time_signature, tempo_map, lead_in_beats = _resolve_meter(cfg, reference_gp5_path, audio)
            export_gp5(
                tab_notes,
                gp5_out,
                tempo=int(round(audio.tempo_analysis.bpm)),
                time_signature=time_signature,
                tempo_grid=audio.tempo_analysis,
                snap_to_grid=snap,
                chord_grid_ticks=cfg.chord_grid_ticks,
                duration_mode=duration,  # type: ignore[arg-type]
                enable_triplets=cfg.enable_triplets,
                trim_leading_silence=cfg.trim_leading_silence,
                tuning=list(guitar_setup.tuning) if guitar_setup else None,
                capo=guitar_setup.capo if guitar_setup else 0,
                tempo_map=tempo_map,
                lead_in_beats=lead_in_beats,
                snap_dominant_grid=cfg.snap_dominant_grid,
                dominant_grid_max_mae=cfg.dominant_grid_max_mae,
                legato=cfg.legato,
                optimize_grid=cfg.optimize_grid,
                auto_rhythm=cfg.auto_rhythm,
            )

    return PipelineResult(
        guitar_wav=audio.guitar_wav,
        filtered_notes=audio.filtered_notes,
        tab_notes=tab_notes,
        tempo_analysis=audio.tempo_analysis,
        reference_bpm=audio.reference_bpm,
        gp5_path=gp5_out,
    )


def tab_notes_to_eval(tab_notes: list[dict]) -> list[EvalNote]:
    return _tab_to_eval(tab_notes)
