"""Tunable pipeline knobs for evaluation and search."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """All Phase-2 knobs that affect audio → tab quality."""

    # Separation
    use_demucs: bool = True
    demucs_output_dir: str | None = None  # None → temp dir per run

    # BasicPitch
    confidence_threshold: float = 0.42
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    pitch_merge_sec: float = 0.15
    peak_normalize: bool = True
    minimum_note_length_ms: float = 58.0
    melodia_trick: bool = True
    minimum_frequency_hz: float | None = 82.0
    maximum_frequency_hz: float | None = 1400.0

    # Harmonic / octave ghost suppression (post BasicPitch). Opt-in: targets
    # over-detection on real audio (octave/overtone false positives).
    suppress_harmonics: bool = False
    harmonic_onset_window_sec: float = 0.05
    harmonic_conf_ratio: float = 1.0

    # Repeat-ghost suppression (drop same-pitch double-triggers within min IOI).
    # Simpler + more reliable than the tempo-grid dedup for detector
    # over-segmentation on sustained/repeated notes (e.g. an open-string pedal).
    suppress_repeat_ghosts: bool = False
    repeat_ghost_min_ioi_sec: float = 0.12

    # Onset refinement (librosa snap)
    snap_to_librosa_onsets: bool = False
    librosa_onset_delta: float = 0.07
    librosa_onset_max_snap_sec: float = 0.05

    # Tempo
    use_reference_bpm: bool = True  # when ref GP5 available
    tempo_hop_length: int = 512
    ticks_per_beat: int = 4
    rhythm_quantizer: str = "adaptive"  # adaptive | grid | fingerstyle_cell | none
    rhythm_merge_frac: float = 0.04  # phase-cluster merge width (beat fraction)
    quantize_to_grid: bool = False  # legacy: forces rhythm_quantizer=grid when True
    max_notes_per_grid_tick: int | None = None  # 1 = one attack per grid step
    use_fingerstyle_cell: bool = False  # legacy: forces fingerstyle_cell quantizer
    use_reference_rhythm: bool = False  # optional eval-only GP5 layout from reference
    reference_rhythm_snap_sec: float = 0.12
    use_reference_guitar_setup: bool = True  # tuning + capo from reference GP5

    # Fingering
    chord_group_bpm: bool = True  # map_sequence bpm grouping
    fingering_mapper: str = "viterbi"
    fingering_config_path: str | None = None  # JSON overrides FingeringConfig

    # Detector backend: "basic_pitch" (default), "recall_first", or "finetuned"
    detector: str = "basic_pitch"
    # Path to a fine-tuned SavedModel (only used when detector="finetuned")
    model_path: str | None = None

    # GP5 export (for comparing exported file vs reference)
    export_gp5: bool = False
    snap_to_grid: bool = False
    chord_grid_ticks: int = 0
    duration_mode: str = "note_value"  # note_value | to_next_onset | grid_unit | detected
    trim_leading_silence: bool = True
    enable_triplets: bool = False  # only honoured by duration_mode="note_value"
    # Meter / tempo for notation. None → read from reference GP5 when available,
    # else default 4/4 (time_signature) / constant tempo (use_reference_tempo_map).
    time_signature: tuple[int, int] | None = None
    use_reference_meter: bool = True       # read time signature from reference GP5
    use_reference_tempo_map: bool = True   # honour reference GP5 mid-song tempo changes
    estimate_meter: bool = True            # estimate meter/downbeat from audio when no reference
    estimate_tempo_map: bool = True        # build a variable tempo map from tracked beats
    refine_beats_to_onsets: bool = True    # snap tracked beats onto note onsets (phase de-jitter)
    beat_refine_window_frac: float = 0.35  # max beat→onset snap distance, as fraction of beat period
    # Live-recording notation (note_value mode). Off by default so the GP5-reference
    # eval keeps its exact rhythm/rest scoring; turn on for human audio recordings.
    # WARNING: snap_dominant_grid locks the WHOLE phrase to ONE subdivision, so it
    # flattens mixed rhythms — below ~48% sixteenths it collapses all sixteenths to
    # eighths (verified). Use it ONLY for steady single-resolution takes. For varied
    # real music prefer legato=True alone: the per-beat quantizer keeps mixed
    # eighth/sixteenth resolution and legato gives readable connected durations.
    snap_dominant_grid: bool = False       # lock onsets to ONE grid — flattens mixed rhythms; steady takes only
    dominant_grid_max_mae: float = 0.12    # beats; mean snap error to accept a coarser grid
    legato: bool = False                   # ignore audio note-offsets; note length = inter-onset interval
    # optimize_grid fixes a few-% tempo-SPACING error (false sixteenths from a wrong
    # BPM estimate). Do NOT enable it for mixed-rhythm material: its eighth-preferring
    # complexity penalty drops genuine minority sixteenths. Use only on a confirmed
    # tempo-estimate error, not as a general rhythm cleaner.
    optimize_grid: bool = False            # joint tempo-spacing + phase alignment (confirmed tempo error only)
    # Self-configuring rhythm: infer legato vs rests, tempo-spacing correction, and
    # mixed-resolution grid from the onsets. Overrides the three flags above. This
    # is the recommended setting for real recordings — no per-clip tuning needed.
    auto_rhythm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> PipelineConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
