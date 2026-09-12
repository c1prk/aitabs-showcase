"""Cache audio-stage outputs so fingering tuning skips BasicPitch/Demucs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aitabs.pipeline.audio.pitch import NoteEvent
from aitabs.pipeline.audio.tempo import TempoAnalysis


@dataclass
class AudioCacheEntry:
    """Per-clip notes + tempo after audio/tempo cleaning."""

    clip_id: str
    filtered_notes: list[NoteEvent]
    tempo_analysis: TempoAnalysis
    reference_bpm: float | None = None
    audio_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        ta = self.tempo_analysis
        return {
            "clip_id": self.clip_id,
            "filtered_notes": [dict(n) for n in self.filtered_notes],
            "tempo_analysis": {
                "bpm": ta.bpm,
                "beat_times": ta.beat_times.tolist(),
                "beat_period": ta.beat_period,
                "origin": ta.origin,
                "ticks_per_beat": ta.ticks_per_beat,
                "confidence": ta.confidence,
                "phase_template": list(ta.phase_template) if ta.phase_template else None,
            },
            "reference_bpm": self.reference_bpm,
            "audio_path": self.audio_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioCacheEntry:
        import numpy as np

        ta_raw = data["tempo_analysis"]
        tempo = TempoAnalysis(
            bpm=float(ta_raw["bpm"]),
            beat_times=np.asarray(ta_raw["beat_times"], dtype=float),
            beat_period=float(ta_raw["beat_period"]),
            origin=float(ta_raw["origin"]),
            ticks_per_beat=int(ta_raw["ticks_per_beat"]),
            confidence=float(ta_raw["confidence"]),
            phase_template=tuple(ta_raw["phase_template"])
            if ta_raw.get("phase_template")
            else None,
        )
        notes: list[NoteEvent] = [
            NoteEvent(
                start=float(n["start"]),
                end=float(n["end"]),
                pitch_midi=int(n["pitch_midi"]),
                confidence=float(n["confidence"]),
            )
            for n in data["filtered_notes"]
        ]
        return cls(
            clip_id=str(data["clip_id"]),
            filtered_notes=notes,
            tempo_analysis=tempo,
            reference_bpm=data.get("reference_bpm"),
            audio_path=data.get("audio_path"),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AudioCacheEntry:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def cache_path_for_clip(cache_dir: str | Path, clip_id: str) -> Path:
    return Path(cache_dir) / f"{clip_id}.json"
