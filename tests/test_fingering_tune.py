"""Tests for fingering config, cache, and cached eval."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from aitabs.eval.cache import AudioCacheEntry
from aitabs.pipeline.audio.tempo import TempoAnalysis
from aitabs.pipeline.mapping.fingering import map_sequence, map_sequence_greedy, transition_cost
from aitabs.pipeline.mapping.fingering_config import FingeringConfig, DEFAULT_FINGERING_CONFIG
from aitabs.pipeline.mapping.registry import apply_fingering


def test_fingering_config_roundtrip():
    cfg = FingeringConfig(mapper="greedy", string_cost=1.5, max_fret_span=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "finger.json"
        cfg.save_json(path)
        loaded = FingeringConfig.load_json(path)
    assert loaded.mapper == "greedy"
    assert loaded.string_cost == 1.5
    assert loaded.max_fret_span == 3


def test_transition_cost_respects_config():
    cfg = FingeringConfig(string_cost=2.0, fret_cost=0.5, open_string_bonus=0.0)
    assert transition_cost((2, 5), (3, 5), cfg) == 2.0
    assert transition_cost((2, 5), (2, 7), cfg) == 1.0


def test_apply_fingering_greedy():
    notes = [{"start": 0.0, "end": 0.5, "pitch_midi": 57, "confidence": 0.9}]
    cfg = FingeringConfig(mapper="greedy")
    tab = apply_fingering(notes, cfg)
    assert len(tab) == 1


def test_audio_cache_roundtrip():
    tempo = TempoAnalysis(
        bpm=120.0,
        beat_times=np.array([0.0, 0.5, 1.0]),
        beat_period=0.5,
        origin=0.0,
        ticks_per_beat=4,
        confidence=0.9,
    )
    entry = AudioCacheEntry(
        clip_id="test_clip",
        filtered_notes=[
            {"start": 0.0, "end": 0.5, "pitch_midi": 57, "confidence": 0.9},
        ],
        tempo_analysis=tempo,
        reference_bpm=120.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test_clip.json"
        entry.save(path)
        loaded = AudioCacheEntry.load(path)
    assert loaded.clip_id == "test_clip"
    assert len(loaded.filtered_notes) == 1
    assert loaded.tempo_analysis.bpm == 120.0


def test_map_sequence_defaults_match_viterbi():
    notes = [
        {"start": 0.0, "end": 0.5, "pitch_midi": 57, "confidence": 0.9},
        {"start": 1.0, "end": 1.5, "pitch_midi": 59, "confidence": 0.9},
    ]
    viterbi = map_sequence(notes)
    greedy = map_sequence_greedy(notes, config=DEFAULT_FINGERING_CONFIG)
    assert len(viterbi) == len(greedy) == 2
