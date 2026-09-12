"""Tests for pluggable detector dispatch."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aitabs.pipeline.audio.pitch import (
    BasicPitchDetector,
    KongDetector,
    RecallFirstBasicPitchDetector,
    get_detector,
)

_DETECT_KWARGS = dict(
    onset_threshold=0.5,
    frame_threshold=0.3,
    confidence_threshold=0.42,
    minimum_note_length_ms=58.0,
    minimum_frequency_hz=82.0,
    maximum_frequency_hz=1400.0,
    melodia_trick=True,
)


def test_get_detector_basic_pitch():
    assert isinstance(get_detector("basic_pitch"), BasicPitchDetector)


def test_get_detector_recall_first():
    assert isinstance(get_detector("recall_first"), RecallFirstBasicPitchDetector)


def test_get_detector_unknown_raises():
    with pytest.raises(ValueError, match="Unknown detector"):
        get_detector("nonexistent_model")


def test_recall_first_overrides_onset_and_confidence():
    """RecallFirstBasicPitchDetector must ignore the caller's thresholds and use its own."""
    calls: list[dict] = []

    def capture(self, audio_path: str, **kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return []

    with patch.object(BasicPitchDetector, "detect", capture):
        RecallFirstBasicPitchDetector().detect("fake.wav", **_DETECT_KWARGS)

    assert len(calls) == 1
    assert calls[0]["onset_threshold"] == pytest.approx(RecallFirstBasicPitchDetector._ONSET)
    assert calls[0]["confidence_threshold"] == pytest.approx(RecallFirstBasicPitchDetector._CONFIDENCE)


def test_recall_first_passes_through_other_kwargs():
    """Non-threshold kwargs must be forwarded unchanged."""
    calls: list[dict] = []

    def capture(self, audio_path: str, **kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return []

    with patch.object(BasicPitchDetector, "detect", capture):
        RecallFirstBasicPitchDetector().detect("fake.wav", **_DETECT_KWARGS)

    assert calls[0]["frame_threshold"] == pytest.approx(_DETECT_KWARGS["frame_threshold"])
    assert calls[0]["minimum_note_length_ms"] == pytest.approx(_DETECT_KWARGS["minimum_note_length_ms"])
    assert calls[0]["melodia_trick"] == _DETECT_KWARGS["melodia_trick"]


def test_recall_first_lower_thresholds_than_basic():
    """Sanity: recall_first thresholds must be strictly below basic_pitch defaults."""
    assert RecallFirstBasicPitchDetector._CONFIDENCE < _DETECT_KWARGS["confidence_threshold"]
    assert RecallFirstBasicPitchDetector._ONSET < _DETECT_KWARGS["onset_threshold"]


def test_get_detector_kong():
    """get_detector('kong') returns a KongDetector without loading the model."""
    d = get_detector("kong")
    assert isinstance(d, KongDetector)


def test_get_detector_kong_with_model_path():
    """get_detector('kong', model_path=...) passes path to KongDetector."""
    d = get_detector("kong", model_path="/tmp/fake.pth")
    assert isinstance(d, KongDetector)
    assert d._checkpoint_path == "/tmp/fake.pth"


def test_kong_detector_class_cache_shared():
    """Multiple KongDetector instances share the same _shared_models dict."""
    d1 = KongDetector(checkpoint_path="/fake/path.pth")
    d2 = KongDetector(checkpoint_path="/fake/path.pth")
    assert d1._shared_models is d2._shared_models


def test_kong_hz_to_midi():
    assert KongDetector._hz_to_midi(440.0) == 69   # A4
    assert KongDetector._hz_to_midi(82.41) == 40   # E2 (open low E)
    assert KongDetector._hz_to_midi(329.63) == 64  # E4
