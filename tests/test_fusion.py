"""Tests for the audio+vision fusion layer (Stage-1)."""

from __future__ import annotations

from aitabs.pipeline.fusion.fusion import fuse, source_distribution

# MIDI 50 (D3) is playable at (string, fret) = (0,10), (1,5), (2,0) in standard tuning.
_NOTE = {"start": 1.00, "end": 1.40, "pitch_midi": 50, "confidence": 0.9}


def test_vision_resolves_to_the_touched_position():
    # A finger is on (1, 5) around the onset -> vision picks it over the lowest fret.
    frames = [
        {"timestamp": 0.98, "cells": [(1, 5)]},
        {"timestamp": 1.01, "cells": [(1, 5)]},
    ]
    out = fuse([_NOTE], frames)
    assert len(out) == 1
    assert (out[0]["string"], out[0]["fret"]) == (1, 5)
    assert out[0]["source"] == "vision"
    assert out[0]["pitch_midi"] == 50


def test_fallback_is_lowest_fret_when_no_vision():
    out = fuse([_NOTE], [])
    assert (out[0]["string"], out[0]["fret"]) == (2, 0)  # lowest playable fret
    assert out[0]["source"] == "audio_fallback"


def test_vision_only_counts_playable_positions():
    # Finger seen on a cell that CANNOT produce this pitch -> ignored, falls back.
    frames = [{"timestamp": 1.0, "cells": [(4, 3)]}]  # not a valid position for MIDI 50
    out = fuse([_NOTE], frames)
    assert out[0]["source"] == "audio_fallback"


def test_vision_picks_most_frequent_position():
    # Two valid positions touched; the one seen on more frames wins.
    frames = [
        {"timestamp": 0.99, "cells": [(0, 10)]},
        {"timestamp": 1.00, "cells": [(1, 5)]},
        {"timestamp": 1.01, "cells": [(1, 5)]},
    ]
    out = fuse([_NOTE], frames)
    assert (out[0]["string"], out[0]["fret"]) == (1, 5)
    assert out[0]["source"] == "vision"


def test_onset_window_excludes_far_frames():
    # Finger only appears well after the onset window -> not used.
    frames = [{"timestamp": 1.50, "cells": [(1, 5)]}]
    out = fuse([_NOTE], frames, onset_window_sec=0.066)
    assert out[0]["source"] == "audio_fallback"


def test_source_distribution():
    frames = [{"timestamp": 1.0, "cells": [(1, 5)]}]
    n2 = {"start": 3.0, "end": 3.2, "pitch_midi": 50, "confidence": 0.9}  # no vision
    out = fuse([_NOTE, n2], frames)
    dist = source_distribution(out)
    assert abs(dist["vision"] - 0.5) < 1e-9
    assert abs(dist["audio_fallback"] - 0.5) < 1e-9
