"""Step 2/3: license-clean meter + downbeat estimation and bar anchoring."""

from __future__ import annotations

import numpy as np
from guitarpro.models import BeatStatus

from aitabs.export.gp5_layout import Gp5ExportOptions, validate_gp5_song
from aitabs.export.guitarpro_types import ExportNote
from aitabs.export.rhythm_notation import build_song_from_note_values
from aitabs.pipeline.audio.tempo import TempoMap, estimate_meter


def _synth(meter: int, *, bars: int = 8, bpm: float = 120.0, accents=(3, 1, 1)):
    spb = 60.0 / bpm
    beats = np.arange(bars * meter + 1) * spb
    onsets, strengths = [], []
    for bar in range(bars):
        for b in range(meter):
            t = (bar * meter + b) * spb
            for k in range(accents[b % len(accents)]):
                onsets.append(t + 0.001 * k)
                strengths.append(1.0)
    return onsets, strengths, beats


def test_tempo_map_from_beat_times_constant():
    tm = TempoMap.from_beat_times([0.0, 0.5, 1.0, 1.5, 2.0])  # 120 bpm
    assert abs(tm.seconds_to_beats(1.0) - 2.0) < 1e-6


def test_tempo_map_from_beat_times_accelerando():
    tm = TempoMap.from_beat_times([0.0, 0.6, 1.1, 1.5, 1.8, 2.0])  # speeding up
    assert tm.seconds_to_beats(2.0) >= 4.5


def test_estimate_meter_three_four():
    on, st, bt = _synth(3, accents=(3, 1, 1))
    est = estimate_meter(on, bt, onset_strengths=st)
    assert est.numerator == 3
    assert est.downbeat_beat == 0


def test_estimate_meter_four_four():
    on, st, bt = _synth(4, accents=(3, 1, 2, 1))
    est = estimate_meter(on, bt, onset_strengths=st)
    assert est.numerator == 4


def test_estimate_meter_phase_shift():
    # accent on the 2nd beat → downbeat phase shifted, non-zero pickup
    on, st, bt = _synth(3, accents=(1, 3, 1))
    est = estimate_meter(on, bt, onset_strengths=st)
    assert est.numerator == 3
    assert est.downbeat_beat == 1
    assert est.lead_in_beats == (3 - 1) % 3


def test_phase_from_origin_helper():
    """The origin anchor: bar-position of beat[0] = round(beat0/period) % numerator."""
    from aitabs.pipeline.audio.tempo import _phase_from_origin

    spb = 0.5
    assert _phase_from_origin(np.arange(12) * spb, 4) == 0            # starts at t=0
    assert _phase_from_origin(np.arange(12)[2:] * spb, 4) == 2        # first 2 beats dropped
    assert _phase_from_origin(np.arange(12)[3:] * spb, 4) == 3
    assert _phase_from_origin(np.arange(12)[5:] * spb, 3) == 2        # 5 % 3
    assert _phase_from_origin([0.5], 4) is None                      # too short


def test_origin_anchor_recovers_phase_when_accent_flat():
    """Solo-guitar case: weak accent + beats start mid-bar (dropped beats).

    Beat tracking dropped the first 2 beats so beat_times[0] sits at bar-position
    2, and every beat is equally loud (flat accents) so the accent phase is noise
    (confidence < 0.5). The origin anchor (piece starts on a downbeat) recovers
    phase == 2. Bar length is pinned via ``candidates`` (a flat signal carries no
    periodicity to detect it, but real solo guitar does — this isolates phase).
    """
    spb = 0.5
    bt = np.arange(8 * 4 + 1)[2:] * spb  # 4/4, first 2 beats dropped
    onsets = list(bt)  # one flat onset per beat -> no downbeat accent
    est = estimate_meter(onsets, bt, onset_strengths=[1.0] * len(onsets), candidates=(4,))
    assert est.numerator == 4
    assert est.confidence < 0.5           # accent phase is unreliable here
    assert est.downbeat_beat == 2         # recovered from the origin anchor
    assert est.lead_in_beats == (4 - 2) % 4


def test_origin_anchor_disabled_falls_back_to_accent():
    """With origin_downbeat=False the accent phase (0 for a flat signal) is kept."""
    spb = 0.5
    bt = np.arange(8 * 4 + 1)[2:] * spb
    onsets = list(bt)
    est = estimate_meter(
        onsets, bt, onset_strengths=[1.0] * len(onsets),
        candidates=(4,), origin_downbeat=False,
    )
    assert est.downbeat_beat == 0  # accent argmax on flat signal -> phase 0


def test_estimate_meter_too_short_defaults_to_four():
    est = estimate_meter([0.0, 0.5], [0.0, 0.5, 1.0], onset_strengths=None)
    assert est.numerator == 4
    assert est.confidence == 0.0


def test_lead_in_anchors_barline():
    notes = [
        ExportNote(start=i * 0.5, end=i * 0.5 + 0.5, pitch_midi=64, string=5, fret=0, confidence=1.0)
        for i in range(6)
    ]

    def build(lead):
        opts = Gp5ExportOptions(tempo=120, time_signature=(3, 4), duration_mode="note_value", lead_in_beats=lead)
        return build_song_from_note_values(notes, options=opts)

    s1 = build(1)
    validate_gp5_song(s1)
    first = s1.tracks[0].measures[0].voices[0].beats
    assert first[0].status == BeatStatus.rest  # pickup rest
    # without a pickup the first beat is a note
    assert build(0).tracks[0].measures[0].voices[0].beats[0].status == BeatStatus.normal
