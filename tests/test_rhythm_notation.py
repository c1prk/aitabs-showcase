"""Tests for the note-value notation engine (aitabs.export.rhythm_notation)."""

from __future__ import annotations

import pytest

from aitabs.export.rhythm_notation import (
    EIGHTH_BASE,
    STRAIGHT_BASE,
    TICKS_PER_QUARTER,
    TRIPLET_BASE,
    _SINGLE_VALUES,
    _make_single,
    build_note_value_schedule,
    correct_leading_onset_gap,
    estimate_grid_phase,
    notation_for_length,
    offsets_reliable,
    optimize_grid_alignment,
    quantize_onset_ticks,
    quantize_onsets_dominant_grid,
    residual_jitter_beats,
)


def _note(start: float, end: float, pitch: int = 60, string: int = 0, fret: int = 0):
    return {
        "start": start,
        "end": end,
        "pitch_midi": pitch,
        "string": string,
        "fret": fret,
        "confidence": 1.0,
    }


def _total(durations) -> int:
    return sum(int(d.time) for d in durations)


# --- single-value table -------------------------------------------------

def test_single_value_table_is_self_consistent():
    """Every table entry's notated Duration must equal its declared tick length."""
    for ticks, _ in _SINGLE_VALUES:
        dur = _make_single(ticks)
        assert dur is not None
        assert int(dur.time) == ticks, f"{ticks} notated as {int(dur.time)}"


def test_make_single_returns_none_for_off_table_value():
    assert _make_single(200) is None  # not a clean single value


# --- notation_for_length ------------------------------------------------

@pytest.mark.parametrize(
    "length",
    [240, 480, 720, 960, 1440, 1920, 2880, 3840],  # incl dotted-8th/quarter/half
)
def test_notation_clean_single_values(length):
    out = notation_for_length(length, STRAIGHT_BASE)
    assert len(out) == 1, f"{length} should be a single value, got {len(out)}"
    assert _total(out) == length


@pytest.mark.parametrize(
    "length",
    [240, 480, 720, 960, 1200, 1440, 1680, 1920, 2160, 2400, 3120, 3840, 7680],
)
def test_notation_exact_sum_on_straight_grid(length):
    """Decomposition tick total must always equal the requested length exactly."""
    out = notation_for_length(length, STRAIGHT_BASE)
    assert _total(out) == length


def test_notation_tied_span_across_beat():
    # quarter + sixteenth = 1200 ticks; no clean single, must tie.
    out = notation_for_length(1200, STRAIGHT_BASE)
    assert len(out) >= 2
    assert _total(out) == 1200


def test_notation_triplet_eighth_single_value():
    # 320 ticks == eighth triplet on the 12/quarter grid.
    out = notation_for_length(320, TRIPLET_BASE)
    assert len(out) == 1
    assert _total(out) == 320
    assert out[0].tuplet is not None


def test_notation_zero_length_is_empty():
    assert notation_for_length(0, STRAIGHT_BASE) == []


# --- quantize_onset_ticks -----------------------------------------------

def test_quantize_snaps_to_sixteenth_grid():
    # Raw onsets near (but not on) 16th-grid points.
    raw = [0.0, 242.0, 478.0, 961.0]
    snapped = quantize_onset_ticks(raw, enable_triplets=False)
    assert snapped == [0, 240, 480, 960]


def test_quantize_triplets_disabled_keeps_straight():
    # A triplet-eighth position (~320) snaps to a straight-grid step (the coarse
    # quantizer prefers the eighth grid, absorbing jitter), never the triplet 320.
    raw = [0.0, 320.0]
    snapped = quantize_onset_ticks(raw, enable_triplets=False)
    assert snapped[1] != 320
    assert snapped[1] in (240, 480)  # straight 16th or 8th, never a triplet value


def test_quantize_empty():
    assert quantize_onset_ticks([], enable_triplets=False) == []


# --- grid-phase de-bias --------------------------------------------------

def test_estimate_grid_phase_detects_systematic_offset():
    # Steady quarter-note stream sitting a constant ~0.125 beat (120t) early.
    raw = [k * TICKS_PER_QUARTER - 120.0 for k in range(16)]
    phi = estimate_grid_phase(raw, EIGHTH_BASE)
    assert phi == pytest.approx(-120.0, abs=15.0)


def test_estimate_grid_phase_ignores_spread_onsets():
    # Onsets spread across many sub-beat positions (low concentration) -> no shift.
    raw = [0.0, 130.0, 250.0, 470.0, 610.0, 730.0, 950.0, 1180.0]
    assert estimate_grid_phase(raw, EIGHTH_BASE) == 0.0


def test_grid_phase_debias_fixes_straddling_bulk():
    # Beat-space ticks: a steady quarter stream sitting ~0.875 beat (840t) off the
    # downbeat with jitter that straddles the 16th grid (some snap to 720, some to
    # 960) — the exact pattern that printed steady quarters as syncopated
    # dotted/tied values. Removing the estimated phase lands them all on the beat.
    jit = [-30, 40, -50, 20, -40, 60, -20, 50, -60, 30, -10, 45]
    raw = [k * TICKS_PER_QUARTER + 840 + jit[k] for k in range(12)]

    snapped_raw = quantize_onset_ticks(raw, enable_triplets=False)
    raw_offsets = {s % TICKS_PER_QUARTER for s in snapped_raw}
    assert len(raw_offsets) > 1  # straddles the grid without de-bias

    phi = estimate_grid_phase(raw)
    assert phi != 0.0
    snapped_deb = quantize_onset_ticks([t - phi for t in raw], enable_triplets=False)
    deb_offsets = {s % TICKS_PER_QUARTER for s in snapped_deb}
    assert deb_offsets == {0}  # all collapse onto the downbeat


def test_correct_leading_onset_gap_pulls_in_early_first_note():
    # A note one beat too early (gap 2 beats) before a steady quarter stream.
    q = TICKS_PER_QUARTER
    onsets = [0.0] + [2 * q + k * q for k in range(7)]  # 0, then 2,3,4,..,8 beats
    offsets = [t + 0.5 * q for t in onsets]
    fixed_on, _ = correct_leading_onset_gap(onsets, offsets)
    diffs = [b - a for a, b in zip(sorted(set(fixed_on)), sorted(set(fixed_on))[1:])]
    # First gap is now one quarter, matching the rest of the stream.
    assert diffs[0] == pytest.approx(q)
    assert all(d == pytest.approx(q) for d in diffs)


def test_correct_leading_onset_gap_uses_local_spacing():
    # Quarters at the start, eighths later: the (global) median is an eighth, but
    # the leading-gap correction must use the local quarter spacing, not over-pull.
    q = TICKS_PER_QUARTER
    e = q // 2
    onsets = [0.0, 2 * q, 3 * q, 4 * q, 5 * q]  # leading gap of 2 quarters
    onsets += [5 * q + e * k for k in range(1, 9)]  # eighth run afterwards
    offsets = [t + e for t in onsets]
    fixed_on, _ = correct_leading_onset_gap(onsets, offsets)
    su = sorted(set(fixed_on))
    assert su[1] - su[0] == pytest.approx(q)  # pulled to a quarter, not an eighth


def test_correct_leading_onset_gap_noop_for_steady_stream():
    q = TICKS_PER_QUARTER
    onsets = [k * q for k in range(8)]
    offsets = [t + 0.5 * q for t in onsets]
    assert correct_leading_onset_gap(onsets, offsets) == (onsets, offsets)


def test_schedule_align_grid_phase_is_a_noop_for_on_grid_input():
    # Steady on-grid quarters must be unaffected by the de-bias (phi == 0).
    notes = [_note(k * 0.5, k * 0.5 + 0.5) for k in range(8)]
    on = build_note_value_schedule(notes, bpm=120.0, enable_triplets=False)[0]
    off = build_note_value_schedule(
        notes, bpm=120.0, enable_triplets=False, align_grid_phase=False
    )[0]
    assert [b.tick_start for b in on] == [b.tick_start for b in off]


# --- build_note_value_schedule ------------------------------------------

def test_schedule_inserts_rest_on_clear_gap():
    # One quarter-note worth of sound at 120bpm (0.5s), then 0.5s of silence,
    # then another note. The gap should become a rest beat.
    bpm = 120.0
    notes = [_note(0.0, 0.5), _note(1.0, 1.5)]
    beats, base = build_note_value_schedule(notes, bpm, trim_leading_silence=True)
    assert base == STRAIGHT_BASE
    assert any(b.is_rest for b in beats), "expected a rest in the gap"
    # Total ticks contiguous and increasing.
    cursor = 0
    for b in beats:
        assert b.tick_start == cursor
        cursor += b.duration_ticks


def test_schedule_continuous_notes_no_rest():
    # Back-to-back quarter notes (no gap) -> no rests.
    bpm = 120.0
    notes = [_note(0.0, 0.5), _note(0.5, 1.0), _note(1.0, 1.5)]
    beats, _ = build_note_value_schedule(notes, bpm, trim_leading_silence=True)
    assert not any(b.is_rest for b in beats)
    assert sum(1 for b in beats if not b.is_rest) == 3


def test_schedule_groups_chord():
    # Two notes at the same onset -> a single beat with two notes.
    bpm = 120.0
    notes = [_note(0.0, 0.5, pitch=60), _note(0.0, 0.5, pitch=64)]
    beats, _ = build_note_value_schedule(notes, bpm, trim_leading_silence=True)
    attacks = [b for b in beats if not b.is_rest]
    assert len(attacks) == 1
    assert len(attacks[0].notes) == 2


def test_schedule_empty_input():
    beats, base = build_note_value_schedule([], 120.0)
    assert beats == []
    assert base == STRAIGHT_BASE


def test_constants():
    assert TICKS_PER_QUARTER == 960
    assert STRAIGHT_BASE == 240
    assert TRIPLET_BASE == 80


# --- dominant-grid snapping (live recordings) --------------------------

def test_dominant_grid_snaps_jittered_quarters_to_the_beat():
    # Steady quarter notes with up to ~70 ms jitter at 120 BPM (1 beat = 960 ticks).
    jitter = [0.0, 0.12, -0.09, 0.14, -0.05, 0.11, -0.13, 0.08]
    onsets = [i * TICKS_PER_QUARTER + j * TICKS_PER_QUARTER for i, j in enumerate(jitter)]
    snapped = quantize_onsets_dominant_grid(onsets)
    # The coarse (quarter) grid must win and pull every onset onto an integer beat.
    assert snapped == [i * TICKS_PER_QUARTER for i in range(len(onsets))]


def test_dominant_grid_keeps_real_eighths():
    # Genuine eighth-note stream: the quarter grid has large error, so the eighth
    # grid must be chosen and the off-beats preserved.
    onsets = [i * (TICKS_PER_QUARTER // 2) for i in range(8)]
    snapped = quantize_onsets_dominant_grid(onsets)
    assert snapped == onsets
    assert any(s % TICKS_PER_QUARTER != 0 for s in snapped)


def test_legato_fills_ioi_instead_of_inserting_rests():
    # Notes released early (short sustain) but one per beat: legato must not add rests.
    bpm = 120.0
    notes = [_note(0.0, 0.2), _note(0.5, 0.7), _note(1.0, 1.2)]
    beats, _ = build_note_value_schedule(notes, bpm, legato=True)
    assert not any(b.is_rest for b in beats)
    # Without legato the early releases would create rests.
    beats_strict, _ = build_note_value_schedule(notes, bpm, legato=False)
    assert any(b.is_rest for b in beats_strict)


# --- joint tempo-spacing + phase alignment (optimize_grid) ---------------

def test_optimize_grid_is_noop_on_exact_onsets():
    # MIDI-render-style exact eighths (test1): the identity grid is already best,
    # so the search must return (1.0, 0.0) and not perturb tight takes.
    onsets = [float(i * EIGHTH_BASE) for i in range(16)]
    scale, phase = optimize_grid_alignment(onsets)
    assert scale == 1.0 and phase == 0.0


def test_optimize_grid_recovers_tempo_spacing_error():
    # A 3%-fast BPM estimate spaces a steady eighth-note stream 1.03x too wide, so
    # the per-beat snapper notates spurious off-beat sixteenths. The joint search
    # must shrink the grid (~0.97) so every onset lands back on the eighth grid.
    onsets = [i * EIGHTH_BASE * 1.03 for i in range(16)]
    scale, phase = optimize_grid_alignment(onsets)
    assert scale < 1.0
    corrected = [t * scale - phase for t in onsets]
    snapped = quantize_onset_ticks(corrected, enable_triplets=False)
    assert snapped == [i * EIGHTH_BASE for i in range(16)]


def test_optimize_grid_preserves_genuine_sixteenths():
    # The search band is too narrow (+-6%) to reinterpret a real sixteenth as an
    # eighth (that needs ~2x), so a true sixteenth-note run keeps 16 distinct steps.
    onsets = [float(i * STRAIGHT_BASE) for i in range(16)]
    scale, phase = optimize_grid_alignment(onsets)
    corrected = [t * scale - phase for t in onsets]
    snapped = quantize_onset_ticks(corrected, enable_triplets=False)
    assert len({round(s / STRAIGHT_BASE) for s in snapped}) == 16


def test_optimize_grid_too_few_onsets_is_noop():
    assert optimize_grid_alignment([0.0, 480.0, 960.0]) == (1.0, 0.0)


def test_residual_jitter_rises_with_timing_noise():
    tight = [float(i * EIGHTH_BASE) for i in range(16)]
    loose = [i * EIGHTH_BASE + (40 if i % 2 else -40) for i in range(16)]
    assert residual_jitter_beats(tight) == 0.0
    assert residual_jitter_beats(loose) > residual_jitter_beats(tight)


def test_schedule_optimize_grid_fixes_over_quantization():
    # Eighth notes whose feed-bpm is 3% off (the real-recording over-quantization
    # bug): without optimize_grid some onsets fall on off-beat sixteenths; with it,
    # all 16 notes read as clean eighths.
    bpm = 96.0
    sec_per_eighth = 60.0 / bpm / 2.0
    notes = [_note(i * sec_per_eighth, i * sec_per_eighth + sec_per_eighth * 0.9)
             for i in range(16)]
    bad_bpm = bpm * 1.03

    beats_off, _ = build_note_value_schedule(notes, bad_bpm, align_grid_phase=False)
    onsets_off = [b.tick_start for b in beats_off if not b.is_rest]
    assert any(o % EIGHTH_BASE != 0 for o in onsets_off)  # bug reproduced

    beats_on, _ = build_note_value_schedule(notes, bad_bpm, align_grid_phase=False,
                                            optimize_grid=True)
    onsets_on = [b.tick_start for b in beats_on if not b.is_rest]
    assert onsets_on == [i * EIGHTH_BASE for i in range(16)]


def test_schedule_optimize_grid_noop_on_tight_take():
    # test1-style exact take at the correct tempo: optimize_grid must not change the
    # schedule (no regression of clean MIDI-render rhythm).
    bpm = 96.0
    sec_per_eighth = 60.0 / bpm / 2.0
    notes = [_note(i * sec_per_eighth, i * sec_per_eighth + sec_per_eighth * 0.9)
             for i in range(16)]
    off, _ = build_note_value_schedule(notes, bpm, align_grid_phase=False)
    on, _ = build_note_value_schedule(notes, bpm, align_grid_phase=False, optimize_grid=True)
    assert [(b.tick_start, b.duration_ticks, b.is_rest) for b in off] == \
           [(b.tick_start, b.duration_ticks, b.is_rest) for b in on]


# --- self-gating optimize_grid + offset reliability ----------------------

def test_optimize_grid_self_gates_on_mixed_rhythm_no_tempo_error():
    # Mixed eighths/sixteenths with NO tempo error: the penalty alone would relabel
    # the minority sixteenths as eighths, but the raw-fit guard must make it a no-op.
    onsets: list[float] = []
    t = 0.0
    for d in ([EIGHTH_BASE] * 4 + [STRAIGHT_BASE] * 2) * 4:
        onsets.append(float(t))
        t += d
    assert optimize_grid_alignment(onsets) == (1.0, 0.0)


def test_optimize_grid_still_fires_on_real_tempo_error():
    onsets = [i * EIGHTH_BASE * 1.03 for i in range(16)]
    scale, _ = optimize_grid_alignment(onsets)
    assert scale < 1.0  # a genuine spacing error is still corrected


def test_offsets_reliable_false_for_uniform_short_sustain():
    # Eighth-spaced onsets but every note reports the same tiny sustain (noise).
    onsets = [i * 480.0 for i in range(12)]
    offsets = [o + 90.0 for o in onsets]  # ~0.19 beat sustain regardless of IOI
    assert offsets_reliable(onsets, offsets) is False


def test_offsets_reliable_true_when_notes_sustain():
    onsets = [i * 480.0 for i in range(12)]
    offsets = [o + 450.0 for o in onsets]  # nearly fills the IOI
    assert offsets_reliable(onsets, offsets) is True


# --- auto_rhythm: one mode, no per-clip flags ----------------------------

def _hist16(beats):
    h = {}
    for b in beats:
        if b.is_rest:
            h["REST"] = h.get("REST", 0) + 1
        else:
            h[round(b.duration_ticks / STRAIGHT_BASE)] = \
                h.get(round(b.duration_ticks / STRAIGHT_BASE), 0) + 1
    return h


def test_auto_rhythm_preserves_mixed_rhythm_with_unreliable_offsets():
    # vidtest1 regime: 33% sixteenths, all notes report a constant short sustain.
    # auto must keep the sixteenths (no dominant-grid flatten) and suppress the
    # spurious rests (auto-legato), without any flags from the caller.
    bpm = 96.0
    spb = 60.0 / bpm
    onsets: list[float] = []
    t = 0.0
    for d in [0.5, 0.5, 0.5, 0.5, 0.25, 0.25] * 4:
        onsets.append(t * spb)
        t += d
    notes = [_note(s, s + 0.12) for s in onsets]  # uniform short (unreliable) offsets
    beats, _ = build_note_value_schedule(notes, bpm, auto_rhythm=True)
    h = _hist16(beats)
    assert h.get(1, 0) >= 4          # sixteenths survive
    assert h.get(2, 0) >= 8          # eighths present
    assert h.get("REST", 0) <= 2     # no staccato-rest spray


def test_auto_rhythm_inserts_real_rests_when_offsets_informative():
    bpm = 96.0
    spb = 60.0 / bpm
    onsets = [i * 1.0 * spb for i in range(8)]            # quarter spacing
    sustains = [0.9, 0.2, 0.9, 0.2, 0.9, 0.2, 0.9, 0.2]   # alternating held/short
    notes = [_note(s, s + d) for s, d in zip(onsets, sustains)]
    beats, _ = build_note_value_schedule(notes, bpm, auto_rhythm=True)
    assert any(b.is_rest for b in beats)


def test_auto_rhythm_corrects_tempo_spacing_error():
    bpm = 96.0
    spb = 60.0 / bpm
    onsets = [i * 0.5 * spb for i in range(16)]
    notes = [_note(s, s + 0.4) for s in onsets]
    beats, _ = build_note_value_schedule(notes, bpm * 1.03, auto_rhythm=True)
    steps = [b.tick_start // STRAIGHT_BASE for b in beats if not b.is_rest]
    assert all(s % 2 == 0 for s in steps)   # all land on the eighth grid


def test_auto_rhythm_keeps_clean_eighths_simple():
    bpm = 96.0
    spb = 60.0 / bpm
    onsets = [i * 0.5 * spb for i in range(16)]
    notes = [_note(s, s + 0.45) for s in onsets]
    beats, _ = build_note_value_schedule(notes, bpm, auto_rhythm=True)
    assert _hist16(beats).get(1, 0) == 0    # no manufactured sixteenths


def test_auto_rhythm_preserves_genuine_sixteenth_run():
    bpm = 96.0
    spb = 60.0 / bpm
    onsets = [i * 0.25 * spb for i in range(16)]
    notes = [_note(s, s + 0.2) for s in onsets]
    beats, _ = build_note_value_schedule(notes, bpm, auto_rhythm=True)
    steps = {b.tick_start // STRAIGHT_BASE for b in beats if not b.is_rest}
    assert len(steps) == 16


# --- jitter_adaptive per-beat off-grid guard (Fix 2) ---------------------
# Math reference:  for a beat [0, 480, 720] (eighth + two sixteenths) with
# onset jitter j ticks, eighth_cost=240 always, sixteenth_cost=2j, diff=240-2j.
# Guard fires when the sixteenth onset is >0.28×480=134 ticks from nearest eighth,
# i.e. jitter < 106 ticks (69 ms).  After firing, effective_margin=0.10 →
# threshold=0.10×240×3=72, so diff>72 requires j<84 ticks (55 ms).

def test_quantize_jitter_adaptive_keeps_mixed_beat_sixteenth():
    # Beat [0, 480, 720] with realistic audio jitter (60 ticks ≈ 39 ms at 96 BPM).
    # Genuine positions: downbeat, eighth, dotted-eighth (sixteenth position).
    # Without the guard, a high coarse_margin (0.70) produces threshold=504>diff=120
    # and flattens the sixteenth.  With jitter_adaptive the off-grid check fires
    # (error=180>134) → effective_margin=0.10, threshold=72 < diff=120 → kept.
    j = 60  # ticks ≈ 39 ms
    raw = [0.0, float(EIGHTH_BASE + j), float(EIGHTH_BASE + STRAIGHT_BASE - j)]
    # [0, 540, 660];  true positions: [0, 480, 720]

    high_margin = 0.70

    snapped_no = quantize_onset_ticks(raw, enable_triplets=False, coarse_margin=high_margin)
    snapped_ad = quantize_onset_ticks(raw, enable_triplets=False, coarse_margin=high_margin,
                                      jitter_adaptive=True)

    assert snapped_no[2] != 720, "without adaptive guard, sixteenth should be suppressed"
    assert snapped_ad[2] == 720, f"with adaptive guard, expected 720, got {snapped_ad[2]}"


def test_quantize_jitter_adaptive_noop_when_all_on_eighth_grid():
    # Jittered eighth-note pairs: all onsets land near eighth positions.  The off-grid
    # check must NOT fire (max error=40 < 134 ticks) so the elevated margin is kept,
    # absorbing the jitter and preventing manufactured sixteenths.
    j = 40  # ticks ≈ 26 ms — on-grid jitter, well under the 134-tick threshold
    raw = [0.0, float(EIGHTH_BASE + j), float(TICKS_PER_QUARTER),
           float(TICKS_PER_QUARTER + EIGHTH_BASE - j)]

    high_margin = 0.70
    snapped = quantize_onset_ticks(raw, enable_triplets=False, coarse_margin=high_margin,
                                   jitter_adaptive=True)
    for s in snapped:
        assert s % EIGHTH_BASE == 0, f"jittered eighth {s} should snap to eighth grid"


def test_auto_rhythm_with_audio_jitter_keeps_mixed_beat_sixteenths():
    # vidtest1 regime: mix of pure-eighth beats and beats with a sixteenth run.
    # Audio jitter (±40 ms ≈ ±62 ticks at 96 BPM) elevates the global coarse_margin;
    # without jitter_adaptive the sixteenth beats are flattened.  auto_rhythm=True
    # activates the per-beat off-grid guard and must keep the sixteenths.
    import random
    random.seed(42)
    bpm = 96.0
    spb = 60.0 / bpm
    eighth = spb / 2
    sixteenth = spb / 4
    noise_sec = 0.040  # ±40 ms of onset jitter

    # 2 pure-eighth beats then 2 beats with eighth+sixteenth+sixteenth
    t = 0.0
    onsets = []
    for _ in range(2):
        onsets.append(t + random.uniform(-noise_sec, noise_sec)); t += eighth
        onsets.append(t + random.uniform(-noise_sec, noise_sec)); t += eighth
    for _ in range(2):
        onsets.append(t + random.uniform(-noise_sec, noise_sec)); t += eighth
        onsets.append(t + random.uniform(-noise_sec, noise_sec)); t += sixteenth
        onsets.append(t + random.uniform(-noise_sec, noise_sec)); t += sixteenth

    notes = [_note(s, s + 0.15) for s in onsets]
    beats, _ = build_note_value_schedule(notes, bpm, auto_rhythm=True)
    h = _hist16(beats)
    assert h.get(1, 0) >= 2, f"sixteenths must survive audio jitter, histogram={h}"


def test_auto_rhythm_overrides_snap_dominant_grid():
    # Even if a caller leaves snap_dominant_grid on, auto must not flatten.
    bpm = 96.0
    spb = 60.0 / bpm
    onsets: list[float] = []
    t = 0.0
    for d in [0.5, 0.5, 0.5, 0.5, 0.25, 0.25] * 4:
        onsets.append(t * spb)
        t += d
    notes = [_note(s, s + 0.12) for s in onsets]
    beats, _ = build_note_value_schedule(
        notes, bpm, auto_rhythm=True, snap_dominant_grid=True
    )
    assert _hist16(beats).get(1, 0) >= 4    # sixteenths still survive
