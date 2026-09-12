"""Rhythm notation engine: onset-grid quantization, IOI note values, rests.

This module turns timed notes into a *readable* Guitar Pro beat schedule by
treating rhythm transcription as the two-stage problem described in the
audio-to-score literature (Cambouropoulos 2000; Cemgil & Desain 2000;
Nakamura et al. 2017):

    1. Onset quantization   — snap each attack to a metrical grid, choosing,
                              per beat, the subdivision (straight vs triplet)
                              that minimises total timing deviation.
    2. Note-value / rest    — assign each note a notated length from the
                              inter-onset interval (IOI) and the note's own
                              sounding duration, inserting a rest whenever the
                              note is released well before the next attack.

Durations are then decomposed into Guitar Pro ``Duration`` objects that sum
*exactly*, prefer a single dotted value, split at beat boundaries with ties,
and use tuplet values for triplet-grid lengths.

The legacy ``to_next_onset`` / ``grid_unit`` modes in :mod:`gp5_layout` are
left untouched; this engine is opt-in via ``duration_mode="note_value"``.
"""

from __future__ import annotations

from dataclasses import dataclass

from guitarpro.models import (
    Beat,
    BeatStatus,
    Duration,
    Note,
    NoteType,
    Tuplet,
    Voice,
)

from .guitarpro_types import ExportNote, aitabs_string_to_gp

TICKS_PER_QUARTER = Duration.quarterTime  # 960

# Grid bases (ticks per smallest step within a beat).
#   240 -> 16th-note grid (straight only): 8th=480, 16th=240
#    80 -> 12 steps/quarter: represents straight 8th/16th AND eighth/16th
#          triplets exactly (8th-trip=320, 16th-trip=160, quarter-trip=640)
STRAIGHT_BASE = TICKS_PER_QUARTER // 4   # 240
TRIPLET_BASE = TICKS_PER_QUARTER // 12   # 80
EIGHTH_BASE = TICKS_PER_QUARTER // 2     # 480 (coarse straight grid)


def _dur(value: int, *, dotted: bool = False, triplet: bool = False) -> Duration:
    d = Duration(value=value, isDotted=dotted)
    if triplet:
        d.tuplet = Tuplet(enters=3, times=2)
    return d


# Single-value notation table: exact tick length -> factory producing a fresh
# Duration. Ordered largest-first for greedy fill. Covers plain, dotted, and
# common triplet values; dotted-32nd/64th and rarer tuplets are intentionally
# omitted to keep notation legible.
_SINGLE_VALUES: list[tuple[int, dict]] = [
    (3840, dict(value=Duration.whole)),
    (2880, dict(value=Duration.half, dotted=True)),
    (1920, dict(value=Duration.half)),
    (1440, dict(value=Duration.quarter, dotted=True)),
    (960, dict(value=Duration.quarter)),
    (720, dict(value=Duration.eighth, dotted=True)),
    (640, dict(value=Duration.quarter, triplet=True)),
    (480, dict(value=Duration.eighth)),
    (360, dict(value=Duration.sixteenth, dotted=True)),
    (320, dict(value=Duration.eighth, triplet=True)),
    (240, dict(value=Duration.sixteenth)),
    (160, dict(value=Duration.sixteenth, triplet=True)),
    (120, dict(value=Duration.thirtySecond)),
    (80, dict(value=Duration.thirtySecond, triplet=True)),
    (60, dict(value=Duration.sixtyFourth)),
]
_SINGLE_BY_TICKS = {t: kw for t, kw in _SINGLE_VALUES}


def _make_single(ticks: int) -> Duration | None:
    kw = _SINGLE_BY_TICKS.get(ticks)
    return _dur(**kw) if kw is not None else None


def notation_for_length(length: int, base_unit: int) -> list[Duration]:
    """Decompose a grid-aligned length into tied Duration objects (exact sum).

    ``length`` must be a positive multiple of ``base_unit``. A single clean
    value (including dotted / triplet) is returned when one exists; otherwise
    the length is filled greedily with values that stay on the grid, producing
    a tied sequence whose tick total equals ``length`` exactly.
    """
    if length <= 0:
        return []
    single = _make_single(length)
    if single is not None and length % base_unit == 0:
        return [single]

    out: list[Duration] = []
    remaining = length
    # Candidate values that land cleanly on this beat's grid, largest first.
    candidates = [t for t, _ in _SINGLE_VALUES if t % base_unit == 0]
    while remaining > 0:
        for t in candidates:
            if t <= remaining and (remaining - t) % base_unit == 0:
                out.append(_make_single(t))  # type: ignore[arg-type]
                remaining -= t
                break
        else:  # pragma: no cover - guaranteed reachable only on bad input
            out.append(_dur(value=Duration.sixtyFourth))
            remaining -= 60
    return out


def estimate_grid_phase(
    onset_ticks: list[float],
    base: int = EIGHTH_BASE,
    *,
    min_concentration: float = 0.5,
) -> float:
    """Estimate a constant sub-beat phase offset (in ticks) of the onset grid.

    Detected onsets often sit a fixed fraction of a beat off the metrical grid
    (detector latency + the arbitrary "first onset == tick 0" anchor). Left in,
    that constant offset makes otherwise-steady notes straddle grid points, so
    the per-beat snapper notates them as syncopated dotted/tied values
    (e.g. a stream of quarter notes printed as dotted-eighth + tie).

    We model it as a single phase ``phi`` in ``(-base/2, base/2]`` and return the
    circular mean of ``onset mod base`` — the closed-form minimiser of squared
    circular snap error to a grid of spacing ``base``. ``base`` defaults to the
    eighth-note step so genuine eighth/quarter material yields ``phi ≈ 0`` (the
    grid is left untouched); only a real systematic offset moves it.

    The correction is applied only when the phase is genuinely *systematic*:
    ``min_concentration`` gates on the resultant length ``R`` (0–1) of the
    circular distribution, so material whose onsets are spread across many
    sub-beat positions (e.g. dense sixteenth runs) — where the mean phase is
    meaningless — is left untouched (``phi = 0``).
    """
    if len(onset_ticks) < 4 or base <= 0:
        return 0.0
    import math

    sin_sum = 0.0
    cos_sum = 0.0
    for t in onset_ticks:
        ang = ((t % base) / base) * 2.0 * math.pi
        sin_sum += math.sin(ang)
        cos_sum += math.cos(ang)
    concentration = math.hypot(sin_sum, cos_sum) / len(onset_ticks)
    if concentration < min_concentration:
        return 0.0
    phi = math.atan2(sin_sum, cos_sum) / (2.0 * math.pi) * base
    return float(phi)


def correct_leading_onset_gap(
    onsets: list[float],
    offsets: list[float],
    *,
    outlier_factor: float = 1.75,
    min_onsets: int = 6,
) -> tuple[list[float], list[float]]:
    """Pull the stream back when the *first* inter-onset gap is an outlier.

    A single low/ringing note detected a beat or two before the real first
    downbeat (a common transcription artifact at audio start) gets a huge IOI
    and is notated as an over-long first note that visually breaks the rhythm.
    When the leading gap exceeds ``outlier_factor`` × the median IOI, we shift
    everything *after* the first onset earlier by the excess, so the first note
    gets a normal (median-length) value while the rest of the grid is untouched.
    No-op for steady material (first gap ≈ median) and for short inputs.
    """
    if len(onsets) < min_onsets:
        return onsets, offsets
    uniq = sorted(set(onsets))
    if len(uniq) < 5:
        return onsets, offsets
    diffs = [b - a for a, b in zip(uniq, uniq[1:])]
    import statistics

    # Compare the leading gap to the *local* spacing of the notes right after it
    # (a global median is wrong when the piece mixes quarters and eighths). An odd
    # window keeps the median an actual gap value rather than a quarter/eighth average.
    local = diffs[1:6]
    if len(local) < 3:
        return onsets, offsets
    med = statistics.median_low(local)
    if med <= 0 or diffs[0] <= outlier_factor * med:
        return onsets, offsets

    delta = diffs[0] - med
    first = uniq[0]
    mask = [t > first + 1e-6 for t in onsets]
    onsets = [t - delta if m else t for t, m in zip(onsets, mask)]
    offsets = [o - delta if m else o for o, m in zip(offsets, mask)]
    return onsets, offsets


def optimize_grid_alignment(
    onset_ticks: list[float],
    *,
    max_tempo_dev: float = 0.06,
    n_tempo: int = 13,
    n_phase: int = 16,
    sixteenth_penalty: float = 0.5,
    min_raw_gain: float = 0.20,
) -> tuple[float, float]:
    """Jointly refine global tempo *spacing* and sub-beat *phase* to the grid.

    The phase de-bias (:func:`estimate_grid_phase`) fixes a constant offset but
    **not** a tempo-spacing error: when the estimated BPM is a few % off, onsets
    drift across the grid and steady eighth-note material gets notated as 16ths +
    syncopation. Here we search a *small* tempo band (so a real sixteenth can never
    be reinterpreted as an eighth — that needs ~2x) × phase, and pick the
    ``(scale, phase)`` minimising a complexity-penalised snap cost: each onset pays
    its distance to the nearest eighth, or to the nearest sixteenth plus a penalty.
    The minimiser is the grid on which the onsets fall most simply.

    **Self-gating.** The rescale is only returned when it improves the *raw* snap
    fit (distance to the nearest eighth-or-sixteenth, **without** the complexity
    penalty) by at least ``min_raw_gain`` — i.e. only when there is a genuine
    spacing error to fix. On mixed-rhythm material that has no tempo error, the
    penalty alone would otherwise relabel real sixteenths as eighths; the raw-gain
    guard makes that a no-op, so true sixteenths are preserved.

    Returns ``(scale, phase_ticks)``; apply as ``t' = t * scale - phase``. Returns
    ``(1.0, 0.0)`` (no-op) for short input, when the identity grid is already best
    (e.g. exact MIDI-render onsets), or when no real spacing error is present.
    """
    if len(onset_ticks) < 6:
        return 1.0, 0.0

    eighth = float(EIGHTH_BASE)
    sixteenth = float(STRAIGHT_BASE)
    penalty = sixteenth_penalty * sixteenth

    def cost(scale: float, phase: float, *, pen: float) -> float:
        total = 0.0
        for t in onset_ticks:
            x = t * scale - phase
            e8 = abs(x - round(x / eighth) * eighth)
            e16 = abs(x - round(x / sixteenth) * sixteenth) + pen
            total += min(e8, e16)
        return total

    scales = [1.0 + max_tempo_dev * (2.0 * i / (n_tempo - 1) - 1.0) for i in range(n_tempo)]
    phases = [sixteenth * j / n_phase for j in range(n_phase)]

    best = (1.0, 0.0)
    best_cost = cost(1.0, 0.0, pen=penalty)
    for s in scales:
        for p in phases:
            c = cost(s, p, pen=penalty)
            if c < best_cost - 1e-9:
                best_cost, best = c, (s, p)
    if best == (1.0, 0.0):
        return 1.0, 0.0

    # Raw-fit guard: accept the rescale only if it fixes a real spacing error,
    # not merely a penalty-driven 16th->8th relabel. Compare un-penalised fit.
    raw_id = cost(1.0, 0.0, pen=0.0)
    raw_best = cost(best[0], best[1], pen=0.0)
    floor = 0.02 * sixteenth * len(onset_ticks)  # already-tight material: nothing to fix
    if raw_id <= floor or (raw_id - raw_best) < min_raw_gain * raw_id:
        return 1.0, 0.0
    return best


def residual_jitter_beats(onset_ticks: list[float], base: int = EIGHTH_BASE) -> float:
    """Median absolute deviation (in beats) of onsets from the nearest ``base`` step.

    A robust estimate of a take's timing noise — drives jitter-adaptive grid
    resolution (tight takes resolve sixteenths, loose takes stay on eighths).
    """
    if not onset_ticks:
        return 0.0
    devs = sorted(abs(t - round(t / base) * base) for t in onset_ticks)
    mad = devs[len(devs) // 2]
    return float(mad) / TICKS_PER_QUARTER


def offsets_reliable(
    onsets: list[float],
    offsets: list[float],
    *,
    min_median_ratio: float = 0.6,
) -> bool:
    """Decide whether detected note-offsets carry real duration information.

    Note values can be driven either by sounding duration (giving rests on early
    releases) or, when offsets are unreliable, purely by the inter-onset interval
    (legato). BasicPitch on real audio frequently reports a near-constant short
    sustain regardless of how long the note is actually held, which — if trusted —
    notates every note as a short value plus a spurious rest (the "staccato +
    rests" failure that the ``legato`` flag was a manual workaround for).

    We measure, per note, the ratio of its sounding duration to its IOI and take
    the median. When offsets are uniformly short noise the median ratio is low, so
    they should be ignored (legato). When notes genuinely sustain toward the next
    onset the median is near 1 and the offsets carry usable rest information. The
    median is robust to the mix of held and released notes in real playing.

    Returns ``True`` when offsets should be trusted (use rests), ``False`` when
    note values should follow the IOI (legato).
    """
    ratios: list[float] = []
    for i in range(len(onsets) - 1):
        ioi = onsets[i + 1] - onsets[i]
        if ioi > 1e-6:
            ratios.append((offsets[i] - onsets[i]) / ioi)
    if len(ratios) < 4:
        return True  # too little evidence to override offsets
    ratios.sort()
    median = ratios[len(ratios) // 2]
    return median >= min_median_ratio


def quantize_onset_ticks(
    onset_ticks: list[float],
    *,
    enable_triplets: bool,
    triplet_margin: float = 0.65,
    coarse_margin: float = 0.25,
    jitter_adaptive: bool = False,
    finest_base: int = STRAIGHT_BASE,
) -> list[int]:
    """Snap raw beat-relative tick positions to a metrical grid.

    Per beat we pick the **coarsest grid that fits**: prefer the eighth-note grid
    and only drop to sixteenths when that genuinely reduces the snap error by more
    than ``coarse_margin`` of a sixteenth per onset. This absorbs onset jitter and
    collapses sparse off-beat ghost notes onto the beat (clean eighth-note rhythm),
    while still notating real sixteenth passages. Triplets are chosen per beat when
    they beat the straight grid by ``triplet_margin`` (Cemgil & Desain criterion
    with a complexity penalty favouring simpler rhythm).

    ``jitter_adaptive`` (used by ``auto_rhythm``): when ``coarse_margin`` has been
    raised above 0.25 to suppress jitter noise, this flag adds a per-beat guard so
    real sixteenth-pattern onsets are not accidentally flattened. For each beat, if
    *any* onset is more than 35 % of an eighth note away from every eighth-grid
    position, that onset is genuinely off the eighth grid (not just jitter), so the
    base margin (0.25) is used for that beat instead of the elevated one.
    """
    if not onset_ticks:
        return []

    # Per-beat off-grid threshold: above this fraction of an eighth, an onset is
    # genuinely off the eighth grid (not just timing noise on an eighth position).
    # 0.28 × 480 = 134 ticks ≈ 87 ms.  A genuine sixteenth sits 240 ticks from the
    # nearest eighth, so up to 240-134=106 ticks of jitter before it stops triggering.
    _OFF_GRID_FRAC = 0.28
    # When the off-grid guard fires we use this lower effective margin so the cost
    # comparison confirms the sixteenth even with ≤50 ms onset jitter.
    # Derivation: for [0,480,720] with jitter j, diff=240-2j; need diff > M×240×3.
    # At M=0.10, j=80 ticks (52 ms): 80>72 ✓.  At M=0.25: needs j<30 ticks ✗.
    _JITTER_ADAPTIVE_MARGIN = 0.10
    # Beat-end guard: if the LATEST onset in a beat is within this fraction of the
    # beat duration from the next beat boundary, the onset is most likely a next-beat
    # note detected a bit early (guitar pluck latency, BasicPitch onset shift), not a
    # genuine sub-beat position.  In that case the off-grid activation is suppressed.
    # 0.15 × 960 = 144 ticks ≈ 94 ms.  Verified: test1/test6 false cases (135/82 ticks
    # from beat end) are suppressed; vidtest1 genuine case (300 ticks from beat end) is not.
    _BEAT_END_GUARD = 0.15

    # Group onset indices by beat.
    by_beat: dict[int, list[int]] = {}
    for i, t in enumerate(onset_ticks):
        by_beat.setdefault(int(t // TICKS_PER_QUARTER), []).append(i)

    snapped = [0] * len(onset_ticks)
    for beat_i, idxs in by_beat.items():
        beat0 = beat_i * TICKS_PER_QUARTER

        def cost(base: int) -> tuple[float, dict[int, int]]:
            total = 0.0
            placed: dict[int, int] = {}
            for j in idxs:
                rel = onset_ticks[j] - beat0
                step = round(rel / base)
                q = beat0 + step * base
                total += abs(onset_ticks[j] - q)
                placed[j] = int(q)
            return total, placed

        eighth_cost, eighth = cost(EIGHTH_BASE)
        sixteenth_cost, sixteenth = cost(finest_base)

        # Effective coarse margin for this beat.  When jitter-adaptive mode is on
        # and the global margin has been raised, use a lower margin only when a
        # genuine off-grid onset is present AND no onset is drifting in from the
        # next beat (beat-end guard).
        if jitter_adaptive and coarse_margin > 0.25:
            max_eighth_err = max(
                abs((onset_ticks[j] - beat0)
                    - round((onset_ticks[j] - beat0) / EIGHTH_BASE) * EIGHTH_BASE)
                for j in idxs
            )
            beat_end = (beat_i + 1) * TICKS_PER_QUARTER
            sorted_onsets = sorted(onset_ticks[j] for j in idxs)
            median_onset = sorted_onsets[len(sorted_onsets) // 2]
            near_beat_end = (beat_end - median_onset) < _BEAT_END_GUARD * TICKS_PER_QUARTER
            effective_margin = (
                _JITTER_ADAPTIVE_MARGIN
                if max_eighth_err > _OFF_GRID_FRAC * EIGHTH_BASE and not near_beat_end
                else coarse_margin
            )
        else:
            effective_margin = coarse_margin

        # Use the fine grid only when it cuts error meaningfully (real fast notes).
        threshold = effective_margin * finest_base * len(idxs)
        straight = sixteenth if (eighth_cost - sixteenth_cost) > threshold else eighth
        straight_cost = sixteenth_cost if straight is sixteenth else eighth_cost

        if enable_triplets:
            trip_cost, trip = cost(TRIPLET_BASE)
            chosen = trip if trip_cost < straight_cost * triplet_margin else straight
        else:
            chosen = straight
        for j, q in chosen.items():
            snapped[j] = q
    return snapped


def quantize_onsets_dominant_grid(
    onset_ticks: list[float],
    *,
    max_mae: float = 0.12,
    divisions: tuple[int, ...] = (1, 2, 4),
) -> list[int]:
    """Snap onsets to the *coarsest uniform grid that fits the whole phrase*.

    Human recordings of, say, steady quarter notes carry timing jitter that the
    per-beat snapper (:func:`quantize_onset_ticks`) notates as off-beat eighths and
    dotted/tied values. Here we instead pick a single global subdivision: the
    coarsest ``1/div`` of a beat whose *mean* snap error stays under ``max_mae``
    beats, then hard-snap every onset to it. Because the criterion is mean error
    (not worst-case), a handful of jittered outliers no longer force a finer grid —
    they are pulled onto the beat — so genuine quarter material reads as quarters,
    while real eighth/sixteenth passages (whose coarse-grid error is large) still
    select the finer grid. This mirrors how score-following editors lock a take to
    its dominant rhythmic resolution.
    """
    if not onset_ticks:
        return []
    q = TICKS_PER_QUARTER
    best_div = divisions[-1]
    for div in divisions:
        step = q / div
        total = sum(abs(t - round(t / step) * step) for t in onset_ticks)
        mae_beats = (total / len(onset_ticks)) / q
        if mae_beats <= max_mae:
            best_div = div
            break
    step = q / best_div
    return [int(round(t / step) * step) for t in onset_ticks]


@dataclass
class NoteValueBeat:
    """One resolved beat: a note (with notes) or a rest."""

    tick_start: int       # song ticks from the first onset
    duration_ticks: int
    is_rest: bool
    notes: list[ExportNote]


def build_note_value_schedule(
    notes: list[ExportNote],
    bpm: float,
    *,
    enable_triplets: bool = False,
    min_note_ticks: int | None = None,
    rest_threshold_ticks: int | None = None,
    trim_leading_silence: bool = True,
    tempo_map=None,
    lead_in_beats: int = 0,
    align_grid_phase: bool = True,
    fix_leading_outlier: bool = True,
    snap_dominant_grid: bool = False,
    dominant_grid_max_mae: float = 0.12,
    legato: bool = False,
    optimize_grid: bool = False,
    auto_rhythm: bool = False,
    base_division: int = 4,
) -> tuple[list[NoteValueBeat], int]:
    """Convert timed notes into a rest-aware beat schedule.

    ``base_division`` sets the finest onset grid: 4 = sixteenth (default, back-
    compatible), 8 = thirty-second, 16 = sixty-fourth. A finer grid preserves
    more distinct attacks from fast/close performance timing (fewer notes get
    merged into chords by quantization) at the cost of busier notation — use 8
    for real transcriptions where onset fidelity matters more than tidy rhythm.

    Returns ``(beats, base_unit)`` where ``base_unit`` is the finest grid step
    used (drives decomposition). Chords (notes sharing a quantized onset) are
    grouped into a single beat.

    ``auto_rhythm`` infers the notation strategy from the onsets instead of fixed
    flags, so the caller does not have to know per-clip whether the take is legato,
    has a tempo-estimate error, or mixes sixteenths. It (1) always uses the
    per-beat coarsest-grid quantizer — never the global dominant-grid flatten,
    which collapses mixed rhythms; (2) runs the **self-gating**
    :func:`optimize_grid_alignment`, which corrects a real tempo-spacing error but
    no-ops on mixed-rhythm material with no error; and (3) chooses legato vs rests
    from :func:`offsets_reliable`. It overrides ``snap_dominant_grid``/``legato``/
    ``optimize_grid`` when set.
    """
    if not notes or bpm <= 0:
        return [], STRAIGHT_BASE

    # auto_rhythm resolves the per-clip strategy from the data (see docstring).
    use_optimize = optimize_grid or auto_rhythm
    use_dominant = snap_dominant_grid and not auto_rhythm

    base_unit = TRIPLET_BASE if enable_triplets else (TICKS_PER_QUARTER // max(base_division, 1))
    # Thresholds must be multiples of the grid or lengths drift off-grid.
    min_note = base_unit if min_note_ticks is None else max(min_note_ticks, base_unit)
    rest_thresh = base_unit if rest_threshold_ticks is None else rest_threshold_ticks

    ordered = sorted(notes, key=lambda n: float(n["start"]))
    t0 = float(ordered[0]["start"]) if trim_leading_silence else 0.0

    if tempo_map is not None:
        # Quantise in beat space: correct across mid-piece tempo changes.
        beat0 = tempo_map.seconds_to_beats(t0)

        def _to_ticks(t: float) -> float:
            return (tempo_map.seconds_to_beats(float(t)) - beat0) * TICKS_PER_QUARTER

        raw_onsets = [_to_ticks(n["start"]) for n in ordered]
        raw_offsets = [_to_ticks(n["end"]) for n in ordered]
    else:
        sec_to_tick = bpm / 60.0 * TICKS_PER_QUARTER
        raw_onsets = [(float(n["start"]) - t0) * sec_to_tick for n in ordered]
        raw_offsets = [(float(n["end"]) - t0) * sec_to_tick for n in ordered]

    # Leading-onset outlier fix: a single early/ringing detection before the real
    # first downbeat otherwise becomes an over-long first note that breaks the rhythm.
    if fix_leading_outlier:
        raw_onsets, raw_offsets = correct_leading_onset_gap(raw_onsets, raw_offsets)

    # Auto legato decision: trust detected offsets (insert rests) only when their
    # sounding durations actually carry information; otherwise drive note values
    # from the IOI (legato). Resolved from the data so the caller need not guess.
    if auto_rhythm:
        legato = not offsets_reliable(raw_onsets, raw_offsets)

    # Joint tempo-spacing + phase alignment. A few-% error in the estimated BPM
    # leaves the onset grid mis-*spaced* (not just offset): steady eighths then
    # drift across the metrical grid and get notated as 16ths + false syncopation.
    # Phase de-bias alone cannot fix spacing. ``optimize_grid`` searches a small
    # tempo band (±6%, too narrow to turn a real 16th into an 8th) × phase and
    # rescales the whole tick stream, so it subsumes the phase de-bias below.
    coarse_margin = 0.25
    if use_optimize:
        if auto_rhythm:
            # Strategy selection: a large systematic grid-phase offset signals a
            # MIDI-like take (BasicPitch consistently detects onsets a fixed amount
            # early relative to the tempo grid).  In that case, the classical
            # grid-phase correction restores clean snapping without inflating the
            # coarse margin.  For audio-like takes (small or zero systematic phase),
            # timing noise is random rather than systematic, so we instead apply the
            # tempo-spacing optimiser + jitter-adaptive margin.
            #
            # Threshold: half an eighth note (240 ticks) at the slowest practical
            # tempo corresponds to ~60 ticks at 96 BPM / ~25 ticks at 150 BPM.
            # Empirically: MIDI render clips have |phi| ≈ 94–140 ticks; real-audio
            # clips have |phi| ≈ 0.
            _PHASE_AUDIO_THRESH = 60.0  # ticks; above this → MIDI-like path
            phi = estimate_grid_phase(raw_onsets, EIGHTH_BASE)
            if phi and abs(phi) > _PHASE_AUDIO_THRESH:
                # MIDI-like: apply phase correction and keep base coarse_margin.
                # jitter_adaptive guard is inactive when coarse_margin == 0.25.
                raw_onsets = [t - phi for t in raw_onsets]
                raw_offsets = [t - phi for t in raw_offsets]
            else:
                # Audio-like: tempo-spacing optimiser + elevated coarse margin.
                scale, opp = optimize_grid_alignment(raw_onsets)
                if scale != 1.0 or opp != 0.0:
                    raw_onsets = [t * scale - opp for t in raw_onsets]
                    raw_offsets = [t * scale - opp for t in raw_offsets]
                # Jitter-adaptive resolution: a looser take carries more timing
                # noise, so the per-beat snapper needs a larger margin before it
                # drops to sixteenths (else noise alone manufactures 16th notes).
                jitter = residual_jitter_beats(raw_onsets, base=STRAIGHT_BASE)
                coarse_margin = 0.25 + min(jitter / 0.125, 0.6)
        else:
            scale, opp = optimize_grid_alignment(raw_onsets)
            if scale != 1.0 or opp != 0.0:
                raw_onsets = [t * scale - opp for t in raw_onsets]
                raw_offsets = [t * scale - opp for t in raw_offsets]
            jitter = residual_jitter_beats(raw_onsets, base=STRAIGHT_BASE)
            coarse_margin = 0.25 + min(jitter / 0.125, 0.6)

    # Grid-phase de-bias: remove a constant sub-beat offset between the detected
    # onsets and the metrical grid so steady notes snap cleanly instead of being
    # notated as syncopated dotted/tied values. Shifts onsets and offsets equally
    # so IOIs and sounding durations are preserved. Skipped when ``optimize_grid``
    # already aligned phase jointly with spacing.
    elif align_grid_phase:
        phi = estimate_grid_phase(raw_onsets, EIGHTH_BASE)
        if phi:
            raw_onsets = [t - phi for t in raw_onsets]
            raw_offsets = [t - phi for t in raw_offsets]

    # Downbeat anchoring: shift everything by a pickup so the first real downbeat
    # lands on a barline (leading time becomes rests).
    if lead_in_beats:
        offset = lead_in_beats * TICKS_PER_QUARTER
        raw_onsets = [t + offset for t in raw_onsets]
        raw_offsets = [t + offset for t in raw_offsets]
    if use_dominant:
        snapped = quantize_onsets_dominant_grid(raw_onsets, max_mae=dominant_grid_max_mae)
    else:
        snapped = quantize_onset_ticks(
            raw_onsets, enable_triplets=enable_triplets, coarse_margin=coarse_margin,
            jitter_adaptive=auto_rhythm,
            finest_base=TICKS_PER_QUARTER // max(base_division, 1),
        )

    # Group notes by identical snapped onset (chords / near-simultaneous).
    groups: dict[int, list[int]] = {}
    for i, tick in enumerate(snapped):
        groups.setdefault(tick, []).append(i)
    onset_positions = sorted(groups)

    beats: list[NoteValueBeat] = []
    for k, onset in enumerate(onset_positions):
        members = groups[onset]
        # IOI to the next distinct onset (or this group's longest note at end).
        if k + 1 < len(onset_positions):
            ioi = onset_positions[k + 1] - onset
        else:
            longest = max(raw_offsets[i] for i in members) - onset
            ioi = max(round(longest / base_unit) * base_unit, base_unit)
        ioi = max(ioi, base_unit)

        # Sounding duration = longest member, quantized to the grid.
        sounding = max(raw_offsets[i] for i in members) - onset
        sounding_q = max(round(sounding / base_unit) * base_unit, base_unit)

        # Note value is at most the IOI; a clear early release leaves a rest.
        # Legato (live recordings): ignore unreliable audio offsets and fill the
        # whole IOI so steady playing reads as connected notes, not staccato + rests.
        if not legato and sounding_q + rest_thresh <= ioi:
            note_len = max(sounding_q, min_note)
            rest_len = ioi - note_len
        else:
            note_len = ioi
            rest_len = 0

        beats.append(
            NoteValueBeat(
                tick_start=onset,
                duration_ticks=note_len,
                is_rest=False,
                notes=[ordered[i] for i in members],
            )
        )
        if rest_len > 0:
            beats.append(
                NoteValueBeat(
                    tick_start=onset + note_len,
                    duration_ticks=rest_len,
                    is_rest=True,
                    notes=[],
                )
            )

    return beats, base_unit


# --- Song assembly (beat-boundary aware ties) ----------------------------

from guitarpro.models import Song  # noqa: E402

from .gp5_layout import (  # noqa: E402
    Gp5ExportOptions,
    _TrackCursor,
    _time_signature,
    apply_guitar_setup,
    dedupe_chord_notes,
    validate_gp5_song,
)


def _append_beat(voice: Voice, duration: Duration, notes, *, tie: bool) -> None:
    if not notes:
        voice.beats.append(Beat(voice=voice, duration=duration, status=BeatStatus.rest))
        return
    chord = dedupe_chord_notes(notes)
    beat = Beat(voice=voice, duration=duration, status=BeatStatus.normal)
    ntype = NoteType.tie if tie else NoteType.normal
    for n in chord:
        gp_string = aitabs_string_to_gp(n["string"])
        if not 1 <= gp_string <= 6:
            raise ValueError(f"Invalid string index {n['string']}")
        beat.notes.append(
            Note(beat=beat, string=gp_string, value=int(n["fret"]), velocity=80, type=ntype)
        )
    voice.beats.append(beat)


def _emit_span(
    song: Song,
    track,
    cursor: _TrackCursor,
    measure_length: int,
    length: int,
    notes,
    base_unit: int,
    *,
    is_rest: bool,
) -> None:
    """Emit a note/rest of ``length`` ticks, splitting at measure *and* beat
    boundaries and tying note continuations."""
    remaining = length
    started = False
    while remaining > 0:
        space = measure_length - cursor.tick
        if space <= 0:
            _roll(song, track, cursor, measure_length, base_unit)
            continue

        beat_pos = cursor.tick % TICKS_PER_QUARTER
        to_next_beat = TICKS_PER_QUARTER - beat_pos if beat_pos else TICKS_PER_QUARTER

        # Take the whole remaining span if it lands on a beat, fits the measure,
        # and is a single clean value (lets half / dotted-quarter span beats).
        if (
            beat_pos == 0
            and remaining <= space
            and _make_single(remaining) is not None
        ):
            chunk = remaining
        else:
            chunk = min(remaining, space, to_next_beat)

        voice = cursor.voice(track)
        parts = notation_for_length(chunk, base_unit)
        for i, dur in enumerate(parts):
            tie = (not is_rest) and (started or i > 0)
            _append_beat(voice, dur, [] if is_rest else notes, tie=tie)
        started = True
        cursor.tick += chunk
        remaining -= chunk
        if cursor.tick >= measure_length:
            _roll(song, track, cursor, measure_length, base_unit)


def _pad(cursor: _TrackCursor, track, measure_length: int, base_unit: int) -> None:
    gap = measure_length - cursor.tick
    if gap <= 0:
        return
    voice = cursor.voice(track)
    for dur in notation_for_length(gap, base_unit):
        voice.beats.append(Beat(voice=voice, duration=dur, status=BeatStatus.rest))
    cursor.tick = measure_length


def _roll(song: Song, track, cursor: _TrackCursor, measure_length: int,
          base_unit: int = STRAIGHT_BASE) -> None:
    _pad(cursor, track, measure_length, base_unit)
    cursor.measure_index += 1
    song.newMeasure()
    song.measureHeaders[-1].timeSignature = song.measureHeaders[0].timeSignature
    cursor.tick = 0


def build_song_from_note_values(
    notes: list[ExportNote],
    *,
    title: str = "AITabs Transcription",
    options: Gp5ExportOptions | None = None,
    enable_triplets: bool = False,
) -> Song:
    """Build a GP5 Song using IOI-derived note values and rests."""
    opts = options or Gp5ExportOptions()
    song = Song()
    song.title = title
    song.tempo = opts.tempo

    track = song.tracks[0]
    track.name = "Guitar"
    apply_guitar_setup(track, options=opts)

    num, denom = opts.time_signature
    song.measureHeaders[0].timeSignature = _time_signature(num, denom)
    measure_length = song.measureHeaders[0].length

    beats, base_unit = build_note_value_schedule(
        notes,
        bpm=float(opts.tempo),
        enable_triplets=enable_triplets,
        trim_leading_silence=opts.trim_leading_silence,
        tempo_map=opts.tempo_map,
        lead_in_beats=opts.lead_in_beats,
        snap_dominant_grid=opts.snap_dominant_grid,
        dominant_grid_max_mae=opts.dominant_grid_max_mae,
        legato=opts.legato,
        optimize_grid=opts.optimize_grid,
        auto_rhythm=opts.auto_rhythm,
        base_division=getattr(opts, "base_division", 4),
    )
    if not beats:
        voice = track.measures[0].voices[0]
        voice.beats.append(Beat(voice=voice, duration=Duration(value=Duration.quarter), status=BeatStatus.rest))
        return song

    cursor = _TrackCursor()
    timeline = 0
    for beat in beats:
        gap = beat.tick_start - timeline
        if gap > 0:  # uncovered time before this onset -> rest
            _emit_span(song, track, cursor, measure_length, gap, [], base_unit, is_rest=True)
            timeline += gap
        _emit_span(
            song, track, cursor, measure_length,
            beat.duration_ticks, beat.notes, base_unit, is_rest=beat.is_rest,
        )
        timeline += beat.duration_ticks

    _pad(cursor, track, measure_length, base_unit)
    validate_gp5_song(song)
    return song
