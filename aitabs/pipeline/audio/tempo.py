"""Tempo / beat grid estimation for guitar note cleanup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pitch import NoteEvent, _as_note_event


@dataclass(frozen=True)
class TempoMap:
    """Piecewise-constant tempo as ascending ``(start_beat, start_sec, bpm)`` anchors.

    A *beat* here is one quarter note. Lets onsets be mapped to/from beat positions
    correctly across mid-piece tempo changes — unlike a single constant BPM, which
    drifts every onset after a change off the metrical grid.
    """

    anchors: tuple[tuple[float, float, float], ...]

    @classmethod
    def constant(cls, bpm: float) -> "TempoMap":
        return cls(anchors=((0.0, 0.0, float(max(bpm, 1e-6))),))

    @classmethod
    def from_beat_times(cls, beat_times) -> "TempoMap":
        """Build a variable tempo map from tracked beat times (seconds, ascending).

        Each consecutive beat pair defines one quarter-note's local tempo, so a
        ritardando/accelerando is captured per beat. Falls back to 120 BPM if
        fewer than two beats are given.
        """
        bt = [float(t) for t in (beat_times if beat_times is not None else [])]
        if len(bt) < 2:
            return cls.constant(120.0)
        anchors: list[tuple[float, float, float]] = []
        for i in range(len(bt) - 1):
            dt = bt[i + 1] - bt[i]
            bpm = 60.0 / dt if dt > 1e-6 else 120.0
            anchors.append((float(i), bt[i], float(bpm)))
        # extend the final beat with the last tempo
        anchors.append((float(len(bt) - 1), bt[-1], anchors[-1][2]))
        return cls(anchors=tuple(anchors))

    def _seg_for_time(self, t: float) -> tuple[float, float, float]:
        seg = self.anchors[0]
        for a in self.anchors:
            if a[1] <= t + 1e-9:
                seg = a
            else:
                break
        return seg

    def _seg_for_beat(self, b: float) -> tuple[float, float, float]:
        seg = self.anchors[0]
        for a in self.anchors:
            if a[0] <= b + 1e-9:
                seg = a
            else:
                break
        return seg

    def seconds_to_beats(self, t: float) -> float:
        beat0, sec0, bpm = self._seg_for_time(float(t))
        return beat0 + (float(t) - sec0) * (bpm / 60.0)

    def beats_to_seconds(self, b: float) -> float:
        beat0, sec0, bpm = self._seg_for_beat(float(b))
        return sec0 + (float(b) - beat0) * (60.0 / bpm)


@dataclass(frozen=True)
class TempoAnalysis:
    """Estimated tempo and beat grid for a clip."""

    bpm: float
    beat_times: np.ndarray  # seconds, ascending
    beat_period: float  # seconds per quarter note
    origin: float  # time of first downbeat used for the grid
    ticks_per_beat: int = 4  # 4 = 16th notes, 2 = 8ths
    confidence: float = 1.0  # 0–1 heuristic from beat stability
    # Beat fractions in [0, 1) where attacks cluster (audio-inferred). None = uniform grid.
    phase_template: tuple[float, ...] | None = None

    @property
    def tick_duration(self) -> float:
        if self.phase_template:
            return self.beat_period / max(len(self.phase_template), 1)
        return self.beat_period / self.ticks_per_beat

    @property
    def uses_adaptive_template(self) -> bool:
        return bool(self.phase_template)


def estimate_tempo(
    audio_path: str | None = None,
    y: np.ndarray | None = None,
    sr: int = 22050,
    hop_length: int = 512,
    ticks_per_beat: int = 4,
) -> TempoAnalysis:
    """Estimate BPM and beat times using librosa (onset + beat tracking).

    Args:
        audio_path: WAV/MP3 path (used if ``y`` is not provided).
        y: Mono audio samples.
        sr: Sample rate.
        hop_length: STFT hop for onset/beat features.
        ticks_per_beat: Grid resolution (4 → 16th notes).

    Returns:
        TempoAnalysis with bpm, beat_times, and grid origin.
    """
    import librosa

    if y is None:
        if audio_path is None:
            raise ValueError("Provide audio_path or y")
        y, sr = librosa.load(audio_path, sr=sr, mono=True)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # librosa >= 0.10: librosa.feature.tempo (0.11+ drops feature.rhythm.tempo path)
    tempo_fn = getattr(librosa.feature, "tempo", None)
    if tempo_fn is None:
        tempo_fn = librosa.feature.rhythm.tempo  # type: ignore[attr-defined]

    tempo_prior = tempo_fn(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,
    )
    bpm_prior = float(np.atleast_1d(tempo_prior)[0])

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        bpm=bpm_prior,
        units="frames",
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    beat_times = np.asarray(beat_times, dtype=float)

    bpm = float(np.atleast_1d(tempo)[0])
    if len(beat_times) >= 2:
        ioi = np.diff(beat_times)
        bpm_from_beats = 60.0 / float(np.median(ioi))
        # Prefer beat-interval BPM when close to prior (reduces octave errors)
        if 0.5 * bpm_prior <= bpm_from_beats <= 2.0 * bpm_prior:
            bpm = bpm_from_beats
        beat_period = 60.0 / bpm
        stability = 1.0 - min(float(np.std(ioi) / (np.median(ioi) + 1e-6)), 1.0)
        confidence = float(np.clip(stability, 0.2, 1.0))
    else:
        beat_period = 60.0 / bpm
        confidence = 0.3

    origin = float(beat_times[0]) if len(beat_times) else 0.0

    return TempoAnalysis(
        bpm=bpm,
        beat_times=beat_times,
        beat_period=beat_period,
        origin=origin,
        ticks_per_beat=ticks_per_beat,
        confidence=confidence,
    )


def refine_beats_to_onsets(
    beat_times,
    onsets,
    *,
    window_frac: float = 0.35,
) -> np.ndarray:
    """Snap each tracked beat onto the nearest note onset (beat-phase refinement).

    librosa beat tracking nails the *number* of beats but its phase can jitter by
    a sizeable fraction of a beat on sparse solo-guitar material. Left in, that
    jitter accumulates so steady playing quantizes onto off-beats and a stream of
    quarter notes is notated as dotted/tied eighths (the classic "live recording
    reads badly" failure). Note onsets are the strongest beat evidence here, so we
    nudge each beat to the closest onset within ``window_frac`` of the local beat
    period. This preserves librosa's beat structure — beats with no nearby onset
    (rests / held notes) keep their tracked time, so multi-beat sustains are not
    compressed — while removing the systematic phase error. Each onset anchors at
    most one beat, and the result is kept strictly ascending.
    """
    bt = np.asarray(beat_times, dtype=float)
    ons = np.asarray(sorted(float(o) for o in onsets), dtype=float)
    if bt.size < 2 or ons.size == 0:
        return bt
    period = float(np.median(np.diff(bt)))
    if period <= 0:
        return bt
    win = period * window_frac
    used = np.zeros(ons.size, dtype=bool)
    refined = bt.astype(float).copy()
    for i, b in enumerate(bt):
        cand = np.where((np.abs(ons - b) <= win) & ~used)[0]
        if cand.size:
            j = cand[int(np.argmin(np.abs(ons[cand] - b)))]
            refined[i] = ons[j]
            used[j] = True
    # Snapping can reorder adjacent beats; keep the grid strictly ascending,
    # reverting offenders toward their original tracked position.
    for i in range(1, refined.size):
        if refined[i] <= refined[i - 1]:
            refined[i] = max(bt[i], refined[i - 1] + 1e-3)
    return refined


@dataclass(frozen=True)
class MeterEstimate:
    """Estimated meter and where the first beat sits inside the bar."""

    numerator: int
    denominator: int
    downbeat_beat: int   # beat-in-bar index (0-based) of beat_times[0]
    lead_in_beats: int   # beats of pickup before the first full bar
    confidence: float

    @property
    def time_signature(self) -> tuple[int, int]:
        return (self.numerator, self.denominator)


def _beat_accents(onset_times, beat_times, onset_strengths=None) -> np.ndarray:
    """Per-beat accent = summed onset strength falling in each beat interval."""
    beats = np.asarray(beat_times, dtype=float)
    onsets = np.asarray(onset_times, dtype=float)
    if onset_strengths is None:
        strengths = np.ones_like(onsets)
    else:
        strengths = np.asarray(onset_strengths, dtype=float)

    accents = np.zeros(max(len(beats) - 1, 0), dtype=float)
    if accents.size == 0 or onsets.size == 0:
        return accents
    idx = np.searchsorted(beats, onsets, side="right") - 1
    for k, b in enumerate(idx):
        if 0 <= b < accents.size:
            accents[b] += strengths[k]
    return accents


# Below this accent confidence, the downbeat phase is treated as unreliable and
# the origin anchor (piece starts on a downbeat) is used instead. Real solo-guitar
# clips land at ~0.1–0.25; clean synthetic/accented meters at ~1.0.
_ACCENT_PHASE_MIN_CONF = 0.5


def _phase_from_origin(beat_times, numerator: int) -> int | None:
    """Bar-position of ``beat_times[0]`` assuming the piece starts on a downbeat.

    Beat tracking routinely drops the first beat or two, so ``beat_times[0]`` is
    rarely the true downbeat. But most recordings *start* on a downbeat (beat 1 at
    t≈0), so we extrapolate the grid back to the origin: ``beat_times[0]`` sits
    ``round(beat_times[0] / period)`` whole beats after t=0, and its bar-position
    is that count mod the bar length. This is a far stronger downbeat cue than
    onset accent on solo/fingerstyle guitar, where the downbeat carries no extra
    energy. Returns ``None`` when the grid is too short/irregular to trust.
    """
    bt = np.asarray(beat_times, dtype=float)
    if bt.size < 2 or numerator <= 0:
        return None
    period = float(np.median(np.diff(bt)))
    if period <= 0:
        return None
    n0 = int(round(bt[0] / period))
    return n0 % numerator


def estimate_meter(
    onset_times,
    beat_times,
    *,
    onset_strengths=None,
    candidates: tuple[int, ...] = (2, 3, 4),
    denominator: int = 4,
    origin_downbeat: bool = True,
) -> MeterEstimate:
    """Estimate meter (bar length) and downbeat phase.

    Bar length (numerator) comes from the accent-autocorrelation cue: fold the
    per-beat accent signal at each candidate bar length and score by periodicity
    plus downbeat contrast (license-clean, numpy only; no external models).

    Downbeat *phase* is taken from the **origin anchor** by default
    (``origin_downbeat=True``): assume the excerpt starts on a downbeat and
    extrapolate the beat grid to t≈0 (see :func:`_phase_from_origin`). On solo
    guitar the accent cue cannot locate the downbeat (no energy accent on beat 1),
    so the accent-derived phase is used only as a fallback when the origin anchor
    is unavailable. Validated on GuitarSet: origin phase 19/20 vs accent ~0/20.

    Returns a :class:`MeterEstimate`; defaults to 4/4 with low confidence when the
    signal is too short to decide.
    """
    accents = _beat_accents(onset_times, beat_times, onset_strengths)
    n = accents.size
    if n < 4:
        return MeterEstimate(4, denominator, 0, 0, 0.0)

    a = accents - accents.mean()
    denom = float(np.sum(a * a)) or 1.0

    best = None  # (score, numerator, phase)
    # Mild prior so 4/4 wins ties and 2 doesn't masquerade for 4.
    prior = {2: 0.9, 3: 1.0, 4: 1.05}
    for N in candidates:
        if n < 2 * N:
            continue
        # bar-length periodicity via normalized autocorrelation at lag N
        ac = float(np.sum(a[:-N] * a[N:])) / denom
        # downbeat contrast: best phase where every N-th beat is strongest
        phase_means = [accents[p::N].mean() for p in range(N)]
        best_phase = int(np.argmax(phase_means))
        contrast = (max(phase_means) - np.mean(phase_means)) / (accents.mean() + 1e-9)
        score = (contrast * (1.0 + max(ac, 0.0))) * prior.get(N, 1.0)
        if best is None or score > best[0]:
            best = (score, N, best_phase)

    if best is None:
        return MeterEstimate(4, denominator, 0, 0, 0.0)

    score, numerator, phase = best
    confidence = float(np.clip(score, 0.0, 1.0))
    # Trust the accent-derived phase only when the downbeat accent is clear.
    # On solo/fingerstyle guitar the accent contrast is weak (conf ~0.1–0.25) and
    # the phase is essentially noise, so fall back to the origin anchor (the piece
    # starts on a downbeat). A confident accent (e.g. a real pickup that syncs the
    # bar to a strong 2nd beat) still wins.
    if origin_downbeat and confidence < _ACCENT_PHASE_MIN_CONF:
        origin_phase = _phase_from_origin(beat_times, numerator)
        if origin_phase is not None:
            phase = origin_phase
    lead_in = (numerator - phase) % numerator
    return MeterEstimate(numerator, denominator, phase, lead_in, confidence)


def _phase_distance(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 1.0 - d)


def time_to_template_slot(t: float, analysis: TempoAnalysis) -> tuple[int, int]:
    """Map time to (beat index, slot index) on an adaptive phase template."""
    template = analysis.phase_template
    if not template:
        tick = onset_to_tick(t, analysis)
        tpb = max(analysis.ticks_per_beat, 1)
        return tick // tpb, tick % tpb

    beat_period = analysis.beat_period
    if beat_period <= 0:
        return 0, 0

    rel = float(t) - analysis.origin
    beat_i = int(np.floor(rel / beat_period))
    if rel < 0:
        beat_i -= 1
    phase = (rel - beat_i * beat_period) / beat_period
    phase = phase % 1.0
    slot_i = min(range(len(template)), key=lambda i: _phase_distance(phase, template[i]))
    return beat_i, slot_i


def template_slot_to_time(beat_i: int, slot_i: int, analysis: TempoAnalysis) -> float:
    template = analysis.phase_template
    if not template:
        tpb = max(analysis.ticks_per_beat, 1)
        tick = beat_i * tpb + slot_i
        return analysis.origin + tick * analysis.tick_duration

    beat_period = analysis.beat_period
    phase = template[slot_i % len(template)]
    return analysis.origin + beat_i * beat_period + phase * beat_period


def onset_to_tick(t: float, analysis: TempoAnalysis) -> int:
    """Map a time (seconds) to an integer grid tick (uniform or adaptive template)."""
    if analysis.phase_template:
        beat_i, slot_i = time_to_template_slot(t, analysis)
        return beat_i * len(analysis.phase_template) + slot_i

    tick_dur = analysis.tick_duration
    if tick_dur <= 0:
        return 0
    return int(round((t - analysis.origin) / tick_dur))


def tick_to_time(tick: int, analysis: TempoAnalysis) -> float:
    if analysis.phase_template:
        n = len(analysis.phase_template)
        return template_slot_to_time(tick // n, tick % n, analysis)
    return analysis.origin + tick * analysis.tick_duration


def snap_time_to_grid(t: float, analysis: TempoAnalysis) -> float:
    """Round a time (seconds) to the nearest grid or adaptive template step."""
    if analysis.phase_template:
        beat_i, slot_i = time_to_template_slot(t, analysis)
        return template_slot_to_time(beat_i, slot_i, analysis)
    return tick_to_time(onset_to_tick(t, analysis), analysis)


def classify_same_pitch_pair(
    t1: float,
    t2: float,
    analysis: TempoAnalysis,
    pitch_merge_sec: float = 0.15,
) -> str:
    """Classify two onsets of the same pitch as duplicate chatter vs intentional repeat."""
    dt = abs(t2 - t1)
    if dt < pitch_merge_sec:
        return "duplicate"

    tick1 = onset_to_tick(t1, analysis)
    tick2 = onset_to_tick(t2, analysis)
    tick_gap = abs(tick2 - tick1)

    # Same grid tick, or neighbors within ~1.5 sixteenths → likely one intended onset
    if tick_gap == 0:
        return "duplicate"
    if tick_gap == 1 and dt < analysis.tick_duration * 1.75:
        return "duplicate"

    return "intentional"


def suppress_repeat_ghosts(
    notes: list[NoteEvent] | list[tuple],
    min_ioi_sec: float = 0.12,
) -> list[NoteEvent]:
    """Drop double-triggered onsets: a same-pitch note firing within ``min_ioi_sec``
    of the previous same-pitch note (keeps the higher-confidence of the pair).

    Simpler and more reliable than the tempo-grid dedup for removing detector
    over-segmentation on sustained/repeated notes (e.g. an open-string pedal),
    without dropping genuine repeats spaced beyond ``min_ioi_sec``. On vidtest1
    this cut ghosts ~12->7 with no added misses.
    """
    if min_ioi_sec <= 0:
        return [_as_note_event(n) for n in notes]
    normalized = sorted((_as_note_event(n) for n in notes), key=lambda n: n["start"])
    kept: list[NoteEvent] = []
    last_idx: dict[int, int] = {}   # pitch -> index in kept
    for note in normalized:
        p = note["pitch_midi"]
        if p in last_idx and note["start"] - kept[last_idx[p]]["start"] < min_ioi_sec:
            j = last_idx[p]
            if note["confidence"] > kept[j]["confidence"]:
                kept[j] = note          # keep the stronger of the double-trigger
            continue
        kept.append(note)
        last_idx[p] = len(kept) - 1
    return kept


def clean_notes_with_tempo(
    notes: list[NoteEvent] | list[tuple],
    analysis: TempoAnalysis,
    pitch_merge_sec: float = 0.15,
) -> list[NoteEvent]:
    """Deduplicate notes using time window + tempo grid (keeps intentional repeats)."""
    normalized = [_as_note_event(n) for n in notes]
    normalized.sort(key=lambda n: (n["start"], -n["confidence"]))

    kept: list[NoteEvent] = []
    for note in normalized:
        merged = False
        for i, existing in enumerate(kept):
            if note["pitch_midi"] != existing["pitch_midi"]:
                continue
            if (
                classify_same_pitch_pair(
                    existing["start"],
                    note["start"],
                    analysis,
                    pitch_merge_sec=pitch_merge_sec,
                )
                == "duplicate"
            ):
                if note["confidence"] > existing["confidence"]:
                    kept[i] = note
                merged = True
                break
        if not merged:
            kept.append(note)

    return sorted(kept, key=lambda n: n["start"])


def annotate_notes_for_display(
    notes: list[NoteEvent],
    analysis: TempoAnalysis,
) -> list[dict]:
    """Add beat/tick labels for notebook tables and debugging."""
    out: list[dict] = []
    for n in notes:
        tick = onset_to_tick(n["start"], analysis)
        beat_idx = tick // analysis.ticks_per_beat
        sub_idx = tick % analysis.ticks_per_beat
        out.append(
            {
                **n,
                "tick": tick,
                "beat": beat_idx + 1,
                "subdivision": sub_idx + 1,
                "grid_time": tick_to_time(tick, analysis),
            }
        )
    return out
