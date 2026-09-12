"""Rhythm- and notation-level accuracy metrics for GP5 export.

The note-level metrics in :mod:`metrics` score *what* was played (pitch,
string, fret) and onset timing. These metrics score *how the rhythm was
notated* — the thing a player actually reads off the tab:

    metric_position_f1   — do predicted attacks land on the same score-time
                           grid positions as the reference? (pitch-agnostic
                           onset alignment; the beat/downbeat F-score used in
                           the audio-to-score literature)
    note_value_accuracy  — of the matched attacks, what fraction got the same
                           notated *duration* (tied continuations folded in,
                           so dotted-quarter == quarter-tied-to-eighth)?
    rest_f1              — are rests notated in the same places? (trailing
                           end-of-piece bar padding is ignored)

Everything is computed in *score time* (ticks at 960/quarter) so it is tempo
independent: a clip transcribed at the right relative rhythm scores well even
if the absolute BPM estimate is off.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import guitarpro
from guitarpro.models import BeatStatus, Duration, NoteType

_TPQ = Duration.quarterTime


@dataclass(frozen=True)
class RhythmNote:
    tick: int        # score ticks from first attack
    duration: int    # ticks, folding tied continuations


@dataclass(frozen=True)
class RhythmRest:
    tick: int
    duration: int


@dataclass(frozen=True)
class BeatScores:
    """Beat-tracking accuracy of the pipeline against ground-truth beats.

    ``beat_f`` / ``downbeat_f`` are the standard mir_eval beat F-measure (a
    predicted beat counts as correct if within ``tolerance`` seconds of a
    reference beat, one-to-one). ``tempo_pct_err`` is the relative BPM error
    after octave-folding (a 2x/0.5x tempo is scored against its nearest octave,
    since beat *phase* is what quantisation needs). ``meter_correct`` compares
    the estimated bar length (numerator) to the annotation.
    """

    clip_id: str
    beat_f: float
    downbeat_f: float
    tempo_bpm_ref: float
    tempo_bpm_pred: float
    tempo_pct_err: float
    meter_correct: bool
    n_ref_beats: int
    n_pred_beats: int

    def summary_line(self) -> str:
        meter = "ok" if self.meter_correct else "MISS"
        return (
            f"{self.clip_id}: beat_F={self.beat_f:.3f} downbeat_F={self.downbeat_f:.3f} "
            f"tempo {self.tempo_bpm_pred:.1f}/{self.tempo_bpm_ref:.1f} "
            f"(err {self.tempo_pct_err*100:.1f}%) meter={meter} "
            f"beats {self.n_pred_beats}/{self.n_ref_beats}"
        )


def _octave_folded_pct_err(pred_bpm: float, ref_bpm: float) -> float:
    """Relative BPM error, folding 2x/0.5x octave confusions onto the reference.

    Beat tracking commonly locks to double or half tempo; for note quantisation
    the *grid phase* matters, not which octave, so we score against the nearest
    of {0.5x, 1x, 2x} pred.
    """
    if ref_bpm <= 0 or pred_bpm <= 0:
        return 0.0 if pred_bpm == ref_bpm else 1.0
    candidates = (pred_bpm, pred_bpm * 2.0, pred_bpm * 0.5)
    return min(abs(c - ref_bpm) / ref_bpm for c in candidates)


def score_beats(
    reference_beats,
    predicted_beats,
    *,
    reference_downbeats=None,
    predicted_downbeats=None,
    reference_bpm: float = 0.0,
    predicted_bpm: float = 0.0,
    reference_numerator: int = 0,
    predicted_numerator: int = 0,
    clip_id: str = "clip",
    tolerance: float = 0.07,
) -> BeatScores:
    """Score predicted beats/downbeats against ground truth via mir_eval.

    Pure logic (no I/O): callers load the audio-estimated grid and the
    annotation, then hand both here. ``tolerance`` is the mir_eval F-measure
    window in seconds (0.07 is the MIREX default).
    """
    import numpy as np
    import mir_eval.beat as beat_metrics

    ref = np.asarray(sorted(float(x) for x in reference_beats), dtype=float)
    pred = np.asarray(sorted(float(x) for x in predicted_beats), dtype=float)
    beat_f = float(beat_metrics.f_measure(ref, pred, f_measure_threshold=tolerance))

    if reference_downbeats is not None and predicted_downbeats is not None:
        ref_db = np.asarray(sorted(float(x) for x in reference_downbeats), dtype=float)
        pred_db = np.asarray(sorted(float(x) for x in predicted_downbeats), dtype=float)
        downbeat_f = float(
            beat_metrics.f_measure(ref_db, pred_db, f_measure_threshold=tolerance)
        )
    else:
        downbeat_f = float("nan")

    return BeatScores(
        clip_id=clip_id,
        beat_f=beat_f,
        downbeat_f=downbeat_f,
        tempo_bpm_ref=float(reference_bpm),
        tempo_bpm_pred=float(predicted_bpm),
        tempo_pct_err=_octave_folded_pct_err(predicted_bpm, reference_bpm),
        meter_correct=bool(predicted_numerator == reference_numerator)
        if reference_numerator
        else False,
        n_ref_beats=int(ref.size),
        n_pred_beats=int(pred.size),
    )


@dataclass(frozen=True)
class RhythmScores:
    clip_id: str
    metric_position_precision: float
    metric_position_recall: float
    metric_position_f1: float
    note_value_accuracy: float
    rest_f1: float
    ref_attacks: int
    pred_attacks: int

    def summary_line(self) -> str:
        return (
            f"{self.clip_id}: pos F1={self.metric_position_f1:.3f} "
            f"(P={self.metric_position_precision:.3f} R={self.metric_position_recall:.3f}) "
            f"note_value_acc={self.note_value_accuracy:.3f} rest_F1={self.rest_f1:.3f} "
            f"attacks {self.pred_attacks}/{self.ref_attacks}"
        )


def read_notes_and_rests(
    gp5_path: str | Path,
    *,
    track_index: int = 0,
    voice_index: int = 0,
) -> tuple[list[RhythmNote], list[RhythmRest]]:
    """Flatten a GP5 voice into attacks (tied beats folded) and rests.

    A beat whose notes are all tie-continuations is *not* a new attack; its
    duration is added to the preceding note. Attacks and rests are returned in
    score ticks, normalized so the first attack sits at tick 0.
    """
    song = guitarpro.parse(str(Path(gp5_path)))
    track = song.tracks[track_index]

    notes: list[RhythmNote] = []
    rests: list[RhythmRest] = []
    tick = 0
    for measure in track.measures:
        if voice_index >= len(measure.voices):
            continue
        for beat in measure.voices[voice_index].beats:
            dur = int(beat.duration.time)
            if beat.status == BeatStatus.rest or not beat.notes:
                rests.append(RhythmRest(tick, dur))
            elif all(n.type == NoteType.tie for n in beat.notes) and notes:
                # Tie continuation: extend the previous attack.
                prev = notes[-1]
                notes[-1] = RhythmNote(prev.tick, prev.duration + dur)
            else:
                notes.append(RhythmNote(tick, dur))
            tick += dur

    if notes:
        shift = notes[0].tick
        notes = [RhythmNote(n.tick - shift, n.duration) for n in notes]
        rests = [RhythmRest(r.tick - shift, r.duration) for r in rests]
        last_attack = notes[-1].tick
        # Drop end-of-piece padding rests (after the final attack).
        rests = [r for r in rests if r.tick < last_attack]
    return notes, rests


def _match(ref_ticks: list[int], pred_ticks: list[int], tol: int) -> list[tuple[int, int]]:
    """Greedy nearest one-to-one matching of positions within ``tol`` ticks."""
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for ri, rt in enumerate(ref_ticks):
        best_j, best_d = None, tol + 1
        for pj, pt in enumerate(pred_ticks):
            if pj in used:
                continue
            d = abs(pt - rt)
            if d <= tol and d < best_d:
                best_j, best_d = pj, d
        if best_j is not None:
            used.add(best_j)
            pairs.append((ri, best_j))
    return pairs


def compare_rhythm(
    reference_gp5: str | Path,
    predicted_gp5: str | Path,
    *,
    clip_id: str = "clip",
    position_tolerance_ticks: int = _TPQ // 8,   # 1/32 note
    duration_tolerance_ticks: int = _TPQ // 16,  # 1/64 note slack on value
) -> RhythmScores:
    """Score predicted GP5 notation against a reference GP5 (score time)."""
    ref_notes, ref_rests = read_notes_and_rests(reference_gp5)
    pred_notes, pred_rests = read_notes_and_rests(predicted_gp5)

    n_ref, n_pred = len(ref_notes), len(pred_notes)
    pairs = _match([n.tick for n in ref_notes], [n.tick for n in pred_notes],
                   position_tolerance_ticks)
    hits = len(pairs)
    p = hits / n_pred if n_pred else 0.0
    r = hits / n_ref if n_ref else 0.0
    pos_f1 = (2 * p * r / (p + r) if (p + r) > 0
              else (1.0 if not n_ref and not n_pred else 0.0))

    correct = sum(
        1 for ri, pj in pairs
        if abs(ref_notes[ri].duration - pred_notes[pj].duration) <= duration_tolerance_ticks
    )
    note_value_acc = correct / hits if hits else 0.0

    rest_pairs = _match([x.tick for x in ref_rests], [x.tick for x in pred_rests],
                        position_tolerance_ticks)
    rhits = len(rest_pairs)
    rp = rhits / len(pred_rests) if pred_rests else 0.0
    rr = rhits / len(ref_rests) if ref_rests else 0.0
    if not ref_rests and not pred_rests:
        rest_f1 = 1.0
    else:
        rest_f1 = 2 * rp * rr / (rp + rr) if (rp + rr) > 0 else 0.0

    return RhythmScores(
        clip_id=clip_id,
        metric_position_precision=p,
        metric_position_recall=r,
        metric_position_f1=pos_f1,
        note_value_accuracy=note_value_acc,
        rest_f1=rest_f1,
        ref_attacks=n_ref,
        pred_attacks=n_pred,
    )
