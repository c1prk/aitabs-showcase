"""Characterize prediction false positives by their relation to the reference.

The octave-ghost filter (Track 2 v1) was a net negative: it removed real octaves
on clean clips while barely denting the over-detection on test7/test5. Rather than
guess again, this module classifies each *unmatched* predicted note by how it
relates to concurrent reference notes, so the dominant false-positive type drives
the next filter.

Categories (priority order):
  octave            — interval ±12/±24 from a concurrent reference note (overtone)
  fifth_or_twelfth  — ±7/±19 (perfect fifth / 3rd-harmonic alias)
  duplicate_sustain — same pitch as a concurrent reference note but mistimed
  other_concurrent  — some reference sounds, but no harmonic relation (wrong note)
  isolated          — nothing in the reference sounds there (spurious / noise)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .metrics import _shift_notes, estimate_time_offset
from .types import EvalNote

_OCTAVE = {12, -12, 24, -24}
_FIFTH = {7, 19, -5, -17, 28}  # perfect fifth, octave+fifth (3rd harmonic), inversions


def _matched_pred_indices(
    reference: list[EvalNote],
    prediction: list[EvalNote],
    tol: float,
) -> set[int]:
    """Greedy pitch+onset match → indices of predictions that hit a reference."""
    used: set[int] = set()
    for ref in reference:
        best_j: int | None = None
        best_dt = tol + 1.0
        for j, pred in enumerate(prediction):
            if j in used or pred["pitch_midi"] != ref["pitch_midi"]:
                continue
            dt = abs(pred["start"] - ref["start"])
            if dt <= tol and dt < best_dt:
                best_dt, best_j = dt, j
        if best_j is not None:
            used.add(best_j)
    return used


def _matched_ref_indices(
    reference: list[EvalNote],
    prediction: list[EvalNote],
    tol: float,
) -> set[int]:
    """Greedy pitch+onset match → indices of *reference* notes that were detected."""
    used_pred: set[int] = set()
    matched_ref: set[int] = set()
    for i, ref in enumerate(reference):
        best_j: int | None = None
        best_dt = tol + 1.0
        for j, pred in enumerate(prediction):
            if j in used_pred or pred["pitch_midi"] != ref["pitch_midi"]:
                continue
            dt = abs(pred["start"] - ref["start"])
            if dt <= tol and dt < best_dt:
                best_dt, best_j = dt, j
        if best_j is not None:
            used_pred.add(best_j)
            matched_ref.add(i)
    return matched_ref


@dataclass
class MissBreakdown:
    """Why reference notes were *missed* (false negatives → recall loss)."""

    clip_id: str
    n_ref: int
    n_matched: int
    n_missed: int
    chord_inner: int = 0     # other voice(s) of a chord we partly caught
    masked_by_sustain: int = 0  # a different note was ringing (let-ring) at its onset
    octave_clash: int = 0    # an octave of a concurrent note (overtone confusion)
    low_register: int = 0    # missed sustained bass (< E3)
    isolated: int = 0        # a plain missed note, nothing else explains it

    def as_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        def pct(x: int) -> str:
            return f"{(100.0 * x / self.n_missed):.0f}%" if self.n_missed else "—"

        return (
            f"{self.clip_id}: missed={self.n_missed}/{self.n_ref} "
            f"chord_inner={pct(self.chord_inner)} sustain={pct(self.masked_by_sustain)} "
            f"octave={pct(self.octave_clash)} bass={pct(self.low_register)} "
            f"isolated={pct(self.isolated)}"
        )


def categorize_false_negatives(
    reference: list[EvalNote],
    prediction: list[EvalNote],
    *,
    clip_id: str = "clip",
    onset_tolerance_sec: float = 0.05,
    concurrency_sec: float = 0.05,
    low_register_midi: int = 52,
    estimate_offset: bool = True,
) -> MissBreakdown:
    """Classify *missed* reference notes to explain recall loss.

    Priority per missed note: chord-inner (we caught a simultaneous note) →
    masked-by-sustain (a different note was ringing) → octave clash → low register
    → isolated. This separates "polyphony/recall" misses from plain detection gaps.
    """
    offset = (
        estimate_time_offset(reference, prediction, onset_tolerance_sec=onset_tolerance_sec)
        if estimate_offset
        else 0.0
    )
    pred = _shift_notes(prediction, offset)
    matched_ref = _matched_ref_indices(reference, pred, onset_tolerance_sec)

    bd = MissBreakdown(
        clip_id=clip_id,
        n_ref=len(reference),
        n_matched=len(matched_ref),
        n_missed=0,
    )

    for i, r in enumerate(reference):
        if i in matched_ref:
            continue
        bd.n_missed += 1

        concurrent = [
            (j, o)
            for j, o in enumerate(reference)
            if j != i and abs(o["start"] - r["start"]) <= concurrency_sec
        ]
        sustaining = [
            o
            for o in reference
            if o["pitch_midi"] != r["pitch_midi"]
            and o["start"] < r["start"] - concurrency_sec
            and o["end"] > r["start"]
        ]

        if any(j in matched_ref for j, _ in concurrent):
            bd.chord_inner += 1
        elif any(abs(r["pitch_midi"] - o["pitch_midi"]) in (12, 24) for _, o in concurrent):
            bd.octave_clash += 1
        elif sustaining:
            bd.masked_by_sustain += 1
        elif r["pitch_midi"] < low_register_midi:
            bd.low_register += 1
        else:
            bd.isolated += 1

    return bd


@dataclass
class FalsePositiveBreakdown:
    clip_id: str
    n_ref: int
    n_pred: int
    n_matched: int
    n_false: int
    octave: int = 0
    fifth_or_twelfth: int = 0
    duplicate_sustain: int = 0
    other_concurrent: int = 0
    isolated: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        def pct(x: int) -> str:
            return f"{(100.0 * x / self.n_false):.0f}%" if self.n_false else "—"

        return (
            f"{self.clip_id}: FP={self.n_false}/{self.n_pred} "
            f"oct={pct(self.octave)} 5th={pct(self.fifth_or_twelfth)} "
            f"dup={pct(self.duplicate_sustain)} other={pct(self.other_concurrent)} "
            f"isolated={pct(self.isolated)}"
        )


def categorize_false_positives(
    reference: list[EvalNote],
    prediction: list[EvalNote],
    *,
    clip_id: str = "clip",
    onset_tolerance_sec: float = 0.05,
    concurrency_sec: float = 0.07,
    estimate_offset: bool = True,
) -> FalsePositiveBreakdown:
    """Classify unmatched predictions relative to concurrent reference notes."""
    offset = (
        estimate_time_offset(reference, prediction, onset_tolerance_sec=onset_tolerance_sec)
        if estimate_offset
        else 0.0
    )
    pred = _shift_notes(prediction, offset)
    matched = _matched_pred_indices(reference, pred, onset_tolerance_sec)

    bd = FalsePositiveBreakdown(
        clip_id=clip_id,
        n_ref=len(reference),
        n_pred=len(pred),
        n_matched=len(matched),
        n_false=0,
    )

    for j, p in enumerate(pred):
        if j in matched:
            continue
        bd.n_false += 1
        concurrent = [
            r
            for r in reference
            if abs(r["start"] - p["start"]) <= concurrency_sec
            or (r["start"] <= p["start"] <= r["end"])
        ]
        if not concurrent:
            bd.isolated += 1
            continue
        intervals = {p["pitch_midi"] - r["pitch_midi"] for r in concurrent}
        if intervals & _OCTAVE:
            bd.octave += 1
        elif intervals & _FIFTH:
            bd.fifth_or_twelfth += 1
        elif 0 in intervals:
            bd.duplicate_sustain += 1
        else:
            bd.other_concurrent += 1

    return bd
