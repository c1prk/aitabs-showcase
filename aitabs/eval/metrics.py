"""Note-level and tab-level comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aitabs.pipeline.audio.tempo import TempoAnalysis, onset_to_tick

from .types import EvalNote


@dataclass(frozen=True)
class ClipScores:
    """Scores for one clip (reference vs prediction)."""

    clip_id: str
    time_offset_sec: float
    reference_bpm: float
    estimated_bpm: float

    ref_note_count: int
    pred_note_count: int

    pitch_precision: float
    pitch_recall: float
    pitch_f1: float
    pitch_onset_mae_sec: float

    tab_precision: float
    tab_recall: float
    tab_f1: float

    def summary_line(self) -> str:
        return (
            f"{self.clip_id}: pitch F1={self.pitch_f1:.3f} "
            f"(P={self.pitch_precision:.3f} R={self.pitch_recall:.3f}) "
            f"tab F1={self.tab_f1:.3f} onset_MAE={self.pitch_onset_mae_sec*1000:.0f}ms "
            f"offset={self.time_offset_sec:+.3f}s "
            f"notes {self.pred_note_count}/{self.ref_note_count}"
        )


def _shift_notes(notes: list[EvalNote], offset_sec: float) -> list[EvalNote]:
    if offset_sec == 0:
        return notes
    return [
        EvalNote(
            start=n["start"] + offset_sec,
            end=n["end"] + offset_sec,
            pitch_midi=n["pitch_midi"],
            string=n["string"],
            fret=n["fret"],
            confidence=n["confidence"],
        )
        for n in notes
    ]


def estimate_time_offset(
    reference: list[EvalNote],
    prediction: list[EvalNote],
    *,
    onset_tolerance_sec: float = 0.05,
    max_shift_sec: float = 1.0,
    step_sec: float = 0.01,
) -> float:
    """Search constant time shift on prediction to maximize pitch matches."""
    if not reference or not prediction:
        return 0.0

    best_offset = 0.0
    best_score = -1.0
    shifts = np.arange(-max_shift_sec, max_shift_sec + step_sec * 0.5, step_sec)

    for offset in shifts:
        shifted = _shift_notes(prediction, float(offset))
        hits, errors, _ = _match_notes(
            reference,
            shifted,
            onset_tolerance_sec=onset_tolerance_sec,
            match_tab=False,
        )
        # Prefer more hits; tie-break by lower mean onset error.
        score = float(hits) - 0.01 * float(np.mean(errors) if errors else 0.0)
        if score > best_score:
            best_score = score
            best_offset = float(offset)

    return best_offset


def _warp_notes(notes: list[EvalNote], scale: float, offset: float) -> list[EvalNote]:
    """Apply a linear time warp: t' = scale*t + offset (for tempo-robust matching)."""
    return [
        EvalNote(
            start=n["start"] * scale + offset,
            end=n["end"] * scale + offset,
            pitch_midi=n["pitch_midi"],
            string=n["string"],
            fret=n["fret"],
            confidence=n["confidence"],
        )
        for n in notes
    ]


def estimate_time_warp(
    reference: list[EvalNote],
    prediction: list[EvalNote],
    *,
    onset_tolerance_sec: float = 0.05,
    scale_range: tuple[float, float] = (0.85, 1.15),
    scale_step: float = 0.005,
    max_shift_sec: float = 0.75,
    shift_step: float = 0.01,
) -> tuple[float, float]:
    """Find a linear (scale, offset) warp of the prediction maximizing pitch hits.

    Corrects a *global tempo mismatch* between a notated reference (constant
    tempo) and a human performance — the constant-offset search alone cannot.
    ``scale`` ~ notated_tempo / performed_tempo. Returns (1.0, 0.0) on empty input.
    """
    if not reference or not prediction:
        return 1.0, 0.0
    ref_t = np.array([n["start"] for n in reference], dtype=float)
    ref_p = np.array([n["pitch_midi"] for n in reference], dtype=int)
    pred_t = np.array([n["start"] for n in prediction], dtype=float)
    pred_p = np.array([n["pitch_midi"] for n in prediction], dtype=int)

    def count_hits(pt: np.ndarray) -> int:
        used = np.zeros(pt.shape[0], dtype=bool)
        hits = 0
        for rt, rp in zip(ref_t, ref_p):
            cand = np.where((~used) & (pred_p == rp) & (np.abs(pt - rt) <= onset_tolerance_sec))[0]
            if cand.size:
                used[cand[np.argmin(np.abs(pt[cand] - rt))]] = True
                hits += 1
        return hits

    scales = np.arange(scale_range[0], scale_range[1] + scale_step * 0.5, scale_step)
    shifts = np.arange(-max_shift_sec, max_shift_sec + shift_step * 0.5, shift_step)
    best_hits, best = -1, (1.0, 0.0)
    for s in scales:
        base = pred_t * s
        for off in shifts:
            h = count_hits(base + off)
            if h > best_hits:
                best_hits, best = h, (float(s), float(off))
    return best


def _align_lcs(
    reference: list[EvalNote],
    prediction: list[EvalNote],
    *,
    match_tab: bool,
    band_frac: float = 0.2,
) -> int:
    """Monotonic (order-preserving) pitch match — DTW-style, time-agnostic.

    A banded longest-common-subsequence on the pitch sequence: counts notes that
    match in the same order, allowing arbitrary local tempo elasticity (rubato).
    The Sakoe-Chiba band (``band_frac`` of the longer sequence) stops a note from
    matching a same-pitch note far away in the piece. This is an *upper-bound*
    style score: it ignores absolute timing entirely, so use it as a ceiling, not
    as the primary accuracy number.
    """
    R = sorted(reference, key=lambda n: n["start"])
    P = sorted(prediction, key=lambda n: n["start"])
    nR, nP = len(R), len(P)
    if nR == 0 or nP == 0:
        return 0
    band = max(10, int(band_frac * max(nR, nP)))

    def eq(a: EvalNote, b: EvalNote) -> bool:
        if a["pitch_midi"] != b["pitch_midi"]:
            return False
        if match_tab and (a["string"] != b["string"] or a["fret"] != b["fret"]):
            return False
        return True

    prev = [0] * (nP + 1)
    for i in range(1, nR + 1):
        cur = [0] * (nP + 1)
        center = i * nP // nR
        jlo, jhi = max(1, center - band), min(nP, center + band)
        for j in range(1, nP + 1):
            if j < jlo or j > jhi:
                cur[j] = max(prev[j], cur[j - 1])
            elif eq(R[i - 1], P[j - 1]):
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[nP]


def _match_notes(
    reference: list[EvalNote],
    prediction: list[EvalNote],
    *,
    onset_tolerance_sec: float,
    match_tab: bool,
) -> tuple[int, list[float], list[tuple[EvalNote, EvalNote]]]:
    """Greedy one-to-one matching on pitch (+ optional string/fret)."""
    used_pred: set[int] = set()
    hits = 0
    onset_errors: list[float] = []
    pairs: list[tuple[EvalNote, EvalNote]] = []

    for ref in reference:
        best_j: int | None = None
        best_dt = float("inf")
        for j, pred in enumerate(prediction):
            if j in used_pred:
                continue
            if pred["pitch_midi"] != ref["pitch_midi"]:
                continue
            if match_tab and (pred["string"] != ref["string"] or pred["fret"] != ref["fret"]):
                continue
            dt = abs(pred["start"] - ref["start"])
            if dt <= onset_tolerance_sec and dt < best_dt:
                best_dt = dt
                best_j = j

        if best_j is not None:
            used_pred.add(best_j)
            hits += 1
            onset_errors.append(best_dt)
            pairs.append((ref, prediction[best_j]))

    return hits, onset_errors, pairs


def rhythm_tick_f1(
    reference: list[EvalNote],
    prediction: list[EvalNote],
    analysis: TempoAnalysis,
    *,
    tick_tolerance: int = 0,
) -> tuple[float, float, float]:
    """Pitch-agnostic F1 on quantized onset ticks (rhythm alignment)."""
    ref_ticks = {onset_to_tick(n["start"], analysis) for n in reference}
    pred_ticks = {onset_to_tick(n["start"], analysis) for n in prediction}
    if not ref_ticks and not pred_ticks:
        return 1.0, 1.0, 1.0
    if not ref_ticks or not pred_ticks:
        return 0.0, 0.0, 0.0

    def _match_count(ref: set[int], pred: set[int]) -> int:
        hits = 0
        used: set[int] = set()
        for rt in sorted(ref):
            for pt in sorted(pred):
                if pt in used:
                    continue
                if abs(pt - rt) <= tick_tolerance:
                    hits += 1
                    used.add(pt)
                    break
        return hits

    hits = _match_count(ref_ticks, pred_ticks)
    p = hits / len(pred_ticks) if pred_ticks else 0.0
    r = hits / len(ref_ticks) if ref_ticks else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compare_note_lists(
    reference: list[EvalNote],
    prediction: list[EvalNote],
    *,
    clip_id: str = "clip",
    onset_tolerance_sec: float = 0.05,
    time_offset_sec: float | None = None,
    reference_bpm: float = 0.0,
    estimated_bpm: float = 0.0,
    estimate_offset: bool = True,
    align: str = "none",
) -> ClipScores:
    """Compare reference and predicted note lists.

    ``align`` controls how prediction timing is reconciled with the reference
    before matching — critical when the reference is a *notated* score (GP5,
    constant tempo) but the audio is a *human performance* (different tempo /
    rubato):

    * ``"none"``  – constant global offset only (default; correct for
      performance-timed refs like GuitarSet JAMS).
    * ``"warp"``  – global linear tempo warp (scale + offset). Corrects a tempo
      mismatch a constant offset cannot. Recommended for GP5 references.
    * ``"dtw"``   – monotonic order-preserving pitch match (time-agnostic).
      Tolerates arbitrary rubato/drift; onset MAE is not meaningful here. Use as
      a cross-check; ``"warp"`` is the recommended accuracy metric for GP5 refs.
    """
    n_ref = len(reference)

    if align == "dtw":
        pitch_hits = _align_lcs(reference, prediction, match_tab=False)
        tab_hits = _align_lcs(reference, prediction, match_tab=True)
        pitch_errors: list[float] = []
        offset = 0.0
        pred_shifted = prediction
        n_pred = len(prediction)
    else:
        if align == "warp":
            scale, offset = estimate_time_warp(
                reference, prediction, onset_tolerance_sec=onset_tolerance_sec
            )
            pred_shifted = _warp_notes(prediction, scale, offset)
        else:  # "none"
            offset = time_offset_sec
            if offset is None and estimate_offset:
                offset = estimate_time_offset(
                    reference, prediction, onset_tolerance_sec=onset_tolerance_sec
                )
            elif offset is None:
                offset = 0.0
            pred_shifted = _shift_notes(prediction, offset)

        pitch_hits, pitch_errors, _ = _match_notes(
            reference, pred_shifted, onset_tolerance_sec=onset_tolerance_sec, match_tab=False,
        )
        tab_hits, _, _ = _match_notes(
            reference, pred_shifted, onset_tolerance_sec=onset_tolerance_sec, match_tab=True,
        )
        n_pred = len(pred_shifted)

    def prf(hits: int, n_ref: int, n_pred: int) -> tuple[float, float, float]:
        p = hits / n_pred if n_pred else 0.0
        r = hits / n_ref if n_ref else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    pp, pr, pf1 = prf(pitch_hits, n_ref, n_pred)
    tp, tr, tf1 = prf(tab_hits, n_ref, n_pred)
    mae = float(np.mean(pitch_errors)) if pitch_errors else float("nan")

    return ClipScores(
        clip_id=clip_id,
        time_offset_sec=offset,
        reference_bpm=reference_bpm,
        estimated_bpm=estimated_bpm,
        ref_note_count=n_ref,
        pred_note_count=n_pred,
        pitch_precision=pp,
        pitch_recall=pr,
        pitch_f1=pf1,
        pitch_onset_mae_sec=mae,
        tab_precision=tp,
        tab_recall=tr,
        tab_f1=tf1,
    )
