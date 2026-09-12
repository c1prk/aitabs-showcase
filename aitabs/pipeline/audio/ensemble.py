"""Multi-detector note reconciliation (audio ensemble).

Kong (log-mel front-end, strong high-resolution onsets) and BasicPitch (HCQT
front-end, harmonic-aware) fail on *different* notes: BasicPitch's harmonic
stacking resolves overlapping partials that Kong's mel front-end misses in
polyphony, while Kong's onset regression is crisper in time. Reconciling the two
recovers structural misses a single detector cannot reach at any threshold
(recall vs onset-threshold is empirically flat), while agreement between them is
a strong precision signal.

This module is deliberately detector-agnostic — it operates on ``NoteEvent``
lists, so the same reconciliation serves Kong+BasicPitch today and slots a
third source (e.g. a FretNet-style string expert, or vision) into the same
fusion step later.

Reconciliation modes (``mode``):
  * ``"augment"``      – keep ALL of Kong's notes (the reliable timing-authority),
                         then add BasicPitch notes above ``min_singleton_conf``.
                         Never drops a Kong note, so it never regresses Kong; only
                         adds. **Default** — the only mode that beats Kong-alone on
                         clean (0.921 vs 0.917) AND stacks with preprocessing on
                         degraded audio.
  * ``"union"``        – every note from either detector (matched pairs merged).
                         Max recall (best on degraded: 0.875 raw, 0.918 + prep),
                         small clean-precision cost.
  * ``"intersection"`` – only notes *both* detectors agree on. Max precision
                         (kills ghost notes), lower recall.
  * ``"gated"``        – gates BOTH detectors' singletons by confidence. Drops
                         real Kong notes BasicPitch misses → underperforms
                         "augment"; kept for comparison.
"""
from __future__ import annotations

from typing import Literal

from aitabs.pipeline.audio.pitch import NoteEvent, _as_note_event

Mode = Literal["union", "intersection", "gated", "augment"]

# BasicPitch thresholds calibrated on GuitarSet held-out (player-00): onset 0.7 /
# frame 0.3 / conf 0.5 (BP-alone F1 0.90 there). Calibrating BP was a big ensemble
# win — at the old untuned 0.5/0.3/0.3, BP over-detected (P 0.59) and the recall
# modes were unusable (union F1 0.71); calibrated, union F1 0.90 and the gated
# default (>=0.5) rose to P 0.90 / R 0.90 / F1 0.90 (vs 0.84/0.92/0.875 before).
# See scripts/eval_ensemble.py + RESEARCH_BASICPITCH.md.
_DEFAULT_BP_ONSET = 0.7
_DEFAULT_BP_FRAME = 0.3
_DEFAULT_BP_CONF = 0.5


def _match_pairs(
    primary: list[NoteEvent],
    secondary: list[NoteEvent],
    *,
    onset_tolerance_sec: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Greedy one-to-one match on (pitch, onset), mirroring ``metrics._match_notes``.

    Returns ``(pairs, only_primary_idx, only_secondary_idx)`` where ``pairs`` are
    ``(i, j)`` indices into ``primary``/``secondary``. Matching is greedy nearest-
    onset per primary note, so a prediction is consumed at most once — the same
    contract the evaluator uses, so reconciliation and scoring stay consistent.
    """
    used_secondary: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for i, p in enumerate(primary):
        best_j: int | None = None
        best_dt = float("inf")
        for j, s in enumerate(secondary):
            if j in used_secondary:
                continue
            if s["pitch_midi"] != p["pitch_midi"]:
                continue
            dt = abs(s["start"] - p["start"])
            if dt <= onset_tolerance_sec and dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j is not None:
            used_secondary.add(best_j)
            pairs.append((i, best_j))

    matched_primary = {i for i, _ in pairs}
    only_primary = {i for i in range(len(primary)) if i not in matched_primary}
    only_secondary = {j for j in range(len(secondary)) if j not in used_secondary}
    return pairs, only_primary, only_secondary


def _merge_pair(p: NoteEvent, s: NoteEvent, *, agreement_bonus: float) -> NoteEvent:
    """Merge an agreed pair, keeping the primary detector's (crisper) timing.

    Confidence is raised toward 1.0 by ``agreement_bonus`` because both detectors
    independently fired — agreement is evidence. Offset is the longer of the two.
    """
    conf = min(1.0, max(p["confidence"], s["confidence"]) + agreement_bonus)
    note: NoteEvent = {
        "start": p["start"],
        "end": max(p["end"], s["end"]),
        "pitch_midi": p["pitch_midi"],
        "confidence": conf,
    }
    note["source"] = "both"  # type: ignore[typeddict-unknown-key]
    return note


def reconcile_notes(
    primary: list[NoteEvent] | list[tuple],
    secondary: list[NoteEvent] | list[tuple],
    *,
    onset_tolerance_sec: float = 0.05,
    mode: Mode = "gated",
    min_singleton_conf: float = 0.5,
    agreement_bonus: float = 0.15,
) -> list[NoteEvent]:
    """Reconcile two detectors' note events into one list.

    Args:
        primary: timing-authoritative detector (pass Kong here — its onsets are
            kept for agreed notes).
        secondary: complementary detector (pass BasicPitch here).
        onset_tolerance_sec: two notes of the same pitch within this onset gap are
            the same note (default 50 ms, matching the evaluator).
        mode: ``"union"`` | ``"intersection"`` | ``"gated"`` (see module docstring).
        min_singleton_conf: in ``"gated"`` mode, min confidence for a note seen by
            only one detector to survive.
        agreement_bonus: confidence added to notes both detectors found.

    Returns:
        Reconciled ``NoteEvent`` list sorted by start time. Each note carries a
        ``"source"`` key (``"both"`` / ``"primary"`` / ``"secondary"``) for
        diagnostics.
    """
    prim = [_as_note_event(n) for n in primary]
    sec = [_as_note_event(n) for n in secondary]

    pairs, only_p, only_s = _match_pairs(
        prim, sec, onset_tolerance_sec=onset_tolerance_sec
    )

    out: list[NoteEvent] = [
        _merge_pair(prim[i], sec[j], agreement_bonus=agreement_bonus)
        for i, j in pairs
    ]

    if mode == "intersection":
        return sorted(out, key=lambda n: n["start"])

    def _tagged(note: NoteEvent, src: str) -> NoteEvent:
        n = dict(note)
        n["source"] = src  # type: ignore[typeddict-unknown-key]
        return n  # type: ignore[return-value]

    # Primary (Kong) is the reliable timing-authority — "union"/"augment" keep ALL
    # of its notes and never drop them (gating the primary was a mistake: it cost
    # real Kong notes that BasicPitch didn't corroborate). "gated" gates both.
    keep_all_primary = mode in ("union", "augment")
    for i in only_p:
        if keep_all_primary or prim[i]["confidence"] >= min_singleton_conf:
            out.append(_tagged(prim[i], "primary"))
    # Secondary (BasicPitch) singletons: "union" adds all; "augment"/"gated" add
    # only those above min_singleton_conf.
    for j in only_s:
        if mode == "union" or sec[j]["confidence"] >= min_singleton_conf:
            out.append(_tagged(sec[j], "secondary"))

    return sorted(out, key=lambda n: n["start"])


class EnsembleDetector:
    """Kong (primary/timing) + BasicPitch (complementary polyphony) reconciled.

    Runs both detectors and reconciles them with :func:`reconcile_notes`. The
    ``detect`` thresholds apply to **Kong** (the timing-authoritative primary);
    BasicPitch runs at its own internal thresholds. BasicPitch's octave/harmonic
    ghosts — its dominant false positive on guitar — are suppressed before
    merging (``suppress_bp_harmonics``), which recovered ~11 pts of union
    precision in testing.

    ``mode="augment"`` (default, ``min_singleton_conf=0.7``) keeps all of Kong's
    notes and adds only confident BasicPitch notes — it beats Kong-alone on clean
    and stacks with preprocessing on degraded audio (never regresses). Use
    ``mode="union"`` for maximum recall on degraded/real-world audio, or
    ``mode="intersection"`` for maximum precision. See
    ``scripts/eval_ensemble_preprocess.py``.
    """

    def __init__(
        self,
        kong_checkpoint: str | None = None,
        *,
        mode: Mode = "augment",
        min_singleton_conf: float = 0.7,
        onset_tolerance_sec: float = 0.05,
        bp_onset: float = _DEFAULT_BP_ONSET,
        bp_frame: float = _DEFAULT_BP_FRAME,
        bp_conf: float = _DEFAULT_BP_CONF,
        agreement_bonus: float = 0.15,
        suppress_bp_harmonics: bool = True,
        kong_pcen: bool = False,
    ) -> None:
        from aitabs.pipeline.audio.pitch import BasicPitchDetector, KongDetector

        self._kong = KongDetector(kong_checkpoint, pcen=kong_pcen)
        self._bp = BasicPitchDetector()
        self._mode = mode
        self._min_singleton_conf = min_singleton_conf
        self._tol = onset_tolerance_sec
        self._bp_onset = bp_onset
        self._bp_frame = bp_frame
        self._bp_conf = bp_conf
        self._agreement_bonus = agreement_bonus
        self._suppress_bp_harmonics = suppress_bp_harmonics

    def detect(
        self,
        audio_path: str,
        *,
        onset_threshold: float,
        frame_threshold: float,
        confidence_threshold: float,
        minimum_note_length_ms: float,
        minimum_frequency_hz: float | None,
        maximum_frequency_hz: float | None,
        melodia_trick: bool,
    ) -> list[NoteEvent]:
        from aitabs.pipeline.audio.pitch import clean_notes, suppress_octave_harmonics

        kong = self._kong.detect(
            audio_path,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            confidence_threshold=confidence_threshold,
            minimum_note_length_ms=minimum_note_length_ms,
            minimum_frequency_hz=minimum_frequency_hz,
            maximum_frequency_hz=maximum_frequency_hz,
            melodia_trick=melodia_trick,
        )
        bp = self._bp.detect(
            audio_path,
            onset_threshold=self._bp_onset,
            frame_threshold=self._bp_frame,
            confidence_threshold=self._bp_conf,
            minimum_note_length_ms=minimum_note_length_ms,
            minimum_frequency_hz=minimum_frequency_hz,
            maximum_frequency_hz=maximum_frequency_hz,
            melodia_trick=melodia_trick,
        )
        if self._suppress_bp_harmonics:
            bp = suppress_octave_harmonics(clean_notes(bp), max_conf_ratio=1.0)

        return reconcile_notes(
            kong, bp,
            onset_tolerance_sec=self._tol,
            mode=self._mode,
            min_singleton_conf=self._min_singleton_conf,
            agreement_bonus=self._agreement_bonus,
        )
