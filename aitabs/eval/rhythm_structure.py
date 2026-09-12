"""Rhythm-structure diagnostics from note onsets (no ML deps).

Answers the question screenshots can't: *is the rhythmic variety present in the
detected onsets, or did notation flatten it?* When a transcription comes out as
"nearly all eighth notes" while the ground truth mixes eighths/sixteenths/dotted
figures, there are two opposite causes:

* **Notation flattened it** — the onsets carry sub-eighth inter-onset intervals
  (real sixteenths), but a variety-killing quantizer (``snap_dominant_grid`` /
  ``legato``) or too-coarse a grid collapsed them. Fix is in notation.
* **Detection never resolved it** — the predicted onsets themselves are spaced at
  ~eighth-note intervals; the fast notes were never detected (recall/timing). No
  notation change can recover them; the fix is upstream (onset/detector).

This module profiles the inter-onset-interval (IOI) distribution of a note list
in *beats*, so the two cases are distinguishable from data, not from a render.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical IOI values in beats and the label we report them under. A measured
# IOI is assigned to the nearest value in log space (ratios, not absolute beats,
# match how rhythm tolerance scales). Anything below the sixteenth boundary is
# "finer" (32nds / very fast). Triplet-eighth (1/3) is included so triplet
# figures are not misread as sixteenths.
_GRID_BEATS: list[tuple[float, str]] = [
    (2.0, "half+"),
    (1.5, "dotted-quarter"),
    (1.0, "quarter"),
    (0.75, "dotted-eighth"),
    (2 / 3, "quarter-triplet"),
    (0.5, "eighth"),
    (1 / 3, "eighth-triplet"),
    (0.25, "sixteenth"),
]
# IOIs at or below this many beats require a sixteenth-or-finer grid to notate;
# their presence is the signature of genuine rhythmic variety beyond eighths.
SUB_EIGHTH_BEATS = 0.375  # midpoint between eighth (0.5) and sixteenth (0.25), log-ish
# Below the geometric midpoint of a sixteenth (0.25) and a 32nd (0.125) we call
# the IOI "finer" than a sixteenth rather than snapping it to the 16th bucket.
FINER_BEATS = 0.177


def collapse_onsets(
    notes: list[dict], *, chord_window_sec: float = 0.05
) -> list[float]:
    """Return sorted distinct onset times, merging near-simultaneous attacks.

    Chords/double-stops share one rhythmic onset; without merging they would
    register as zero-length IOIs and inflate the "very fast" bucket.
    """
    starts = sorted(float(n["start"]) for n in notes)
    out: list[float] = []
    for s in starts:
        if not out or s - out[-1] > chord_window_sec:
            out.append(s)
    return out


def onset_iois_in_beats(
    notes: list[dict], bpm: float, *, chord_window_sec: float = 0.05
) -> list[float]:
    """Consecutive inter-onset intervals in beats (quarter = 1.0)."""
    if bpm <= 0:
        return []
    onsets = collapse_onsets(notes, chord_window_sec=chord_window_sec)
    beat_per_sec = bpm / 60.0
    return [(b - a) * beat_per_sec for a, b in zip(onsets, onsets[1:])]


def classify_ioi(ioi_beats: float) -> str:
    """Label an IOI by its nearest canonical rhythmic value (log-ratio nearest)."""
    if ioi_beats <= 0:
        return "finer"
    if ioi_beats > 2.0:
        return "half+"
    import math

    if ioi_beats < FINER_BEATS:
        return "finer"
    best = min(_GRID_BEATS, key=lambda gv: abs(math.log(ioi_beats / gv[0])))
    return best[1]


@dataclass
class RhythmProfile:
    n_onsets: int
    n_iois: int
    median_ioi_beats: float
    sub_eighth_fraction: float          # share of IOIs needing 16th-or-finer
    counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        order = [g[1] for g in _GRID_BEATS] + ["finer"]
        parts = [
            f"{lbl}:{self.counts.get(lbl, 0)}"
            for lbl in order
            if self.counts.get(lbl, 0)
        ]
        return (
            f"onsets={self.n_onsets} iois={self.n_iois} "
            f"median={self.median_ioi_beats:.3f}beat "
            f"sub-eighth={self.sub_eighth_fraction:.0%}  [{' '.join(parts)}]"
        )


def profile_onsets(
    notes: list[dict], bpm: float, *, chord_window_sec: float = 0.05
) -> RhythmProfile:
    iois = onset_iois_in_beats(notes, bpm, chord_window_sec=chord_window_sec)
    counts: dict[str, int] = {}
    for x in iois:
        counts[classify_ioi(x)] = counts.get(classify_ioi(x), 0) + 1
    sub = sum(1 for x in iois if x < SUB_EIGHTH_BEATS)
    median = sorted(iois)[len(iois) // 2] if iois else 0.0
    onsets = collapse_onsets(notes, chord_window_sec=chord_window_sec)
    return RhythmProfile(
        n_onsets=len(onsets),
        n_iois=len(iois),
        median_ioi_beats=float(median),
        sub_eighth_fraction=(sub / len(iois)) if iois else 0.0,
        counts=counts,
    )


def diagnose(
    prediction: list[dict],
    reference: list[dict] | None,
    bpm: float,
    *,
    chord_window_sec: float = 0.05,
    miss_ratio: float = 0.5,
) -> tuple[str, RhythmProfile, RhythmProfile | None]:
    """Return a verdict plus the profiles.

    Verdict logic (only meaningful when a reference is available):

    * reference has substantial sub-eighth IOIs but the prediction's onsets do
      **not** (pred sub-eighth < ``miss_ratio`` x ref) -> the fast notes are
      missing from the detected *onsets*: a **detector/recall** problem; notation
      tweaks cannot bring them back.
    * prediction's onsets **do** carry sub-eighth IOIs comparable to the reference
      -> the variety is in the data, so an all-eighths render means **notation
      flattened it** (suspect ``snap_dominant_grid`` / ``legato`` / coarse grid).
    """
    pred = profile_onsets(prediction, bpm, chord_window_sec=chord_window_sec)
    ref = (
        profile_onsets(reference, bpm, chord_window_sec=chord_window_sec)
        if reference
        else None
    )
    if ref is None:
        if pred.sub_eighth_fraction < 0.05:
            verdict = (
                "Predicted onsets are ~all eighth-or-longer (sub-eighth "
                f"{pred.sub_eighth_fraction:.0%}). No reference given, but the "
                "detector did not produce sixteenth-spaced onsets — if the take "
                "has real sixteenths this is an onset/recall problem upstream, "
                "not a notation one."
            )
        else:
            verdict = (
                f"Predicted onsets carry sub-eighth structure "
                f"({pred.sub_eighth_fraction:.0%}). If the render still shows all "
                "eighths, notation is flattening it (check snap_dominant_grid / "
                "legato / coarse grid)."
            )
        return verdict, pred, ref

    if ref.sub_eighth_fraction < 0.05:
        verdict = (
            "Reference itself is ~all eighths — the all-eighths prediction is "
            "rhythmically faithful; differences are pitch, not rhythm."
        )
    elif pred.sub_eighth_fraction < miss_ratio * ref.sub_eighth_fraction:
        verdict = (
            f"DETECTOR/RECALL: reference is {ref.sub_eighth_fraction:.0%} "
            f"sub-eighth but predicted onsets are only "
            f"{pred.sub_eighth_fraction:.0%} — the sixteenths are missing from "
            "the detected onsets, so no notation change recovers them. Fix "
            "upstream (onset density / detector recall)."
        )
    else:
        verdict = (
            f"NOTATION-FLATTEN: predicted onsets carry sub-eighth structure "
            f"({pred.sub_eighth_fraction:.0%} vs ref {ref.sub_eighth_fraction:.0%}), "
            "so an all-eighths render is notation flattening the variety. Set "
            "snap_dominant_grid=False (it locks a mixed-rhythm phrase to one grid "
            "when sub-eighth < ~48%); keep legato=True for readable durations; do "
            "NOT use optimize_grid for mixed rhythms (its complexity penalty drops "
            "minority sixteenths). The per-beat quantizer keeps mixed resolution."
        )
    return verdict, pred, ref
