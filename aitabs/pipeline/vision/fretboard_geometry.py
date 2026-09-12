"""Canonical fretboard geometry + homography mapping (Stage-1 vision, no training).

The license-clean path from RESEARCH_VISION.md §2: instead of a trained fretboard
detector, put fiducial markers on the guitar, recover a homography that maps a
*canonical* fretboard (nut at x=0, bridge at x=scale_length; strings evenly spaced
in y) to the image, and read off (string, fret) purely by geometry. Robust to the
guitar moving, zero annotation, commercial-safe.

Fret spacing is the physical **12-tone equal-temperament rule** — fret ``n`` sits at
``scale_length * (1 - 2**(-n/12))`` from the nut (so fret 12 is exactly half the
scale). A *fretted* note presses just behind a fret, so the press position for fret
``n`` is the midpoint between fret lines ``n-1`` and ``n``.

This module is pure numpy — the homography is applied/inverted with matrix math, so
the geometry is fully unit-testable without OpenCV or a camera. OpenCV is only
needed to *estimate* the homography from marker correspondences (:func:`homography`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aitabs.pipeline.mapping.guitar import MAX_FRETS, STANDARD_TUNING


@dataclass(frozen=True)
class FretboardGeometry:
    """Canonical fretboard coordinate system (guitar-frame, arbitrary units).

    x runs along the neck (nut=0 → bridge=scale_length); y runs across the strings
    (string 0 = low E at y=0 → string ``num_strings-1`` = high e).
    """

    scale_length: float = 1.0
    num_strings: int = len(STANDARD_TUNING)
    num_frets: int = MAX_FRETS
    string_spacing: float = 1.0

    def fret_line_x(self, fret: int) -> float:
        """x of the *fret wire* for ``fret`` (0 = nut). 12-TET: fret 12 = scale/2."""
        return self.scale_length * (1.0 - 2.0 ** (-fret / 12.0))

    def press_x(self, fret: int) -> float:
        """x where a finger presses for ``fret`` (midpoint behind the wire; 0=open)."""
        if fret <= 0:
            return 0.0
        return 0.5 * (self.fret_line_x(fret - 1) + self.fret_line_x(fret))

    def string_y(self, string: int) -> float:
        return string * self.string_spacing

    def cell_center(self, string: int, fret: int) -> tuple[float, float]:
        """Canonical (x, y) of the press point for a (string, fret) cell."""
        return (self.press_x(fret), self.string_y(string))

    def fret_line_segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Canonical endpoint pairs for each fret wire (spanning all strings)."""
        y0, y1 = self.string_y(0), self.string_y(self.num_strings - 1)
        return [((self.fret_line_x(f), y0), (self.fret_line_x(f), y1))
                for f in range(self.num_frets + 1)]

    def string_line_segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Canonical endpoint pairs for each string (nut to last fret)."""
        x0, x1 = 0.0, self.fret_line_x(self.num_frets)
        return [((x0, self.string_y(s)), (x1, self.string_y(s)))
                for s in range(self.num_strings)]

    def all_cell_centers(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Return (points (N,2), [(string, fret)...]) for every cell (fret 0..N)."""
        pts, cells = [], []
        for s in range(self.num_strings):
            for f in range(self.num_frets + 1):
                pts.append(self.cell_center(s, f))
                cells.append((s, f))
        return np.asarray(pts, dtype=np.float64), cells


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map points (N,2) through 3x3 homography H. Returns (N,2)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)   # (N,3)
    out = hom @ H.T                                               # (N,3)
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return out[:, :2] / w


def image_to_canonical(H: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    """Inverse-map image points to the canonical fretboard frame (H: canon→image)."""
    return apply_homography(np.linalg.inv(H), image_pts)


def nearest_cell(
    canonical_xy: np.ndarray,
    geom: FretboardGeometry,
    *,
    max_string_dist: float | None = None,
    max_fret_frac: float = 0.6,
) -> tuple[int, int] | None:
    """Nearest (string, fret) cell to a point already in the canonical frame.

    Returns ``None`` when the point is implausibly far from any cell (finger not on
    the fretboard). ``max_string_dist`` defaults to half the string spacing;
    ``max_fret_frac`` bounds the along-neck distance as a fraction of the local
    fret width so tolerance shrinks toward the high frets (matching real spacing).
    """
    x, y = float(canonical_xy[0]), float(canonical_xy[1])
    if max_string_dist is None:
        max_string_dist = 0.5 * geom.string_spacing

    s = int(round(y / geom.string_spacing)) if geom.string_spacing else 0
    if not (0 <= s < geom.num_strings) or abs(y - geom.string_y(s)) > max_string_dist:
        return None

    best_f, best_dx = None, np.inf
    for f in range(geom.num_frets + 1):
        dx = abs(x - geom.press_x(f))
        if dx < best_dx:
            best_dx, best_f = dx, f
    if best_f is None:
        return None
    # local fret width (fret 0 uses fret-1 width) for scale-aware tolerance
    lo = max(1, best_f)
    width = geom.fret_line_x(lo) - geom.fret_line_x(lo - 1)
    if best_dx > max_fret_frac * max(width, 1e-9):
        return None
    return s, best_f


def image_point_to_cell(
    image_xy, H: np.ndarray, geom: FretboardGeometry, **kw
) -> tuple[int, int] | None:
    """Map a fingertip pixel through the homography to a (string, fret) cell.

    ``H`` is the canonical→image homography for this frame. Convenience wrapper:
    invert to the fretboard frame, then :func:`nearest_cell`. Returns ``None`` when
    the fingertip isn't on the fretboard.
    """
    canon = image_to_canonical(H, np.atleast_2d(image_xy))[0]
    return nearest_cell(canon, geom, **kw)


def homography(canonical_pts: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    """Estimate the canonical→image homography from >=4 correspondences (OpenCV)."""
    import cv2

    H, _ = cv2.findHomography(
        np.asarray(canonical_pts, dtype=np.float64),
        np.asarray(image_pts, dtype=np.float64),
        method=cv2.RANSAC,
    )
    return H
