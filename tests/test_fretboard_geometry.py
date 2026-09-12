"""Tests for canonical fretboard geometry + homography mapping (Stage-1 vision)."""

from __future__ import annotations

import numpy as np

from aitabs.pipeline.vision.fretboard_geometry import (
    FretboardGeometry,
    apply_homography,
    image_point_to_cell,
    image_to_canonical,
    nearest_cell,
)


def test_fret_spacing_is_12tet():
    g = FretboardGeometry(scale_length=650.0)  # mm
    assert g.fret_line_x(0) == 0.0
    assert abs(g.fret_line_x(12) - 325.0) < 1e-6      # octave at half the scale
    assert abs(g.fret_line_x(24) - 487.5) < 1e-6      # two octaves at 3/4
    # fret wires get closer together toward the bridge
    w1 = g.fret_line_x(1) - g.fret_line_x(0)
    w12 = g.fret_line_x(12) - g.fret_line_x(11)
    assert w12 < w1


def test_press_positions_monotonic():
    g = FretboardGeometry(scale_length=650.0)
    assert g.press_x(0) == 0.0
    xs = [g.press_x(f) for f in range(13)]
    assert all(b > a for a, b in zip(xs, xs[1:]))     # strictly increasing


def _synthetic_homography() -> np.ndarray:
    # canonical (mm) -> image (px): scale, rotate ~12 deg, translate, mild perspective
    th = np.deg2rad(12.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = np.eye(3)
    A[:2, :2] = R * 1.4
    A[:2, 2] = [220.0, 130.0]
    A[2, 0] = 1e-4   # a touch of perspective
    return A


def test_homography_roundtrip_recovers_cells():
    g = FretboardGeometry(scale_length=650.0, string_spacing=10.0)
    H = _synthetic_homography()
    pts, cells = g.all_cell_centers()
    img = apply_homography(H, pts)
    back = image_to_canonical(H, img)
    assert np.allclose(back, pts, atol=1e-6)          # exact inverse
    # every projected cell maps back to its own (string, fret)
    hits = sum(nearest_cell(image_to_canonical(H, img[i:i+1])[0], g) == cells[i]
               for i in range(len(cells)))
    assert hits == len(cells)


def test_nearest_cell_rejects_off_board():
    g = FretboardGeometry(scale_length=650.0, string_spacing=10.0)
    # way above the top string and off the neck end
    assert nearest_cell(np.array([0.0, 500.0]), g) is None
    # a point sitting on a real cell resolves to it
    assert nearest_cell(np.array(g.cell_center(3, 5)), g) == (3, 5)


def test_apply_homography_identity():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(apply_homography(np.eye(3), pts), pts)


def test_image_point_to_cell_maps_fingertip():
    g = FretboardGeometry(scale_length=650.0, string_spacing=10.0)
    H = _synthetic_homography()
    # a fingertip pixel at the image location of the (2, 7) press point
    px = apply_homography(H, np.array([g.cell_center(2, 7)]))[0]
    assert image_point_to_cell(px, H, g) == (2, 7)


def test_grid_line_segments_shapes():
    g = FretboardGeometry(scale_length=650.0, string_spacing=10.0, num_frets=22)
    frets = g.fret_line_segments()
    strings = g.string_line_segments()
    assert len(frets) == 23          # frets 0..22
    assert len(strings) == 6         # six strings
    # fret wires are vertical (constant x), strings are horizontal (constant y)
    assert all(abs(a[0] - b[0]) < 1e-9 for a, b in frets)
    assert all(abs(a[1] - b[1]) < 1e-9 for a, b in strings)
