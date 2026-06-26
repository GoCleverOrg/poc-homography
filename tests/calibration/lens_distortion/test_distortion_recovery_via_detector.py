"""End-to-end regression: detector -> fixed-K solve recovers barrel distortion.

Guards the root cause of issue #53: a previous detector emitted jagged
skeleton edge_pixels whose curvature was noise, so the solver pegged k1 at the
positive rail regardless of the true (barrel, negative) distortion. This test
renders straight floor lines, applies a KNOWN barrel distortion to the image,
and asserts the full repo pipeline recovers the correct negative k1 -- not a
peg. It must fail if the detector ever stops producing smooth centerlines.
"""

from __future__ import annotations

import cv2
import numpy as np

from poc_homography.calibration.lens_distortion.distortion_solver import SolverConfig
from poc_homography.calibration.lens_distortion.floor_line_curation import curate_floor_lines
from poc_homography.calibration.lens_distortion.models import PTZPosition
from poc_homography.calibration.lens_distortion.painted_line_detection import PaintedLineDetector
from poc_homography.calibration.lens_distortion.robust_distortion import solve_robust
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.types import Degrees

_W, _H = 1600, 1200
_FX = _FY = 1500.0
_CX, _CY = _W / 2.0, _H / 2.0
_K = np.array([[_FX, 0, _CX], [0, _FY, _CY], [0, 0, 1.0]])


def _distort(points: np.ndarray, k1: float, k2: float = 0.0) -> np.ndarray:
    """Apply the forward Brown-Conrady radial model to pixel points."""
    x = (points[:, 0] - _CX) / _FX
    y = (points[:, 1] - _CY) / _FY
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    return np.column_stack([x * radial * _FX + _CX, y * radial * _FY + _CY])


def _render_distorted_floor(k1: float) -> np.ndarray:
    """Draw straight floor lines, barrel-distort them, render thick strokes."""
    img = np.full((_H, _W, 3), 110, dtype=np.uint8)
    # Vertical-ish family (left) and horizontal-ish family (right), non-crossing.
    for x in range(200, _W // 2 - 100, 120):
        ideal = np.column_stack([np.full(120, x), np.linspace(80, _H - 80, 120)])
        pts = _distort(ideal, k1).astype(np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 7)
    for y in range(160, _H - 160, 130):
        ideal = np.column_stack([np.linspace(_W // 2 + 100, _W - 120, 120), np.full(120, y)])
        pts = _distort(ideal, k1).astype(np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 7)
    return img


def test_detector_solver_recovers_negative_k1() -> None:
    applied_k1 = -0.18
    img = _render_distorted_floor(applied_k1)

    ptz = PTZPosition(pan_deg=Degrees(0.0), tilt_deg=Degrees(0.0), zoom_factor=1.0)
    lines = [
        c.to_camera_line(line_id=f"l{i}", image_path="syn", ptz_position=ptz)
        for i, c in enumerate(PaintedLineDetector().detect(img))
    ]
    curated = curate_floor_lines(lines, float(_W), float(_H))
    usable = [line for line in curated if line.has_edge_curvature()]
    assert len(usable) >= 4, f"detector yielded too few usable lines: {len(usable)}"

    cfg = SolverConfig(
        use_radial_only=True,
        optimize_intrinsics=False,  # fixed K, the proven recipe
        k1_bounds=(-0.8, 0.8),
        k2_bounds=(-0.5, 0.5),
        k3_bounds=(-0.15, 0.15),
        num_samples_per_line=30,
        max_iterations=3000,
    )
    robust = solve_robust(usable, _K, cfg, initial_guess=LensDistortion(), min_lines=4)
    recovered = float(robust.result.distortion.k1)

    # Correct sign (barrel = negative) and not pegged at the +/- rail.
    assert recovered < 0.0, f"expected negative k1, got {recovered:+.4f} (regression: peg)"
    assert abs(recovered - applied_k1) < 0.08, f"k1={recovered:+.4f} far from {applied_k1}"
    assert abs(recovered) < 0.79, "k1 pegged at the bound"
