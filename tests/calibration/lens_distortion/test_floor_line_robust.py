"""Synthetic tests for floor-line curation + robust distortion rejection.

Strategy: take straight world lines, apply a KNOWN radial distortion to make
curved ``edge_pixels`` across quadrants and orientations (the "floor" lines).
Then inject "car" lines -- off-family clutter and arcs whose curvature does NOT
fit that one distortion. We assert:

* curation keeps the floor families and drops obvious off-family lines;
* the robust solve recovers k1 close to the truth and REJECTS the car lines;
* the tight physical bounds prevent |k1| > 0.6 even on inconsistent input
  (no pegging at +/-1.0);
* too few inliers degrades gracefully (raises ValueError, not a crash).
"""

from __future__ import annotations

import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.distortion_solver import SolverConfig
from poc_homography.calibration.lens_distortion.floor_line_curation import (
    curate_floor_lines,
)
from poc_homography.calibration.lens_distortion.models import CameraLine, PTZPosition
from poc_homography.calibration.lens_distortion.robust_distortion import solve_robust
from poc_homography.domain.vo import LensDistortion

IMAGE_W, IMAGE_H = 1920, 1080
FX = FY = 1400.0
CX, CY = IMAGE_W / 2.0, IMAGE_H / 2.0
TRUE_K1 = -0.18

_PTZ = PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=1.0)


def _forward_distort(points: np.ndarray, k1: float, k2: float = 0.0) -> np.ndarray:
    """Brown-Conrady radial forward distortion (k1, k2 only)."""
    x = (points[:, 0] - CX) / FX
    y = (points[:, 1] - CY) / FY
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    return np.column_stack([x * radial * FX + CX, y * radial * FY + CY])


def _line(line_id: str, pts: np.ndarray) -> CameraLine:
    edge = tuple((float(p[0]), float(p[1])) for p in pts)
    return CameraLine(
        line_id=line_id,
        image_path="synthetic",
        start_pixel=edge[0],
        end_pixel=edge[-1],
        ptz_position=_PTZ,
        edge_pixels=edge,
    )


def _floor_lines(k1: float = TRUE_K1, n_per_axis: int = 6) -> list[CameraLine]:
    """Distorted straight floor lines across quadrants in two orientations."""
    lines: list[CameraLine] = []
    margin = 90.0
    for i, y in enumerate(np.linspace(150, IMAGE_H - 150, n_per_axis)):
        und = np.column_stack([np.linspace(margin, IMAGE_W - margin, 60), np.full(60, y)])
        lines.append(_line(f"h{i}", _forward_distort(und, k1)))
    for i, x in enumerate(np.linspace(220, IMAGE_W - 220, n_per_axis)):
        und = np.column_stack([np.full(60, x), np.linspace(margin, IMAGE_H - margin, 60)])
        lines.append(_line(f"v{i}", _forward_distort(und, k1)))
    return lines


def _car_lines() -> list[CameraLine]:
    """Off-family clutter + arcs inconsistent with a single radial distortion."""
    rng = np.random.default_rng(7)
    lines: list[CameraLine] = []

    # Diagonal off-family clutter at ~45 degrees (a car roof/windshield edge).
    for i in range(3):
        x0 = 300 + 200 * i
        und = np.column_stack([np.linspace(x0, x0 + 400, 40), np.linspace(300, 700, 40)])
        # Add an arbitrary bow that does NOT match the radial model.
        bow = 25.0 * np.sin(np.linspace(0, np.pi, 40))
        und[:, 1] += bow
        lines.append(_line(f"car_diag{i}", und))

    # Random arcs (3-D car edges) with curvature unrelated to the lens.
    for i in range(3):
        cx0, cy0 = rng.uniform(400, 1500), rng.uniform(300, 800)
        radius = rng.uniform(150, 350)
        ang = np.linspace(0.2, 1.4, 40)
        arc = np.column_stack([cx0 + radius * np.cos(ang), cy0 + radius * np.sin(ang)])
        arc += rng.normal(0, 1.0, arc.shape)
        lines.append(_line(f"car_arc{i}", arc))
    return lines


def _intrinsics() -> np.ndarray:
    return np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)


def _solver_config() -> SolverConfig:
    return SolverConfig(
        use_radial_only=True,
        optimize_intrinsics=False,
        k1_bounds=(-0.6, 0.3),
        k2_bounds=(-0.3, 0.3),
        k3_bounds=(-0.15, 0.15),
        num_samples_per_line=20,
        max_iterations=500,
    )


# ---------------------------------------------------------------------------
# (a) curation keeps the family, drops off-family
# ---------------------------------------------------------------------------


def test_curation_keeps_families_drops_off_family() -> None:
    floor = _floor_lines()
    # Inject strongly off-family diagonal clutter.
    clutter = _car_lines()[:3]  # the ~45-degree diagonals
    curated = curate_floor_lines(floor + clutter, IMAGE_W, IMAGE_H, angle_tol_deg=12.0, min_lines=6)
    kept_ids = {ln.line_id for ln in curated}
    # All horizontal/vertical floor families survive.
    assert sum(1 for ln in floor if ln.line_id in kept_ids) == len(floor)
    # The 45-degree diagonal clutter is dropped.
    assert not any(ln.line_id.startswith("car_diag") for ln in curated)


# ---------------------------------------------------------------------------
# (b) robust solve recovers k1 and rejects car lines
# ---------------------------------------------------------------------------


def test_robust_solve_recovers_k1_and_rejects_cars() -> None:
    floor = _floor_lines()
    cars = _car_lines()
    # Curate first (mirrors the production path), then robust-solve.
    curated = curate_floor_lines(floor + cars, IMAGE_W, IMAGE_H, min_lines=6)
    usable = [ln for ln in curated if ln.has_edge_curvature()]

    robust = solve_robust(
        usable,
        _intrinsics(),
        _solver_config(),
        initial_guess=LensDistortion(),
        min_lines=6,
    )

    assert abs(float(robust.result.distortion.k1) - TRUE_K1) < 0.05
    # Any car line that survived curation must be rejected by the robust loop.
    rejected_ids = {ln.line_id for ln in robust.rejected_lines}
    surviving_cars = [ln for ln in usable if ln.line_id.startswith("car")]
    for car in surviving_cars:
        assert car.line_id in rejected_ids, f"{car.line_id} not rejected"
    # The floor lines are kept.
    kept_ids = {ln.line_id for ln in robust.kept_lines}
    assert any(cid.startswith(("h", "v")) for cid in kept_ids)


# ---------------------------------------------------------------------------
# (c) bounds prevent pegging at +/-1.0 even on inconsistent input
# ---------------------------------------------------------------------------


def test_bounds_prevent_pegging_on_inconsistent_input() -> None:
    # Feed ONLY mutually-inconsistent car arcs: a runaway solve would peg at the
    # bound. With physical bounds it must stay within +/-0.6.
    cars = _car_lines()
    usable = [ln for ln in cars if ln.has_edge_curvature()]
    assert usable, "synthetic car arcs must carry curvature"
    config = _solver_config()
    from poc_homography.calibration.lens_distortion.distortion_solver import (
        DistortionSolver,
    )

    result = DistortionSolver(config).solve(usable, _intrinsics(), LensDistortion())
    assert abs(float(result.distortion.k1)) <= 0.6 + 1e-9
    assert abs(float(result.distortion.k1)) < 1.0  # never pegged at the old bound
    assert abs(float(result.distortion.k2)) <= 0.3 + 1e-9
    assert abs(float(result.distortion.k3)) <= 0.15 + 1e-9


# ---------------------------------------------------------------------------
# (d) too few inliers degrades gracefully
# ---------------------------------------------------------------------------


def test_robust_solve_too_few_lines_raises() -> None:
    floor = _floor_lines(n_per_axis=1)  # only 2 lines total
    usable = [ln for ln in floor if ln.has_edge_curvature()]
    with pytest.raises(ValueError, match="robust solve needs"):
        solve_robust(usable, _intrinsics(), _solver_config(), min_lines=6)
