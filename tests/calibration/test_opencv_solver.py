"""Tests for annotated-line distortion calibration solver.

Tests cover:
- Camera line annotation construction (N-point and fallback)
- Deterministic train/validation split
- Line-straightness solver with annotated lines
- Radial-only mode
- Edge cases (insufficient lines, empty input)
"""

from __future__ import annotations

import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.opencv_solver import (
    AnnotatedLineSolver,
    AnnotatedLineSolverConfig,
    CameraLineAnnotation,
    LineSplitResult,
    build_camera_line_annotations,
    split_lines,
)
from poc_homography.camera_parameters import DistortionCoefficients


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_camera_line_annotations_npoint() -> list[dict]:
    """Sample camera line annotations with N-point data."""
    return [
        {
            "line_id": "L1",
            "start_pixel_x": 100.0, "start_pixel_y": 200.0,
            "end_pixel_x": 500.0, "end_pixel_y": 600.0,
            "points": [
                [100.0, 200.0], [150.0, 260.0], [200.0, 320.0],
                [250.0, 380.0], [300.0, 440.0], [350.0, 500.0],
                [400.0, 555.0], [450.0, 580.0], [500.0, 600.0],
            ],
        },
        {
            "line_id": "L2",
            "start_pixel_x": 200.0, "start_pixel_y": 300.0,
            "end_pixel_x": 600.0, "end_pixel_y": 700.0,
            "points": [
                [200.0, 300.0], [280.0, 370.0], [360.0, 440.0],
                [440.0, 510.0], [520.0, 580.0], [600.0, 700.0],
            ],
        },
        {
            "line_id": "L3",
            "start_pixel_x": 608.0, "start_pixel_y": 169.5,
            "end_pixel_x": 1724.0, "end_pixel_y": 580.5,
            "points": [
                [608.0, 169.5], [750.0, 230.0], [900.0, 300.0],
                [1050.0, 370.0], [1200.0, 430.0], [1350.0, 490.0],
                [1500.0, 540.0], [1724.0, 580.5],
            ],
        },
        {
            "line_id": "L4",
            "start_pixel_x": 1874.0, "start_pixel_y": 533.5,
            "end_pixel_x": 714.0, "end_pixel_y": 160.5,
            "points": [
                [1874.0, 533.5], [1700.0, 480.0], [1500.0, 410.0],
                [1300.0, 340.0], [1100.0, 270.0], [900.0, 210.0],
                [714.0, 160.5],
            ],
        },
    ]


@pytest.fixture
def sample_camera_line_annotations_flat() -> list[dict]:
    """Sample camera line annotations with only start/end (no points array)."""
    return [
        {
            "line_id": "L1",
            "start_pixel_x": 100.0, "start_pixel_y": 200.0,
            "end_pixel_x": 500.0, "end_pixel_y": 600.0,
        },
        {
            "line_id": "L2",
            "start_pixel_x": 200.0, "start_pixel_y": 300.0,
            "end_pixel_x": 600.0, "end_pixel_y": 700.0,
        },
    ]


@pytest.fixture
def intrinsic_matrix() -> np.ndarray:
    """Sample 3x3 intrinsic matrix."""
    return np.array([
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0],
    ])


def _make_lines(
    n: int = 6,
    npoints: int = 10,
    spread: float = 500.0,
) -> list[CameraLineAnnotation]:
    """Create synthetic approximately-straight line annotations."""
    lines = []
    for j in range(n):
        pts = []
        for k in range(npoints):
            t = k / max(npoints - 1, 1)
            x = 200.0 + j * 60.0 + t * spread
            y = 150.0 + j * 40.0 + t * spread * 0.7
            pts.append((x, y))
        lines.append(CameraLineAnnotation(line_id=f"L{j}", points=pts))
    return lines


# ---------------------------------------------------------------------------
# Tests: Camera Line Annotation Construction
# ---------------------------------------------------------------------------


class TestBuildCameraLineAnnotations:
    """Tests for build_camera_line_annotations function."""

    def test_npoint_data_used(self, sample_camera_line_annotations_npoint):
        """N-point data is consumed when available."""
        result = build_camera_line_annotations(sample_camera_line_annotations_npoint)
        assert len(result) == 4
        l1 = next(a for a in result if a.line_id == "L1")
        assert len(l1.points) == 9

    def test_fallback_to_start_end(self, sample_camera_line_annotations_flat):
        """Falls back to [start, end] when no points array."""
        result = build_camera_line_annotations(sample_camera_line_annotations_flat)
        assert len(result) == 2
        l1 = next(a for a in result if a.line_id == "L1")
        assert len(l1.points) == 2
        assert l1.points[0] == pytest.approx((100.0, 200.0))
        assert l1.points[1] == pytest.approx((500.0, 600.0))

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert build_camera_line_annotations([]) == []

    def test_missing_coords_skipped(self):
        """Annotations without valid coordinates are skipped."""
        result = build_camera_line_annotations([{"line_id": "bad"}])
        assert len(result) == 0

    def test_points_are_float_tuples(self, sample_camera_line_annotations_npoint):
        """All points are (float, float) tuples."""
        result = build_camera_line_annotations(sample_camera_line_annotations_npoint)
        for ann in result:
            for pt in ann.points:
                assert isinstance(pt[0], float)
                assert isinstance(pt[1], float)


# ---------------------------------------------------------------------------
# Tests: Line Split
# ---------------------------------------------------------------------------


class TestSplitLines:
    """Tests for deterministic line splitting."""

    def _make_lines(self, n: int) -> list[CameraLineAnnotation]:
        return [
            CameraLineAnnotation(
                line_id=f"L{i}",
                points=[(0.0, 0.0), (50.0, 50.0), (100.0, 100.0)],
            )
            for i in range(n)
        ]

    def test_deterministic(self):
        """Same input produces same split."""
        lines = self._make_lines(20)
        split1 = split_lines(lines, 0.7)
        split2 = split_lines(lines, 0.7)
        assert [l.line_id for l in split1.training_lines] == [l.line_id for l in split2.training_lines]
        assert [l.line_id for l in split1.validation_lines] == [l.line_id for l in split2.validation_lines]

    def test_approximate_ratio(self):
        """Split ratio approximately respected for large N."""
        lines = self._make_lines(100)
        split = split_lines(lines, 0.7)
        total = len(split.training_lines) + len(split.validation_lines)
        assert total == 100
        train_ratio = len(split.training_lines) / total
        assert 0.55 <= train_ratio <= 0.85

    def test_at_least_one_each(self):
        """At least one line in each set when possible."""
        lines = self._make_lines(2)
        split = split_lines(lines, 0.9)
        assert len(split.training_lines) >= 1
        assert len(split.validation_lines) >= 1

    def test_empty_input(self):
        """Empty input returns empty split."""
        split = split_lines([], 0.7)
        assert split.training_lines == []
        assert split.validation_lines == []

    def test_all_lines_accounted_for(self):
        """No lines lost in split."""
        lines = self._make_lines(15)
        split = split_lines(lines, 0.7)
        all_ids = {l.line_id for l in lines}
        split_ids = {l.line_id for l in split.training_lines} | {l.line_id for l in split.validation_lines}
        assert all_ids == split_ids


# ---------------------------------------------------------------------------
# Tests: AnnotatedLineSolver
# ---------------------------------------------------------------------------


class TestAnnotatedLineSolver:
    """Core solver tests using line straightness only."""

    def test_basic_solve(self, intrinsic_matrix):
        """Solver converges on synthetic straight lines."""
        lines = _make_lines(n=6, npoints=10)
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, intrinsic_matrix)

        assert result.success
        assert isinstance(result.distortion, DistortionCoefficients)
        assert result.overall_rmse >= 0.0
        assert result.iterations >= 0

    def test_returns_solver_result_fields(self, intrinsic_matrix):
        """All SolverResult fields are populated."""
        lines = _make_lines(n=6, npoints=10)
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, intrinsic_matrix)

        assert isinstance(result.distortion, DistortionCoefficients)
        assert isinstance(result.initial_error, float)
        assert isinstance(result.final_error, float)
        assert isinstance(result.rmse_per_line, list)
        assert isinstance(result.overall_rmse, float)
        assert isinstance(result.iterations, int)
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.line_errors, list)

    def test_distortion_coefficients_format(self, intrinsic_matrix):
        """Distortion coefficients match Brown-Conrady model."""
        lines = _make_lines(n=6, npoints=10)
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, intrinsic_matrix)

        dc = result.distortion
        assert hasattr(dc, "k1")
        assert hasattr(dc, "k2")
        assert hasattr(dc, "k3")
        assert hasattr(dc, "p1")
        assert hasattr(dc, "p2")
        arr = dc.to_array()
        assert len(arr) == 5

    def test_intrinsics_returned(self, intrinsic_matrix):
        """Solver returns intrinsics dict from input matrix."""
        lines = _make_lines(n=6, npoints=10)
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, intrinsic_matrix)

        assert result.intrinsics is not None
        assert result.intrinsics["fx"] == pytest.approx(1000.0)
        assert result.intrinsics["fy"] == pytest.approx(1000.0)
        assert result.intrinsics["cx"] == pytest.approx(960.0)
        assert result.intrinsics["cy"] == pytest.approx(540.0)

    def test_line_errors_format(self, intrinsic_matrix):
        """Line errors contain line_id and rmse_pixels."""
        lines = _make_lines(n=6, npoints=10)
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, intrinsic_matrix)

        assert len(result.line_errors) > 0
        for err in result.line_errors:
            assert "line_id" in err
            assert "rmse_pixels" in err
            assert isinstance(err["rmse_pixels"], float)

    def test_npoint_annotations_used(
        self, sample_camera_line_annotations_npoint, intrinsic_matrix,
    ):
        """Solver works with build_camera_line_annotations output."""
        lines = build_camera_line_annotations(sample_camera_line_annotations_npoint)
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, intrinsic_matrix)

        assert result.success
        assert isinstance(result.distortion, DistortionCoefficients)

    def test_initial_guess_used(self, intrinsic_matrix):
        """Custom initial guess is accepted."""
        lines = _make_lines(n=6, npoints=10)
        guess = DistortionCoefficients(k1=-0.1, k2=0.01)
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, intrinsic_matrix, initial_guess=guess)

        assert result.success

    def test_radial_only_mode(self, intrinsic_matrix):
        """Radial-only mode sets p1=p2=0."""
        lines = _make_lines(n=6, npoints=10)
        config = AnnotatedLineSolverConfig(use_radial_only=True)
        solver = AnnotatedLineSolver(config=config)
        result = solver.solve(lines, intrinsic_matrix)

        assert result.success
        assert float(result.distortion.p1) == 0.0
        assert float(result.distortion.p2) == 0.0

    def test_train_validation_split(self, intrinsic_matrix):
        """Message reports training and validation line counts."""
        lines = _make_lines(n=10, npoints=8)
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, intrinsic_matrix)

        assert result.success
        assert "training" in result.message
        assert "validation" in result.message


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: insufficient lines, 2-point lines, empty input."""

    def test_no_lines_fails(self, intrinsic_matrix):
        """Empty line list fails gracefully."""
        solver = AnnotatedLineSolver()
        result = solver.solve([], intrinsic_matrix)
        assert not result.success

    def test_only_2point_lines_fails(self, intrinsic_matrix):
        """Lines with only 2 points (no straightness constraint) fails."""
        lines = [
            CameraLineAnnotation("L1", [(100.0, 200.0), (500.0, 600.0)]),
            CameraLineAnnotation("L2", [(200.0, 300.0), (600.0, 700.0)]),
        ]
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, intrinsic_matrix)
        assert not result.success
        assert "3 points" in result.message

    def test_missing_intrinsics_fails(self):
        """None intrinsics fails gracefully."""
        lines = _make_lines(n=4)
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, None)
        assert not result.success

    def test_single_line_works(self, intrinsic_matrix):
        """A single multi-point line is sufficient (split guarantees at least 1 training)."""
        lines = [
            CameraLineAnnotation(
                "L0",
                [(100.0 + i * 50, 200.0 + i * 30) for i in range(10)],
            ),
        ]
        solver = AnnotatedLineSolver()
        result = solver.solve(lines, intrinsic_matrix)
        # With only 1 line, split gives 1 training + 0 validation (or vice versa)
        # Either way, at least 1 training line exists.
        assert result.success
