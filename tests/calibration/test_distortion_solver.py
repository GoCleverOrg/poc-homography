"""Tests for distortion coefficient solver.

Tests the public API and behavior of the solver without knowledge
of internal implementation details.
"""

import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.distortion_solver import (
    DistortionSolver,
    SolverConfig,
    SolverResult,
    straightness_rmse,
)
from poc_homography.calibration.lens_distortion.models import CameraLine, PTZPosition
from poc_homography.camera_parameters import DistortionCoefficients
from poc_homography.types import Unitless


def make_edge_pixels(
    start: tuple[float, float], end: tuple[float, float], num_points: int = 20
) -> tuple[tuple[float, float], ...]:
    """Generate edge_pixels as linear interpolation between start and end.

    This is used for tests that need valid CameraLine objects with edge_pixels.
    The resulting points form a perfectly straight line.
    """
    t = np.linspace(0, 1, num_points)
    points = []
    for ti in t:
        x = start[0] + ti * (end[0] - start[0])
        y = start[1] + ti * (end[1] - start[1])
        points.append((x, y))
    return tuple(points)


class TestSolverConfig:
    """Tests for SolverConfig dataclass."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        config = SolverConfig()

        assert config.num_samples_per_line > 0
        assert config.max_iterations > 0
        assert config.tolerance > 0

    def test_get_bounds_full(self):
        """Should return bounds for all 5 coefficients by default."""
        config = SolverConfig()

        bounds = config.get_bounds()

        assert len(bounds) == 5  # k1, k2, k3, p1, p2

    def test_get_bounds_radial_only(self):
        """Should return bounds for 3 coefficients when radial_only."""
        config = SolverConfig(use_radial_only=True)

        bounds = config.get_bounds()

        assert len(bounds) == 3  # k1, k2, k3 only


class TestSolverResult:
    """Tests for SolverResult dataclass."""

    def test_is_improved_when_error_reduced(self):
        """Should return True when final error is less than initial."""
        result = SolverResult(
            distortion=DistortionCoefficients(),
            initial_error=1.0,
            final_error=0.5,
            rmse_per_line=[0.5],
            overall_rmse=0.5,
            iterations=10,
            success=True,
            message="Optimization terminated successfully.",
        )

        assert result.is_improved() is True

    def test_is_improved_false_when_error_increased(self):
        """Should return False when final error is greater than initial."""
        result = SolverResult(
            distortion=DistortionCoefficients(),
            initial_error=0.5,
            final_error=1.0,
            rmse_per_line=[1.0],
            overall_rmse=1.0,
            iterations=10,
            success=True,
            message="Optimization terminated successfully.",
        )

        assert result.is_improved() is False

    def test_improvement_ratio_half(self):
        """Should return 0.5 when error is halved."""
        result = SolverResult(
            distortion=DistortionCoefficients(),
            initial_error=1.0,
            final_error=0.5,
            rmse_per_line=[0.5],
            overall_rmse=0.5,
            iterations=10,
            success=True,
            message="Optimization terminated successfully.",
        )

        assert result.improvement_ratio() == 0.5

    def test_improvement_ratio_no_change(self):
        """Should return 1.0 when error unchanged."""
        result = SolverResult(
            distortion=DistortionCoefficients(),
            initial_error=1.0,
            final_error=1.0,
            rmse_per_line=[1.0],
            overall_rmse=1.0,
            iterations=10,
            success=True,
            message="Optimization terminated successfully.",
        )

        assert result.improvement_ratio() == 1.0

    def test_improvement_ratio_handles_zero_initial(self):
        """Should return 1.0 when initial error is zero (no change possible)."""
        result = SolverResult(
            distortion=DistortionCoefficients(),
            initial_error=0.0,
            final_error=0.0,
            rmse_per_line=[0.0],
            overall_rmse=0.0,
            iterations=10,
            success=True,
            message="Optimization terminated successfully.",
        )

        assert result.improvement_ratio() == 1.0


class TestDistortionSolver:
    """Tests for DistortionSolver."""

    @pytest.fixture
    def intrinsic_matrix(self):
        """Standard intrinsic matrix for tests."""
        return np.array(
            [
                [1000.0, 0.0, 960.0],  # fx, 0, cx
                [0.0, 1000.0, 540.0],  # 0, fy, cy
                [0.0, 0.0, 1.0],
            ]
        )

    @pytest.fixture
    def ptz_position(self):
        """Standard PTZ position for tests."""
        return PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=1.0)

    def test_solve_raises_with_no_lines(self, intrinsic_matrix):
        """Should raise ValueError when no lines provided."""
        solver = DistortionSolver()

        with pytest.raises(ValueError, match="At least one line required"):
            solver.solve([], intrinsic_matrix)

    def test_solve_raises_with_invalid_intrinsic_matrix(self, ptz_position):
        """Should raise ValueError for non-3x3 intrinsic matrix."""
        solver = DistortionSolver()
        start, end = (100.0, 200.0), (500.0, 200.0)
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=start,
            end_pixel=end,
            ptz_position=ptz_position,
            edge_pixels=make_edge_pixels(start, end),
        )
        invalid_matrix = np.array([[1, 0], [0, 1]])

        with pytest.raises(ValueError, match="Intrinsic matrix must be 3x3"):
            solver.solve([line], invalid_matrix)

    def test_solve_returns_result(self, intrinsic_matrix, ptz_position):
        """Should return SolverResult with valid structure."""
        solver = DistortionSolver()
        start1, end1 = (100.0, 300.0), (800.0, 300.0)
        start2, end2 = (100.0, 500.0), (800.0, 500.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start1,
                end_pixel=end1,  # Horizontal line
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start1, end1),
            ),
            CameraLine(
                line_id="line_2",
                image_path="/path/to/image.jpg",
                start_pixel=start2,
                end_pixel=end2,  # Another horizontal line
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start2, end2),
            ),
        ]

        result = solver.solve(lines, intrinsic_matrix)

        assert isinstance(result, SolverResult)
        assert isinstance(result.distortion, DistortionCoefficients)
        assert result.iterations >= 0
        assert isinstance(result.success, bool)
        assert len(result.rmse_per_line) == 2
        assert result.overall_rmse >= 0

    def test_solve_returns_rmse_per_line(self, intrinsic_matrix, ptz_position):
        """Should return RMSE for each input line."""
        solver = DistortionSolver()
        start1, end1 = (100.0, 300.0), (800.0, 300.0)
        start2, end2 = (100.0, 500.0), (800.0, 500.0)
        start3, end3 = (100.0, 700.0), (800.0, 700.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start1,
                end_pixel=end1,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start1, end1),
            ),
            CameraLine(
                line_id="line_2",
                image_path="/path/to/image.jpg",
                start_pixel=start2,
                end_pixel=end2,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start2, end2),
            ),
            CameraLine(
                line_id="line_3",
                image_path="/path/to/image.jpg",
                start_pixel=start3,
                end_pixel=end3,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start3, end3),
            ),
        ]

        result = solver.solve(lines, intrinsic_matrix)

        assert len(result.rmse_per_line) == 3
        assert all(rmse >= 0 for rmse in result.rmse_per_line)

    def test_solve_returns_line_errors_with_ids(self, intrinsic_matrix, ptz_position):
        """Should return line errors with line IDs."""
        solver = DistortionSolver()
        start1, end1 = (100.0, 300.0), (800.0, 300.0)
        start2, end2 = (100.0, 500.0), (800.0, 500.0)
        lines = [
            CameraLine(
                line_id="my_line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start1,
                end_pixel=end1,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start1, end1),
            ),
            CameraLine(
                line_id="my_line_2",
                image_path="/path/to/image.jpg",
                start_pixel=start2,
                end_pixel=end2,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start2, end2),
            ),
        ]

        result = solver.solve(lines, intrinsic_matrix)

        assert len(result.line_errors) == 2
        assert result.line_errors[0]["line_id"] == "my_line_1"
        assert result.line_errors[1]["line_id"] == "my_line_2"
        assert "rmse_pixels" in result.line_errors[0]
        assert "num_samples" in result.line_errors[0]

    def test_solve_with_initial_guess(self, intrinsic_matrix, ptz_position):
        """Should accept initial guess for coefficients."""
        solver = DistortionSolver()
        start, end = (100.0, 300.0), (800.0, 300.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start,
                end_pixel=end,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start, end),
            ),
        ]
        initial = DistortionCoefficients(
            k1=Unitless(-0.1),
            k2=Unitless(0.01),
            k3=Unitless(0.0),
            p1=Unitless(0.0),
            p2=Unitless(0.0),
        )

        result = solver.solve(lines, intrinsic_matrix, initial_guess=initial)

        assert isinstance(result, SolverResult)

    def test_solve_with_radial_only_config(self, intrinsic_matrix, ptz_position):
        """Should optimize only radial coefficients when configured."""
        config = SolverConfig(use_radial_only=True)
        solver = DistortionSolver(config=config)
        start, end = (100.0, 300.0), (800.0, 300.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start,
                end_pixel=end,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start, end),
            ),
        ]

        result = solver.solve(lines, intrinsic_matrix)

        # p1 and p2 should be zero when radial_only
        assert float(result.distortion.p1) == 0.0
        assert float(result.distortion.p2) == 0.0

    def test_solve_perfectly_straight_lines_have_low_error(
        self, intrinsic_matrix, ptz_position
    ):
        """Perfectly straight lines should have very low straightness error."""
        solver = DistortionSolver()
        # Create perfectly horizontal lines (no distortion)
        start1, end1 = (100.0, 300.0), (1800.0, 300.0)
        start2, end2 = (100.0, 600.0), (1800.0, 600.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start1,
                end_pixel=end1,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start1, end1),
            ),
            CameraLine(
                line_id="line_2",
                image_path="/path/to/image.jpg",
                start_pixel=start2,
                end_pixel=end2,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start2, end2),
            ),
        ]

        result = solver.solve(lines, intrinsic_matrix)

        # With straight input lines and no distortion to correct,
        # the error should be very small
        assert result.overall_rmse < 1.0  # Less than 1 pixel

    def test_calculate_line_errors(self, intrinsic_matrix, ptz_position):
        """Should calculate errors for lines with given distortion."""
        solver = DistortionSolver()
        start, end = (100.0, 300.0), (800.0, 300.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start,
                end_pixel=end,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start, end),
            ),
        ]
        distortion = DistortionCoefficients()  # Zero distortion

        errors = solver.calculate_line_errors(lines, intrinsic_matrix, distortion)

        assert len(errors) == 1
        assert errors[0]["line_id"] == "line_1"
        assert "rmse_pixels" in errors[0]
        assert errors[0]["rmse_pixels"] >= 0


class TestStraightnessRmse:
    """Tests for straightness_rmse convenience function."""

    @pytest.fixture
    def intrinsic_matrix(self):
        """Standard intrinsic matrix for tests."""
        return np.array(
            [
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @pytest.fixture
    def ptz_position(self):
        """Standard PTZ position for tests."""
        return PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=1.0)

    def test_returns_rmse_value(self, intrinsic_matrix, ptz_position):
        """Should return a non-negative RMSE value."""
        start, end = (100.0, 300.0), (800.0, 300.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start,
                end_pixel=end,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start, end),
            ),
        ]

        rmse = straightness_rmse(lines, intrinsic_matrix)

        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_with_zero_distortion(self, intrinsic_matrix, ptz_position):
        """Should work with zero distortion coefficients."""
        start, end = (100.0, 300.0), (800.0, 300.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start,
                end_pixel=end,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start, end),
            ),
        ]
        zero_distortion = DistortionCoefficients()

        rmse = straightness_rmse(lines, intrinsic_matrix, distortion=zero_distortion)

        assert rmse >= 0

    def test_with_custom_distortion(self, intrinsic_matrix, ptz_position):
        """Should accept custom distortion coefficients."""
        start, end = (100.0, 300.0), (800.0, 300.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start,
                end_pixel=end,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start, end),
            ),
        ]
        distortion = DistortionCoefficients(
            k1=Unitless(-0.1),
            k2=Unitless(0.05),
            k3=Unitless(0.0),
            p1=Unitless(0.0),
            p2=Unitless(0.0),
        )

        rmse = straightness_rmse(lines, intrinsic_matrix, distortion=distortion)

        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_with_custom_num_samples(self, intrinsic_matrix, ptz_position):
        """Should accept custom number of samples."""
        start, end = (100.0, 300.0), (800.0, 300.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start,
                end_pixel=end,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start, end, num_points=50),
            ),
        ]

        rmse = straightness_rmse(lines, intrinsic_matrix, num_samples=50)

        assert rmse >= 0

    def test_straight_lines_have_low_rmse(self, intrinsic_matrix, ptz_position):
        """Perfectly straight lines should have very low RMSE."""
        start, end = (100.0, 300.0), (1800.0, 300.0)
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=start,
                end_pixel=end,
                ptz_position=ptz_position,
                edge_pixels=make_edge_pixels(start, end),
            ),
        ]

        rmse = straightness_rmse(lines, intrinsic_matrix)

        # Straight lines with no distortion should have near-zero RMSE
        assert rmse < 0.01  # Less than 0.01 pixels
