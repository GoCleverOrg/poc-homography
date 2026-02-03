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

    def test_get_bounds_with_optimize_intrinsics(self):
        """Should append 4 intrinsic bounds when optimize_intrinsics=True."""
        config = SolverConfig(optimize_intrinsics=True)

        bounds = config.get_bounds(image_cx=960.0, image_cy=540.0)

        assert len(bounds) == 9  # 5 distortion + 4 intrinsics (fx, fy, cx, cy)

    def test_get_bounds_radial_only_with_intrinsics(self):
        """Should return 7 bounds when radial_only + optimize_intrinsics."""
        config = SolverConfig(use_radial_only=True, optimize_intrinsics=True)

        bounds = config.get_bounds(image_cx=960.0, image_cy=540.0)

        assert len(bounds) == 7  # 3 radial + 4 intrinsics

    def test_default_intrinsic_bounds(self):
        """Should have sensible default bounds for intrinsics."""
        config = SolverConfig()

        assert config.fx_bounds[0] > 0
        assert config.fx_bounds[1] > config.fx_bounds[0]
        assert config.fy_bounds[0] > 0
        # cx/cy bounds are None by default (auto-derived from image dimensions)
        assert config.cx_bounds is None
        assert config.cy_bounds is None

    def test_auto_derived_cx_cy_bounds(self):
        """Should auto-derive cx/cy bounds from image dimensions."""
        config = SolverConfig(optimize_intrinsics=True)

        bounds = config.get_bounds(image_cx=960.0, image_cy=540.0)
        # Last two bounds are cx and cy, auto-derived as (0, 2*cx) and (0, 2*cy)
        assert bounds[-2] == (0.0, 1920.0)
        assert bounds[-1] == (0.0, 1080.0)

    def test_explicit_cx_cy_bounds_override(self):
        """Explicit cx/cy bounds should override auto-derived ones."""
        config = SolverConfig(
            optimize_intrinsics=True,
            cx_bounds=(800.0, 1200.0),
            cy_bounds=(400.0, 700.0),
        )

        bounds = config.get_bounds(image_cx=960.0, image_cy=540.0)
        assert bounds[-2] == (800.0, 1200.0)
        assert bounds[-1] == (400.0, 700.0)


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
        line = CameraLine(
            line_id="line_1",
            image_path="/path/to/image.jpg",
            start_pixel=(100.0, 200.0),
            end_pixel=(500.0, 200.0),
            ptz_position=ptz_position,
        )
        invalid_matrix = np.array([[1, 0], [0, 1]])

        with pytest.raises(ValueError, match="Intrinsic matrix must be 3x3"):
            solver.solve([line], invalid_matrix)

    def test_solve_returns_result(self, intrinsic_matrix, ptz_position):
        """Should return SolverResult with valid structure."""
        solver = DistortionSolver()
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(800.0, 300.0),  # Horizontal line
                ptz_position=ptz_position,
            ),
            CameraLine(
                line_id="line_2",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 500.0),
                end_pixel=(800.0, 500.0),  # Another horizontal line
                ptz_position=ptz_position,
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
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(800.0, 300.0),
                ptz_position=ptz_position,
            ),
            CameraLine(
                line_id="line_2",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 500.0),
                end_pixel=(800.0, 500.0),
                ptz_position=ptz_position,
            ),
            CameraLine(
                line_id="line_3",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 700.0),
                end_pixel=(800.0, 700.0),
                ptz_position=ptz_position,
            ),
        ]

        result = solver.solve(lines, intrinsic_matrix)

        assert len(result.rmse_per_line) == 3
        assert all(rmse >= 0 for rmse in result.rmse_per_line)

    def test_solve_returns_line_errors_with_ids(self, intrinsic_matrix, ptz_position):
        """Should return line errors with line IDs."""
        solver = DistortionSolver()
        lines = [
            CameraLine(
                line_id="my_line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(800.0, 300.0),
                ptz_position=ptz_position,
            ),
            CameraLine(
                line_id="my_line_2",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 500.0),
                end_pixel=(800.0, 500.0),
                ptz_position=ptz_position,
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
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(800.0, 300.0),
                ptz_position=ptz_position,
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
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(800.0, 300.0),
                ptz_position=ptz_position,
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
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(1800.0, 300.0),
                ptz_position=ptz_position,
            ),
            CameraLine(
                line_id="line_2",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 600.0),
                end_pixel=(1800.0, 600.0),
                ptz_position=ptz_position,
            ),
        ]

        result = solver.solve(lines, intrinsic_matrix)

        # With straight input lines and no distortion to correct,
        # the error should be very small
        assert result.overall_rmse < 1.0  # Less than 1 pixel

    def test_calculate_line_errors(self, intrinsic_matrix, ptz_position):
        """Should calculate errors for lines with given distortion."""
        solver = DistortionSolver()
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(800.0, 300.0),
                ptz_position=ptz_position,
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
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(800.0, 300.0),
                ptz_position=ptz_position,
            ),
        ]

        rmse = straightness_rmse(lines, intrinsic_matrix)

        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_with_zero_distortion(self, intrinsic_matrix, ptz_position):
        """Should work with zero distortion coefficients."""
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(800.0, 300.0),
                ptz_position=ptz_position,
            ),
        ]
        zero_distortion = DistortionCoefficients()

        rmse = straightness_rmse(lines, intrinsic_matrix, distortion=zero_distortion)

        assert rmse >= 0

    def test_with_custom_distortion(self, intrinsic_matrix, ptz_position):
        """Should accept custom distortion coefficients."""
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(800.0, 300.0),
                ptz_position=ptz_position,
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
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(800.0, 300.0),
                ptz_position=ptz_position,
            ),
        ]

        rmse = straightness_rmse(lines, intrinsic_matrix, num_samples=50)

        assert rmse >= 0

    def test_straight_lines_have_low_rmse(self, intrinsic_matrix, ptz_position):
        """Perfectly straight lines should have very low RMSE."""
        lines = [
            CameraLine(
                line_id="line_1",
                image_path="/path/to/image.jpg",
                start_pixel=(100.0, 300.0),
                end_pixel=(1800.0, 300.0),
                ptz_position=ptz_position,
            ),
        ]

        rmse = straightness_rmse(lines, intrinsic_matrix)

        # Straight lines with no distortion should have near-zero RMSE
        assert rmse < 0.01  # Less than 0.01 pixels


class TestRoundTripDistortionRecovery:
    """Test that the solver can recover known distortion coefficients.

    Apply a known barrel distortion to straight lines, then run the solver
    and verify it recovers approximately correct k1.
    """

    @pytest.fixture
    def intrinsic_matrix(self):
        return np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0],
        ])

    @pytest.fixture
    def ptz_position(self):
        return PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=1.0)

    @staticmethod
    def _apply_forward_distortion(
        points: np.ndarray, k1: float, fx: float, fy: float, cx: float, cy: float
    ) -> np.ndarray:
        """Apply forward barrel distortion to undistorted points."""
        x = (points[:, 0] - cx) / fx
        y = (points[:, 1] - cy) / fy
        r2 = x * x + y * y
        radial = 1 + k1 * r2
        x_d = x * radial
        y_d = y * radial
        return np.column_stack([x_d * fx + cx, y_d * fy + cy])

    def test_recover_barrel_distortion_k1(self, intrinsic_matrix, ptz_position):
        """Solver should recover k1 from synthetically distorted lines."""
        true_k1 = -0.15
        fx, fy, cx, cy = 1000.0, 1000.0, 960.0, 540.0

        # Create straight lines across the image, then distort them
        lines = []
        for y_pos in [200, 400, 600, 800]:
            # Horizontal line
            undistorted_pts = np.column_stack([
                np.linspace(200, 1700, 30),
                np.full(30, float(y_pos)),
            ])
            distorted_pts = self._apply_forward_distortion(
                undistorted_pts, true_k1, fx, fy, cx, cy
            )
            edge_pixels = tuple((float(p[0]), float(p[1])) for p in distorted_pts)
            line = CameraLine(
                line_id=f"h_line_{y_pos}",
                image_path="synthetic",
                start_pixel=edge_pixels[0],
                end_pixel=edge_pixels[-1],
                ptz_position=ptz_position,
                edge_pixels=edge_pixels,
            )
            lines.append(line)

        for x_pos in [200, 600, 1400, 1700]:
            # Vertical line
            undistorted_pts = np.column_stack([
                np.full(30, float(x_pos)),
                np.linspace(100, 900, 30),
            ])
            distorted_pts = self._apply_forward_distortion(
                undistorted_pts, true_k1, fx, fy, cx, cy
            )
            edge_pixels = tuple((float(p[0]), float(p[1])) for p in distorted_pts)
            line = CameraLine(
                line_id=f"v_line_{x_pos}",
                image_path="synthetic",
                start_pixel=edge_pixels[0],
                end_pixel=edge_pixels[-1],
                ptz_position=ptz_position,
                edge_pixels=edge_pixels,
            )
            lines.append(line)

        config = SolverConfig(use_radial_only=True, num_samples_per_line=30)
        solver = DistortionSolver(config=config)
        result = solver.solve(lines, intrinsic_matrix)

        # The solver should recover k1 approximately
        recovered_k1 = float(result.distortion.k1)
        assert abs(recovered_k1 - true_k1) < 0.05, (
            f"Expected k1 ~{true_k1}, got {recovered_k1}"
        )
        # RMSE should be very low after correction
        assert result.overall_rmse < 1.0
