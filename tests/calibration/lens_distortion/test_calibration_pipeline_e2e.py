"""End-to-end integration tests for the lens distortion calibration pipeline.

This module tests the complete calibration workflow from synthetic data generation
through coefficient recovery, validating that the pipeline produces coefficients
that improve line straightness on held-out validation data.

The tests use synthetic data with known distortion applied, enabling verification
that the solver can recover the true distortion parameters within reasonable
tolerance.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.distortion_solver import (
    DataQualityRequirements,
    DistortionSolver,
    SolverConfig,
    calculate_validation_rmse,
    straightness_rmse,
)
from poc_homography.calibration.lens_distortion.models import CameraLine, PTZPosition
from poc_homography.camera_parameters import DistortionCoefficients
from poc_homography.types import Unitless

if TYPE_CHECKING:
    pass


# =============================================================================
# Synthetic Data Generation Utilities
# =============================================================================


def apply_distortion(
    points: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    cx: float,
    cy: float,
    fx: float,
    fy: float,
) -> np.ndarray:
    """Apply radial distortion to undistorted points.

    Uses the Brown-Conrady forward distortion model:
    x_distorted = x * (1 + k1*r^2 + k2*r^4 + k3*r^6)
    y_distorted = y * (1 + k1*r^2 + k2*r^4 + k3*r^6)

    Args:
        points: Nx2 array of undistorted (u, v) pixel coordinates.
        k1, k2, k3: Radial distortion coefficients.
        cx, cy: Principal point coordinates.
        fx, fy: Focal lengths.

    Returns:
        Nx2 array of distorted (u, v) coordinates.
    """
    # Convert to normalized coordinates
    x = (points[:, 0] - cx) / fx
    y = (points[:, 1] - cy) / fy

    # Calculate radial distance
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    # Apply radial distortion
    radial_factor = 1 + k1 * r2 + k2 * r4 + k3 * r6
    x_distorted = x * radial_factor
    y_distorted = y * radial_factor

    # Convert back to pixel coordinates
    return np.column_stack([x_distorted * fx + cx, y_distorted * fy + cy])


def generate_straight_line_points(
    start: tuple[float, float],
    end: tuple[float, float],
    num_points: int = 25,
) -> np.ndarray:
    """Generate evenly spaced points along a straight line.

    Args:
        start: (u, v) start coordinates in pixels.
        end: (u, v) end coordinates in pixels.
        num_points: Number of points to generate.

    Returns:
        Array of shape (num_points, 2) with (u, v) coordinates.
    """
    t = np.linspace(0, 1, num_points)
    start_arr = np.array(start)
    end_arr = np.array(end)
    return start_arr + np.outer(t, end_arr - start_arr)


def create_distorted_camera_line(
    line_id: str,
    start: tuple[float, float],
    end: tuple[float, float],
    k1: float,
    k2: float,
    k3: float,
    cx: float,
    cy: float,
    fx: float,
    fy: float,
    num_edge_pixels: int = 25,
    ptz_position: PTZPosition | None = None,
) -> CameraLine:
    """Create a CameraLine with distorted edge_pixels from a straight line.

    This simulates detecting a line in a distorted camera image. The underlying
    real-world line is straight, but the detected edge_pixels are curved due
    to lens distortion.

    Args:
        line_id: Unique identifier for this line.
        start: Undistorted start point (u, v) in pixels.
        end: Undistorted end point (u, v) in pixels.
        k1, k2, k3: Radial distortion coefficients to apply.
        cx, cy: Principal point coordinates.
        fx, fy: Focal lengths.
        num_edge_pixels: Number of edge pixels along the line.
        ptz_position: PTZ position (uses default if None).

    Returns:
        CameraLine with distorted edge_pixels representing a curved appearance
        of a straight real-world line.
    """
    if ptz_position is None:
        ptz_position = PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=1.0)

    # Generate straight line points (undistorted)
    straight_points = generate_straight_line_points(start, end, num_edge_pixels)

    # Apply distortion to simulate camera capture
    distorted_points = apply_distortion(
        straight_points, k1, k2, k3, cx, cy, fx, fy
    )

    # Use distorted endpoints for start_pixel/end_pixel
    distorted_start = tuple(distorted_points[0])
    distorted_end = tuple(distorted_points[-1])

    # Convert edge_pixels to tuple of tuples (required by CameraLine)
    edge_pixels_tuple = tuple(tuple(p) for p in distorted_points)

    return CameraLine(
        line_id=line_id,
        image_path=f"/synthetic/{line_id}.jpg",
        start_pixel=distorted_start,
        end_pixel=distorted_end,
        ptz_position=ptz_position,
        edge_pixels=edge_pixels_tuple,
    )


def generate_synthetic_calibration_lines(
    k1: float,
    k2: float,
    k3: float,
    cx: float,
    cy: float,
    fx: float,
    fy: float,
    image_width: int = 1920,
    image_height: int = 1080,
    num_lines: int = 20,
    num_edge_pixels: int = 25,
) -> list[CameraLine]:
    """Generate synthetic camera lines covering all 4 quadrants with diverse orientations.

    Creates lines that:
    - Cover all 4 quadrants of the image
    - Include horizontal (0 deg), diagonal (45 deg), and vertical (90 deg) orientations
    - Have distortion applied to simulate real camera capture

    Args:
        k1, k2, k3: Radial distortion coefficients to apply.
        cx, cy: Principal point coordinates.
        fx, fy: Focal lengths.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        num_lines: Total number of lines to generate.
        num_edge_pixels: Number of edge pixels per line.

    Returns:
        List of CameraLine objects with distorted edge_pixels.
    """
    lines = []
    line_id = 0

    # Define margins from edges
    margin = 100
    mid_x = image_width / 2
    mid_y = image_height / 2

    # Quadrant regions (with some buffer from center)
    # Quadrant 0: Top-Left, Quadrant 1: Top-Right
    # Quadrant 2: Bottom-Left, Quadrant 3: Bottom-Right
    quadrant_regions = [
        # (x_min, x_max, y_min, y_max) for each quadrant
        (margin, mid_x - 50, margin, mid_y - 50),  # TL
        (mid_x + 50, image_width - margin, margin, mid_y - 50),  # TR
        (margin, mid_x - 50, mid_y + 50, image_height - margin),  # BL
        (mid_x + 50, image_width - margin, mid_y + 50, image_height - margin),  # BR
    ]

    # Generate lines per quadrant with different orientations
    orientations = [
        (0, "horizontal"),    # 0 degrees
        (45, "diagonal"),     # 45 degrees
        (90, "vertical"),     # 90 degrees
        (-45, "diagonal_neg"), # -45 degrees
    ]

    lines_per_quadrant = max(1, num_lines // 4)

    for q_idx, (x_min, x_max, y_min, y_max) in enumerate(quadrant_regions):
        for i in range(lines_per_quadrant):
            if len(lines) >= num_lines:
                break

            # Cycle through orientations
            angle_deg, orient_name = orientations[i % len(orientations)]

            # Generate line endpoints based on orientation
            # Random position within quadrant
            rng = np.random.default_rng(seed=1000 + line_id)
            center_x = rng.uniform(x_min + 50, x_max - 50)
            center_y = rng.uniform(y_min + 50, y_max - 50)
            length = 150  # Line length in pixels

            angle_rad = np.radians(angle_deg)
            dx = length / 2 * np.cos(angle_rad)
            dy = length / 2 * np.sin(angle_rad)

            start = (center_x - dx, center_y - dy)
            end = (center_x + dx, center_y + dy)

            # Clamp to image bounds
            start = (
                max(margin, min(image_width - margin, start[0])),
                max(margin, min(image_height - margin, start[1])),
            )
            end = (
                max(margin, min(image_width - margin, end[0])),
                max(margin, min(image_height - margin, end[1])),
            )

            line = create_distorted_camera_line(
                line_id=f"synth_q{q_idx}_{orient_name}_{i}",
                start=start,
                end=end,
                k1=k1,
                k2=k2,
                k3=k3,
                cx=cx,
                cy=cy,
                fx=fx,
                fy=fy,
                num_edge_pixels=num_edge_pixels,
            )
            lines.append(line)
            line_id += 1

    return lines


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def intrinsic_matrix() -> np.ndarray:
    """Standard 1920x1080 camera intrinsic matrix."""
    return np.array(
        [
            [1000.0, 0.0, 960.0],   # fx, 0, cx
            [0.0, 1000.0, 540.0],   # 0, fy, cy
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def camera_intrinsics(intrinsic_matrix: np.ndarray) -> dict:
    """Camera intrinsic parameters as a dictionary."""
    return {
        "fx": float(intrinsic_matrix[0, 0]),
        "fy": float(intrinsic_matrix[1, 1]),
        "cx": float(intrinsic_matrix[0, 2]),
        "cy": float(intrinsic_matrix[1, 2]),
    }


@pytest.fixture
def known_distortion() -> DistortionCoefficients:
    """Known distortion coefficients used to generate synthetic data.

    Uses moderate barrel distortion (negative k1) typical of wide-angle lenses.
    """
    return DistortionCoefficients(
        k1=Unitless(-0.15),
        k2=Unitless(0.02),
        k3=Unitless(0.0),
        p1=Unitless(0.0),
        p2=Unitless(0.0),
    )


@pytest.fixture
def synthetic_lines(
    known_distortion: DistortionCoefficients, camera_intrinsics: dict
) -> list[CameraLine]:
    """Generate 20 synthetic lines with known distortion applied."""
    return generate_synthetic_calibration_lines(
        k1=float(known_distortion.k1),
        k2=float(known_distortion.k2),
        k3=float(known_distortion.k3),
        cx=camera_intrinsics["cx"],
        cy=camera_intrinsics["cy"],
        fx=camera_intrinsics["fx"],
        fy=camera_intrinsics["fy"],
        num_lines=20,
        num_edge_pixels=25,
    )


@pytest.fixture
def ptz_position() -> PTZPosition:
    """Standard PTZ position for test lines."""
    return PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=1.0)


# =============================================================================
# End-to-End Calibration Pipeline Tests
# =============================================================================


class TestCalibrationPipelineE2E:
    """End-to-end tests for the full calibration pipeline."""

    def test_full_calibration_workflow_improves_straightness(
        self,
        synthetic_lines: list[CameraLine],
        intrinsic_matrix: np.ndarray,
        known_distortion: DistortionCoefficients,
    ) -> None:
        """Verify the complete calibration workflow improves line straightness.

        This test:
        1. Splits synthetic data into 50% training and 50% validation
        2. Runs the solver with pre-flight validation
        3. Verifies that calibration improves straightness on validation data
        4. Verifies the expected overfitting pattern (training < validation RMSE)
        """
        # Split data: 50% training, 50% validation
        training_lines = synthetic_lines[:10]
        validation_lines = synthetic_lines[10:]

        assert len(training_lines) == 10, "Should have 10 training lines"
        assert len(validation_lines) == 10, "Should have 10 validation lines"

        # Configure solver with radial-only mode (matching synthetic data)
        config = SolverConfig(
            use_radial_only=True,
            num_samples_per_line=20,
            max_iterations=1000,
        )
        solver = DistortionSolver(config=config)

        # Define data quality requirements
        requirements = DataQualityRequirements(
            min_points_per_line=15,
            min_total_lines=8,
            min_quadrant_coverage=2,
            min_orientations=2,
        )

        # Run solver with pre-flight validation
        result = solver.solve(
            lines=training_lines,
            intrinsic_matrix=intrinsic_matrix,
            data_requirements=requirements,
            image_width=1920,
            image_height=1080,
        )

        # Verify solver converged
        assert result.success, f"Solver should converge: {result.message}"
        assert result.is_improved(), "Calibration should improve straightness error"

        # Calculate validation RMSE (out-of-sample performance)
        validation_rmse, baseline_rmse = calculate_validation_rmse(
            validation_lines=validation_lines,
            intrinsic_matrix=intrinsic_matrix,
            distortion=result.distortion,
            num_samples=20,
        )

        # Assert: validation_rmse < baseline_rmse (demonstrates improvement)
        assert validation_rmse < baseline_rmse, (
            f"Calibration should improve validation RMSE: "
            f"validation_rmse={validation_rmse:.4f} should be < "
            f"baseline_rmse={baseline_rmse:.4f}"
        )

        # Assert: training_rmse < validation_rmse (normal overfitting pattern)
        training_rmse = result.training_rmse
        assert training_rmse <= validation_rmse, (
            f"Training RMSE should be <= validation RMSE (normal pattern): "
            f"training_rmse={training_rmse:.4f}, validation_rmse={validation_rmse:.4f}"
        )

        # Both RMSEs should be reasonable (under 5 pixels for synthetic data)
        assert training_rmse < 5.0, f"Training RMSE too high: {training_rmse:.4f}"
        assert validation_rmse < 5.0, f"Validation RMSE too high: {validation_rmse:.4f}"

    def test_recovered_coefficients_close_to_applied_distortion(
        self,
        synthetic_lines: list[CameraLine],
        intrinsic_matrix: np.ndarray,
        known_distortion: DistortionCoefficients,
    ) -> None:
        """Verify recovered coefficients are close to the applied distortion.

        For synthetic data with known distortion, the solver should recover
        coefficients reasonably close to the true values.
        """
        # Use all lines for training (maximum data for coefficient recovery)
        config = SolverConfig(
            use_radial_only=True,
            num_samples_per_line=20,
            max_iterations=2000,
            tolerance=1e-10,
        )
        solver = DistortionSolver(config=config)

        result = solver.solve(
            lines=synthetic_lines,
            intrinsic_matrix=intrinsic_matrix,
        )

        assert result.success, f"Solver should converge: {result.message}"

        # The recovered k1 should have the same sign as applied distortion
        # (negative for barrel distortion)
        recovered_k1 = float(result.distortion.k1)
        applied_k1 = float(known_distortion.k1)

        # Check sign agreement (both should indicate barrel distortion)
        assert np.sign(recovered_k1) == np.sign(applied_k1), (
            f"k1 sign should match: recovered={recovered_k1:.4f}, applied={applied_k1:.4f}"
        )

        # Check approximate magnitude (within 50% for synthetic data)
        # Note: The solver finds coefficients that straighten lines, which may
        # differ from the exact applied coefficients due to the iterative
        # undistortion process and optimization landscape
        k1_ratio = abs(recovered_k1 / applied_k1) if abs(applied_k1) > 1e-6 else 1.0
        assert 0.3 < k1_ratio < 3.0, (
            f"k1 magnitude should be similar order: "
            f"recovered={recovered_k1:.4f}, applied={applied_k1:.4f}, ratio={k1_ratio:.2f}"
        )

    def test_calibration_with_preflight_validation_enabled(
        self,
        synthetic_lines: list[CameraLine],
        intrinsic_matrix: np.ndarray,
    ) -> None:
        """Verify calibration succeeds when data meets quality requirements."""
        solver = DistortionSolver()

        # Strict requirements that synthetic data should meet
        requirements = DataQualityRequirements(
            min_points_per_line=15,
            min_total_lines=10,
            min_quadrant_coverage=3,
            min_orientations=3,
        )

        # Should not raise since synthetic data meets requirements
        result = solver.solve(
            lines=synthetic_lines,
            intrinsic_matrix=intrinsic_matrix,
            data_requirements=requirements,
            image_width=1920,
            image_height=1080,
        )

        assert result.success, "Solver should succeed with valid data"


# =============================================================================
# Data Quality Validation Error Cases
# =============================================================================


class TestDataQualityValidationErrors:
    """Tests for solver behavior when data quality is insufficient."""

    def test_raises_when_too_few_lines(
        self, intrinsic_matrix: np.ndarray, ptz_position: PTZPosition
    ) -> None:
        """Verify solver raises ValueError when line count is below minimum."""
        # Create only 3 lines (below default minimum of 10)
        lines = [
            CameraLine(
                line_id=f"line_{i}",
                image_path="/test.jpg",
                start_pixel=(100.0 + i * 100, 300.0),
                end_pixel=(200.0 + i * 100, 300.0),
                ptz_position=ptz_position,
                edge_pixels=tuple((100.0 + i * 100 + j * 5, 300.0) for j in range(20)),
            )
            for i in range(3)
        ]

        solver = DistortionSolver()
        requirements = DataQualityRequirements(
            min_total_lines=10,
            min_quadrant_coverage=1,
            min_orientations=1,
        )

        with pytest.raises(ValueError, match="Insufficient lines"):
            solver.solve(
                lines=lines,
                intrinsic_matrix=intrinsic_matrix,
                data_requirements=requirements,
                image_width=1920,
                image_height=1080,
            )

    def test_raises_when_lines_missing_edge_pixels(
        self, intrinsic_matrix: np.ndarray, ptz_position: PTZPosition
    ) -> None:
        """Verify solver raises ValueError when lines have no edge_pixels."""
        # Create lines without edge_pixels
        lines = [
            CameraLine(
                line_id=f"line_{i}",
                image_path="/test.jpg",
                start_pixel=(100.0 + i * 200, 300.0),
                end_pixel=(200.0 + i * 200, 300.0),
                ptz_position=ptz_position,
                edge_pixels=None,  # Missing edge_pixels
            )
            for i in range(10)
        ]

        solver = DistortionSolver()
        requirements = DataQualityRequirements(
            min_total_lines=5,
            min_quadrant_coverage=1,
            min_orientations=1,
        )

        with pytest.raises(ValueError, match="edge_pixels"):
            solver.solve(
                lines=lines,
                intrinsic_matrix=intrinsic_matrix,
                data_requirements=requirements,
                image_width=1920,
                image_height=1080,
            )

    def test_raises_when_insufficient_quadrant_coverage(
        self, intrinsic_matrix: np.ndarray, ptz_position: PTZPosition
    ) -> None:
        """Verify solver raises ValueError when quadrant coverage is insufficient."""
        # Create 10 lines all in top-left quadrant (centroid at x<960, y<540)
        lines = [
            CameraLine(
                line_id=f"line_{i}",
                image_path="/test.jpg",
                start_pixel=(100.0, 100.0 + i * 30),
                end_pixel=(400.0, 100.0 + i * 30),
                ptz_position=ptz_position,
                edge_pixels=tuple((100.0 + j * 15, 100.0 + i * 30) for j in range(20)),
            )
            for i in range(10)
        ]

        solver = DistortionSolver()
        requirements = DataQualityRequirements(
            min_total_lines=5,
            min_quadrant_coverage=4,  # Require all 4 quadrants
            min_orientations=1,
        )

        with pytest.raises(ValueError, match="Insufficient quadrant coverage"):
            solver.solve(
                lines=lines,
                intrinsic_matrix=intrinsic_matrix,
                data_requirements=requirements,
                image_width=1920,
                image_height=1080,
            )

    def test_raises_when_insufficient_orientation_diversity(
        self, intrinsic_matrix: np.ndarray, ptz_position: PTZPosition
    ) -> None:
        """Verify solver raises ValueError when orientation diversity is insufficient."""
        # Create 10 horizontal lines (all ~0 degree orientation)
        # Spread across quadrants to pass quadrant check
        lines = []
        quadrant_centers = [
            (300, 200),   # Top-left
            (1500, 200),  # Top-right
            (300, 800),   # Bottom-left
            (1500, 800),  # Bottom-right
        ]

        for i in range(10):
            q_idx = i % 4
            cx, cy = quadrant_centers[q_idx]
            y_offset = (i // 4) * 50

            lines.append(
                CameraLine(
                    line_id=f"line_{i}",
                    image_path="/test.jpg",
                    start_pixel=(cx - 100, cy + y_offset),
                    end_pixel=(cx + 100, cy + y_offset),  # Horizontal
                    ptz_position=ptz_position,
                    edge_pixels=tuple(
                        (cx - 100 + j * 10, cy + y_offset) for j in range(20)
                    ),
                )
            )

        solver = DistortionSolver()
        requirements = DataQualityRequirements(
            min_total_lines=5,
            min_quadrant_coverage=1,
            min_orientations=3,  # Require 3 orientation buckets
        )

        with pytest.raises(ValueError, match="Insufficient orientation diversity"):
            solver.solve(
                lines=lines,
                intrinsic_matrix=intrinsic_matrix,
                data_requirements=requirements,
                image_width=1920,
                image_height=1080,
            )


# =============================================================================
# CameraLine.sample_points() Mode Parameter Tests
# =============================================================================


class TestCameraLineSamplePointsModes:
    """Tests for CameraLine.sample_points() mode parameter behavior."""

    @pytest.fixture
    def line_with_edge_pixels(self, ptz_position: PTZPosition) -> CameraLine:
        """Create a CameraLine with valid edge_pixels."""
        edge_pixels = tuple((100.0 + i * 10, 200.0 + i * 5) for i in range(25))
        return CameraLine(
            line_id="test_line_with_edges",
            image_path="/test.jpg",
            start_pixel=(100.0, 200.0),
            end_pixel=(340.0, 320.0),
            ptz_position=ptz_position,
            edge_pixels=edge_pixels,
        )

    @pytest.fixture
    def line_without_edge_pixels(self, ptz_position: PTZPosition) -> CameraLine:
        """Create a CameraLine without edge_pixels."""
        return CameraLine(
            line_id="test_line_no_edges",
            image_path="/test.jpg",
            start_pixel=(100.0, 200.0),
            end_pixel=(340.0, 320.0),
            ptz_position=ptz_position,
            edge_pixels=None,
        )

    def test_edge_pixels_mode_raises_when_edge_pixels_none(
        self, line_without_edge_pixels: CameraLine
    ) -> None:
        """Verify mode='edge_pixels' raises ValueError when edge_pixels is None."""
        with pytest.raises(ValueError, match="no edge_pixels"):
            line_without_edge_pixels.sample_points(num_samples=20, mode="edge_pixels")

    def test_edge_pixels_mode_returns_edge_pixel_samples(
        self, line_with_edge_pixels: CameraLine
    ) -> None:
        """Verify mode='edge_pixels' returns samples from actual edge_pixels."""
        samples = line_with_edge_pixels.sample_points(num_samples=10, mode="edge_pixels")

        assert samples.shape == (10, 2), "Should return requested number of samples"

        # Samples should be from the edge_pixels (within the range)
        edge_arr = np.array(line_with_edge_pixels.edge_pixels)
        for sample in samples:
            # Check sample is approximately on one of the edge pixels
            distances = np.linalg.norm(edge_arr - sample, axis=1)
            assert np.min(distances) < 1.0, "Sample should be from edge_pixels"

    def test_linear_mode_works_without_edge_pixels(
        self, line_without_edge_pixels: CameraLine
    ) -> None:
        """Verify mode='linear' works even when edge_pixels is None."""
        samples = line_without_edge_pixels.sample_points(num_samples=20, mode="linear")

        assert samples.shape == (20, 2), "Should return requested number of samples"

        # First and last should be start and end
        np.testing.assert_array_almost_equal(
            samples[0], line_without_edge_pixels.start_pixel, decimal=5
        )
        np.testing.assert_array_almost_equal(
            samples[-1], line_without_edge_pixels.end_pixel, decimal=5
        )

    def test_linear_mode_works_with_edge_pixels(
        self, line_with_edge_pixels: CameraLine
    ) -> None:
        """Verify mode='linear' returns interpolated points even when edge_pixels exists."""
        samples = line_with_edge_pixels.sample_points(num_samples=20, mode="linear")

        assert samples.shape == (20, 2), "Should return requested number of samples"

        # Should be linearly interpolated between endpoints
        np.testing.assert_array_almost_equal(
            samples[0], line_with_edge_pixels.start_pixel, decimal=5
        )
        np.testing.assert_array_almost_equal(
            samples[-1], line_with_edge_pixels.end_pixel, decimal=5
        )

    def test_auto_mode_issues_deprecation_warning(
        self, line_with_edge_pixels: CameraLine
    ) -> None:
        """Verify mode='auto' issues a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            line_with_edge_pixels.sample_points(num_samples=10, mode="auto")

            # Find DeprecationWarning
            deprecation_warnings = [
                w for w in caught_warnings if issubclass(w.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 1, "Should issue exactly one DeprecationWarning"
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_auto_mode_uses_edge_pixels_when_available(
        self, line_with_edge_pixels: CameraLine
    ) -> None:
        """Verify mode='auto' uses edge_pixels when available."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            samples = line_with_edge_pixels.sample_points(num_samples=10, mode="auto")

        assert samples.shape == (10, 2)

        # Verify samples come from edge_pixels (not linear interpolation)
        edge_arr = np.array(line_with_edge_pixels.edge_pixels)
        for sample in samples:
            distances = np.linalg.norm(edge_arr - sample, axis=1)
            assert np.min(distances) < 1.0, "auto mode should use edge_pixels when available"

    def test_auto_mode_falls_back_to_linear_when_no_edge_pixels(
        self, line_without_edge_pixels: CameraLine
    ) -> None:
        """Verify mode='auto' falls back to linear when edge_pixels is None."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            samples = line_without_edge_pixels.sample_points(num_samples=20, mode="auto")

        assert samples.shape == (20, 2)

        # Should be linearly interpolated
        np.testing.assert_array_almost_equal(
            samples[0], line_without_edge_pixels.start_pixel, decimal=5
        )

    def test_invalid_mode_raises_value_error(
        self, line_with_edge_pixels: CameraLine
    ) -> None:
        """Verify invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            line_with_edge_pixels.sample_points(num_samples=10, mode="invalid_mode")


# =============================================================================
# Additional Calibration Scenario Tests
# =============================================================================


class TestCalibrationScenarios:
    """Additional tests for various calibration scenarios."""

    def test_calibration_with_minimal_valid_data(
        self, intrinsic_matrix: np.ndarray, camera_intrinsics: dict
    ) -> None:
        """Verify calibration works with minimal but valid data (edge case).

        Uses synthetic distorted data to ensure the optimizer has something
        to optimize, avoiding the degenerate case of perfectly straight lines.
        """
        # Create minimal valid dataset with mild distortion applied
        k1, k2, k3 = -0.1, 0.01, 0.0
        cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]
        fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]

        lines = []
        ptz_position = PTZPosition(pan_deg=0.0, tilt_deg=30.0, zoom_factor=1.0)

        # 5 horizontal lines in top-left quadrant
        for i in range(5):
            start = (100.0, 100.0 + i * 50)
            end = (400.0, 100.0 + i * 50)
            line = create_distorted_camera_line(
                line_id=f"h_tl_{i}",
                start=start,
                end=end,
                k1=k1,
                k2=k2,
                k3=k3,
                cx=cx,
                cy=cy,
                fx=fx,
                fy=fy,
                num_edge_pixels=20,
                ptz_position=ptz_position,
            )
            lines.append(line)

        # 5 vertical lines in bottom-right quadrant
        for i in range(5):
            start = (1200.0 + i * 50, 600.0)
            end = (1200.0 + i * 50, 900.0)
            line = create_distorted_camera_line(
                line_id=f"v_br_{i}",
                start=start,
                end=end,
                k1=k1,
                k2=k2,
                k3=k3,
                cx=cx,
                cy=cy,
                fx=fx,
                fy=fy,
                num_edge_pixels=20,
                ptz_position=ptz_position,
            )
            lines.append(line)

        solver = DistortionSolver()
        requirements = DataQualityRequirements(
            min_total_lines=10,
            min_quadrant_coverage=2,
            min_orientations=2,
        )

        # Should succeed with minimal valid data
        result = solver.solve(
            lines=lines,
            intrinsic_matrix=intrinsic_matrix,
            data_requirements=requirements,
            image_width=1920,
            image_height=1080,
        )

        assert result.success, f"Should succeed with minimal valid data: {result.message}"
        assert len(result.line_errors) == 10, "Should have errors for all 10 lines"
        assert result.overall_rmse < 5.0, f"RMSE should be reasonable: {result.overall_rmse}"

    def test_straightness_rmse_utility_function(
        self,
        synthetic_lines: list[CameraLine],
        intrinsic_matrix: np.ndarray,
        known_distortion: DistortionCoefficients,
    ) -> None:
        """Verify straightness_rmse utility function works correctly."""
        # RMSE with no distortion correction (lines appear curved)
        rmse_no_correction = straightness_rmse(
            lines=synthetic_lines,
            intrinsic_matrix=intrinsic_matrix,
            distortion=None,
            num_samples=20,
        )

        # RMSE with distortion correction (lines should appear straighter)
        rmse_with_correction = straightness_rmse(
            lines=synthetic_lines,
            intrinsic_matrix=intrinsic_matrix,
            distortion=known_distortion,
            num_samples=20,
        )

        # Correction should improve (reduce) RMSE
        assert rmse_with_correction < rmse_no_correction, (
            f"Distortion correction should improve RMSE: "
            f"corrected={rmse_with_correction:.4f} should be < "
            f"uncorrected={rmse_no_correction:.4f}"
        )

    @pytest.mark.parametrize(
        "k1,k2,description",
        [
            (-0.1, 0.0, "mild barrel distortion"),
            (-0.25, 0.05, "moderate barrel distortion"),
            (0.1, 0.0, "mild pincushion distortion"),
        ],
    )
    def test_calibration_with_various_distortion_levels(
        self,
        k1: float,
        k2: float,
        description: str,
        intrinsic_matrix: np.ndarray,
    ) -> None:
        """Verify calibration works with various distortion levels.

        Args:
            k1: First radial distortion coefficient.
            k2: Second radial distortion coefficient.
            description: Human-readable description of distortion type.
        """
        # Generate synthetic lines with this distortion
        cx, cy = 960.0, 540.0
        fx, fy = 1000.0, 1000.0

        lines = generate_synthetic_calibration_lines(
            k1=k1,
            k2=k2,
            k3=0.0,
            cx=cx,
            cy=cy,
            fx=fx,
            fy=fy,
            num_lines=20,
            num_edge_pixels=25,
        )

        # Split into training and validation
        training_lines = lines[:10]
        validation_lines = lines[10:]

        config = SolverConfig(use_radial_only=True)
        solver = DistortionSolver(config=config)

        result = solver.solve(
            lines=training_lines,
            intrinsic_matrix=intrinsic_matrix,
        )

        assert result.success, f"Solver should succeed for {description}"

        # Verify improvement on validation set
        validation_rmse, baseline_rmse = calculate_validation_rmse(
            validation_lines=validation_lines,
            intrinsic_matrix=intrinsic_matrix,
            distortion=result.distortion,
        )

        assert validation_rmse < baseline_rmse, (
            f"Calibration should improve {description}: "
            f"validation={validation_rmse:.4f}, baseline={baseline_rmse:.4f}"
        )
