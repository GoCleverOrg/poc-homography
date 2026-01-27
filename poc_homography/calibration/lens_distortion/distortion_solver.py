"""Distortion coefficient solver using line straightness optimization.

This module implements the core calibration algorithm: finding distortion
coefficients that minimize the curvature of lines that should be straight.

Key insight: In an undistorted image, straight lines in the real world
appear straight. By measuring how curved detected parking lines are and
optimizing distortion coefficients to straighten them, we can determine
the camera's lens distortion.

The solver uses scipy.optimize.minimize with the L-BFGS-B algorithm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from poc_homography.camera_parameters import DistortionCoefficients
from poc_homography.types import Unitless

if TYPE_CHECKING:
    from poc_homography.calibration.lens_distortion.models import CameraLine

logger = logging.getLogger(__name__)


@dataclass
class DataQualityRequirements:
    """Configuration for data quality validation requirements.

    Attributes:
        min_points_per_line: Minimum edge_pixels per line required.
        min_total_lines: Minimum total number of lines required.
        min_quadrant_coverage: Minimum number of image quadrants that must
            have at least one line (based on line centroids).
        min_orientations: Minimum number of different line orientations
            required (angles binned into 30 degree buckets).
    """

    min_points_per_line: int = 15
    min_total_lines: int = 10
    min_quadrant_coverage: int = 2
    min_orientations: int = 2


@dataclass
class ValidationResult:
    """Result of calibration data quality validation.

    Attributes:
        is_valid: Whether the data meets all requirements.
        errors: List of critical errors that prevent calibration.
        warnings: List of non-critical issues that may affect quality.
        statistics: Dictionary containing detailed data quality statistics.
    """

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    statistics: dict


def validate_calibration_data(
    lines: list[CameraLine],
    requirements: DataQualityRequirements,
    image_width: int,
    image_height: int,
) -> ValidationResult:
    """Validate calibration data quality before running the solver.

    Performs comprehensive validation of the input lines to ensure sufficient
    data quality for reliable distortion coefficient estimation.

    Args:
        lines: List of detected camera lines to validate.
        requirements: Data quality requirements to check against.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        ValidationResult with validation status, errors, warnings, and statistics.
    """
    errors: list[str] = []
    warnings: list[str] = []
    statistics: dict = {}

    # Track per-line statistics
    points_per_line: list[int] = []
    lines_with_edge_pixels = 0
    lines_without_edge_pixels = 0

    # Track quadrant coverage (based on line centroids)
    # Quadrants: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    quadrant_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    mid_x = image_width / 2
    mid_y = image_height / 2

    # Track orientation diversity (30-degree buckets)
    # Angles range from -180 to 180, we bin into 12 buckets of 30 degrees each
    # Bucket 0: [-180, -150), Bucket 1: [-150, -120), ... Bucket 11: [150, 180)
    orientation_buckets: set[int] = set()
    line_angles: list[float] = []

    for line in lines:
        # Check edge_pixels availability
        if line.edge_pixels is None:
            lines_without_edge_pixels += 1
            errors.append(
                f"Line '{line.line_id}' has no edge_pixels (edge_pixels is None). "
                "Edge pixels are required for distortion calibration."
            )
            continue

        lines_with_edge_pixels += 1
        num_points = len(line.edge_pixels)
        points_per_line.append(num_points)

        # Check minimum points per line
        if num_points < requirements.min_points_per_line:
            warnings.append(
                f"Line '{line.line_id}' has only {num_points} edge_pixels "
                f"(minimum: {requirements.min_points_per_line})"
            )

        # Calculate centroid for quadrant coverage
        centroid_x = (line.start_pixel[0] + line.end_pixel[0]) / 2
        centroid_y = (line.start_pixel[1] + line.end_pixel[1]) / 2

        # Determine quadrant (0=TL, 1=TR, 2=BL, 3=BR)
        quadrant = 0
        if centroid_x >= mid_x:
            quadrant += 1
        if centroid_y >= mid_y:
            quadrant += 2
        quadrant_counts[quadrant] += 1

        # Calculate line angle and bin into 30-degree buckets
        angle = line.angle_degrees  # Returns angle in degrees
        line_angles.append(angle)
        # Normalize angle to [-180, 180) and bin into 30-degree buckets
        bucket = int((angle + 180) // 30) % 12
        orientation_buckets.add(bucket)

    # Calculate statistics
    num_quadrants_covered = sum(1 for count in quadrant_counts.values() if count > 0)
    num_orientations = len(orientation_buckets)

    statistics = {
        "total_lines": len(lines),
        "lines_with_edge_pixels": lines_with_edge_pixels,
        "lines_without_edge_pixels": lines_without_edge_pixels,
        "points_per_line": points_per_line,
        "min_points_per_line": min(points_per_line) if points_per_line else 0,
        "max_points_per_line": max(points_per_line) if points_per_line else 0,
        "avg_points_per_line": (
            sum(points_per_line) / len(points_per_line) if points_per_line else 0
        ),
        "quadrant_counts": quadrant_counts,
        "num_quadrants_covered": num_quadrants_covered,
        "orientation_buckets": sorted(orientation_buckets),
        "num_orientations": num_orientations,
        "line_angles": line_angles,
    }

    # Validate total line count
    if lines_with_edge_pixels < requirements.min_total_lines:
        errors.append(
            f"Insufficient lines with edge_pixels: {lines_with_edge_pixels} "
            f"(minimum: {requirements.min_total_lines})"
        )

    # Validate quadrant coverage
    if num_quadrants_covered < requirements.min_quadrant_coverage:
        quadrant_names = {0: "top-left", 1: "top-right", 2: "bottom-left", 3: "bottom-right"}
        covered = [quadrant_names[q] for q, c in quadrant_counts.items() if c > 0]
        errors.append(
            f"Insufficient quadrant coverage: {num_quadrants_covered} quadrants covered "
            f"(minimum: {requirements.min_quadrant_coverage}). "
            f"Covered quadrants: {covered}"
        )

    # Validate orientation diversity
    if num_orientations < requirements.min_orientations:
        errors.append(
            f"Insufficient orientation diversity: {num_orientations} orientation buckets "
            f"(minimum: {requirements.min_orientations}). "
            f"Lines should have varied angles for robust calibration."
        )

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        statistics=statistics,
    )


@dataclass
class SolverConfig:
    """Configuration for the distortion solver.

    Attributes:
        num_samples_per_line: Number of points to sample along each line.
        k1_bounds: (min, max) bounds for k1 coefficient.
        k2_bounds: (min, max) bounds for k2 coefficient.
        k3_bounds: (min, max) bounds for k3 coefficient.
        p1_bounds: (min, max) bounds for p1 coefficient.
        p2_bounds: (min, max) bounds for p2 coefficient.
        max_iterations: Maximum optimization iterations.
        tolerance: Convergence tolerance for optimizer.
        use_radial_only: If True, only optimize k1, k2, k3 (set p1, p2 = 0).
    """

    num_samples_per_line: int = 20
    k1_bounds: tuple[float, float] = (-1.0, 1.0)
    k2_bounds: tuple[float, float] = (-1.0, 1.0)
    k3_bounds: tuple[float, float] = (-1.0, 1.0)
    p1_bounds: tuple[float, float] = (-0.1, 0.1)
    p2_bounds: tuple[float, float] = (-0.1, 0.1)
    max_iterations: int = 1000
    tolerance: float = 1e-8
    use_radial_only: bool = False

    def get_bounds(self) -> list[tuple[float, float]]:
        """Get bounds list for scipy optimizer."""
        if self.use_radial_only:
            return [self.k1_bounds, self.k2_bounds, self.k3_bounds]
        return [
            self.k1_bounds,
            self.k2_bounds,
            self.k3_bounds,
            self.p1_bounds,
            self.p2_bounds,
        ]


@dataclass
class SolverResult:
    """Result from distortion coefficient optimization.

    Attributes:
        distortion: Optimized distortion coefficients.
        initial_error: Total straightness error before optimization.
        final_error: Total straightness error after optimization.
        rmse_per_line: RMSE for each line (list of floats).
        overall_rmse: Training RMSE (in-sample error calculated on lines used
            for optimization). This measures how well the model fits the
            training data, NOT generalization to unseen lines.
        iterations: Number of optimization iterations.
        success: Whether optimization converged successfully.
        message: Optimizer termination message.
        line_errors: Per-line error details.
    """

    distortion: DistortionCoefficients
    initial_error: float
    final_error: float
    rmse_per_line: list[float]
    overall_rmse: float
    iterations: int
    success: bool
    message: str
    line_errors: list[dict] = field(default_factory=list)

    @property
    def training_rmse(self) -> float:
        """Alias for overall_rmse (training/in-sample RMSE)."""
        return self.overall_rmse

    def is_improved(self) -> bool:
        """Check if optimization improved the error."""
        return self.final_error < self.initial_error

    def improvement_ratio(self) -> float:
        """Calculate ratio of improvement (1.0 = no change, 0.5 = 50% reduction)."""
        if self.initial_error == 0:
            return 1.0
        return self.final_error / self.initial_error


class DistortionSolver:
    """Solves for lens distortion coefficients using line straightness.

    The solver works by:
    1. Sampling points along each detected line
    2. Undistorting the points using candidate distortion coefficients
    3. Fitting a straight line to the undistorted points
    4. Measuring perpendicular distance of each point to the fitted line
    5. Optimizing coefficients to minimize total squared distance
    """

    def __init__(
        self,
        config: SolverConfig | None = None,
    ) -> None:
        """Initialize the solver.

        Args:
            config: Solver configuration. Uses defaults if None.
        """
        self.config = config or SolverConfig()

    def solve(
        self,
        lines: list[CameraLine],
        intrinsic_matrix: np.ndarray,
        initial_guess: DistortionCoefficients | None = None,
        data_requirements: DataQualityRequirements | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> SolverResult:
        """Solve for distortion coefficients that straighten the given lines.

        Args:
            lines: List of detected camera lines to use for calibration.
            intrinsic_matrix: 3x3 camera intrinsic matrix K.
            initial_guess: Initial distortion coefficients. Uses zeros if None.
            data_requirements: Data quality requirements for validation.
                If provided, validation is performed before optimization.
            image_width: Width of the image in pixels. Required if
                data_requirements is provided.
            image_height: Height of the image in pixels. Required if
                data_requirements is provided.

        Returns:
            SolverResult with optimized coefficients and error metrics.

        Raises:
            ValueError: If no lines provided, intrinsic matrix is invalid,
                or data quality validation fails.
        """
        if not lines:
            raise ValueError("At least one line required for calibration")

        if intrinsic_matrix.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix must be 3x3, got {intrinsic_matrix.shape}")

        # Perform data quality validation if requirements are provided
        if data_requirements is not None:
            if image_width is None or image_height is None:
                raise ValueError(
                    "image_width and image_height are required when "
                    "data_requirements is provided"
                )

            validation_result = validate_calibration_data(
                lines=lines,
                requirements=data_requirements,
                image_width=image_width,
                image_height=image_height,
            )

            # Log data quality statistics (INFO level)
            stats = validation_result.statistics
            logger.info(
                "Data quality statistics: "
                f"total_lines={stats['total_lines']}, "
                f"lines_with_edge_pixels={stats['lines_with_edge_pixels']}, "
                f"avg_points_per_line={stats['avg_points_per_line']:.1f}, "
                f"min_points={stats['min_points_per_line']}, "
                f"max_points={stats['max_points_per_line']}, "
                f"quadrants_covered={stats['num_quadrants_covered']}/4, "
                f"orientations={stats['num_orientations']}"
            )

            # Log quadrant distribution
            quadrant_names = {
                0: "top-left",
                1: "top-right",
                2: "bottom-left",
                3: "bottom-right",
            }
            quadrant_summary = ", ".join(
                f"{quadrant_names[q]}={c}"
                for q, c in stats["quadrant_counts"].items()
            )
            logger.info(f"Quadrant distribution: {quadrant_summary}")

            # Log warnings
            for warning in validation_result.warnings:
                logger.warning(warning)

            # Fail if validation errors
            if not validation_result.is_valid:
                error_summary = "; ".join(validation_result.errors)
                raise ValueError(
                    f"Calibration data quality validation failed: {error_summary}"
                )

        # Extract principal point from intrinsic matrix
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]

        # Prepare line samples
        line_samples = []
        for line in lines:
            samples = line.sample_points(self.config.num_samples_per_line)
            line_samples.append(samples)

        # Initial guess
        if initial_guess is None:
            initial_guess = DistortionCoefficients()

        if self.config.use_radial_only:
            x0 = np.array([initial_guess.k1, initial_guess.k2, initial_guess.k3])
        else:
            x0 = initial_guess.to_array()

        # Calculate initial error
        initial_error = self._total_straightness_error(x0, line_samples, cx, cy, fx, fy)
        logger.info(f"Initial straightness error: {initial_error:.6f}")

        # Optimize
        result = minimize(
            self._total_straightness_error,
            x0,
            args=(line_samples, cx, cy, fx, fy),
            method="L-BFGS-B",
            bounds=self.config.get_bounds(),
            options={
                "maxiter": self.config.max_iterations,
                "ftol": self.config.tolerance,
            },
        )

        # Extract optimized coefficients
        if self.config.use_radial_only:
            optimized = DistortionCoefficients(
                k1=Unitless(float(result.x[0])),
                k2=Unitless(float(result.x[1])),
                k3=Unitless(float(result.x[2])),
                p1=Unitless(0.0),
                p2=Unitless(0.0),
            )
        else:
            optimized = DistortionCoefficients.from_array(result.x)

        # Calculate per-line errors with optimized coefficients
        final_coeffs = result.x
        rmse_per_line = []
        line_errors = []

        for i, samples in enumerate(line_samples):
            undistorted = self._undistort_points(samples, final_coeffs, cx, cy, fx, fy)
            error = self._line_straightness_error(undistorted)
            num_samples = len(samples)
            rmse = np.sqrt(error / num_samples) if num_samples > 0 else 0.0
            rmse_per_line.append(float(rmse))

            line_errors.append(
                {
                    "line_id": lines[i].line_id,
                    "rmse_pixels": float(rmse),
                    "num_samples": num_samples,
                }
            )

        total_samples = sum(len(s) for s in line_samples)
        overall_rmse = np.sqrt(result.fun / total_samples) if total_samples > 0 else 0.0

        logger.info(f"Final straightness error: {result.fun:.6f}")
        logger.info(f"Training RMSE: {overall_rmse:.4f} pixels (in-sample, optimization result)")
        logger.info(
            f"Optimized k1={optimized.k1:.6f}, k2={optimized.k2:.6f}, k3={optimized.k3:.6f}"
        )

        return SolverResult(
            distortion=optimized,
            initial_error=initial_error,
            final_error=result.fun,
            rmse_per_line=rmse_per_line,
            overall_rmse=float(overall_rmse),
            iterations=result.nit,
            success=result.success,
            message=result.message,
            line_errors=line_errors,
        )

    def _total_straightness_error(
        self,
        coeffs: np.ndarray,
        line_samples: list[np.ndarray],
        cx: float,
        cy: float,
        fx: float,
        fy: float,
    ) -> float:
        """Calculate total straightness error for all lines.

        Args:
            coeffs: Distortion coefficients [k1, k2, k3, p1, p2] or [k1, k2, k3].
            line_samples: List of point arrays, one per line.
            cx, cy: Principal point coordinates.
            fx, fy: Focal lengths.

        Returns:
            Total sum of squared perpendicular distances.
        """
        total_error = 0.0

        for samples in line_samples:
            undistorted = self._undistort_points(samples, coeffs, cx, cy, fx, fy)
            error = self._line_straightness_error(undistorted)
            total_error += error

        return total_error

    def _undistort_points(
        self,
        points: np.ndarray,
        coeffs: np.ndarray,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        """Apply inverse distortion model to points.

        Uses the Brown-Conrady distortion model:
        x_distorted = x(1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
        y_distorted = y(1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y

        This method computes the inverse (undistortion).

        Args:
            points: Nx2 array of (u, v) pixel coordinates.
            coeffs: Distortion coefficients.
            cx, cy: Principal point coordinates.
            fx, fy: Focal lengths.

        Returns:
            Nx2 array of undistorted (u, v) coordinates.
        """
        # Extract coefficients
        k1, k2, k3 = coeffs[0], coeffs[1], coeffs[2]
        if len(coeffs) >= 5:
            p1, p2 = coeffs[3], coeffs[4]
        else:
            p1, p2 = 0.0, 0.0

        # Convert to normalized coordinates
        x = (points[:, 0] - cx) / fx
        y = (points[:, 1] - cy) / fy

        # Apply undistortion iteratively (Newton-Raphson approximation)
        # Start with distorted coordinates as initial guess
        x_u = x.copy()
        y_u = y.copy()

        # Newton-Raphson typically converges in 3-5 iterations for reasonable distortion
        max_undistort_iterations = 10
        # Minimum radial factor to prevent division by zero
        min_radial_factor = 1e-6

        for _ in range(max_undistort_iterations):
            r2 = x_u * x_u + y_u * y_u
            r4 = r2 * r2
            r6 = r4 * r2

            # Radial distortion factor (clamped to prevent division by zero)
            radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
            radial = np.maximum(np.abs(radial), min_radial_factor) * np.sign(radial + 1e-10)

            # Tangential distortion
            dx_tangential = 2 * p1 * x_u * y_u + p2 * (r2 + 2 * x_u * x_u)
            dy_tangential = p1 * (r2 + 2 * y_u * y_u) + 2 * p2 * x_u * y_u

            # Update estimate
            x_u = (x - dx_tangential) / radial
            y_u = (y - dy_tangential) / radial

        # Convert back to pixel coordinates
        undistorted = np.column_stack(
            [
                x_u * fx + cx,
                y_u * fy + cy,
            ]
        )

        return undistorted

    def _line_straightness_error(self, points: np.ndarray) -> float:
        """Calculate sum of squared perpendicular distances to best-fit line.

        Args:
            points: Nx2 array of (x, y) coordinates.

        Returns:
            Sum of squared perpendicular distances from points to fitted line.
        """
        if len(points) < 2:
            return 0.0

        # Fit line using SVD (total least squares)
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # SVD of centered points
        _, _, Vt = np.linalg.svd(centered)

        # Direction of the line is the first principal component
        line_direction = Vt[0]

        # Normal to the line
        line_normal = np.array([-line_direction[1], line_direction[0]])

        # Perpendicular distances
        distances = np.abs(centered @ line_normal)

        return float(np.sum(distances**2))

    def calculate_line_errors(
        self,
        lines: list[CameraLine],
        intrinsic_matrix: np.ndarray,
        distortion: DistortionCoefficients,
    ) -> list[dict]:
        """Calculate straightness errors for lines with given distortion.

        Useful for evaluating calibration quality without re-solving.

        Args:
            lines: List of camera lines.
            intrinsic_matrix: 3x3 camera intrinsic matrix.
            distortion: Distortion coefficients to evaluate.

        Returns:
            List of error dictionaries with line_id and rmse_pixels.
        """
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]

        coeffs = distortion.to_array()
        errors = []

        for line in lines:
            samples = line.sample_points(self.config.num_samples_per_line)
            undistorted = self._undistort_points(samples, coeffs, cx, cy, fx, fy)
            error = self._line_straightness_error(undistorted)
            rmse = np.sqrt(error / len(samples))

            errors.append(
                {
                    "line_id": line.line_id,
                    "rmse_pixels": float(rmse),
                    "num_samples": len(samples),
                }
            )

        return errors


def straightness_rmse(
    lines: list[CameraLine],
    intrinsic_matrix: np.ndarray,
    distortion: DistortionCoefficients | None = None,
    num_samples: int = 20,
) -> float:
    """Convenience function to calculate overall straightness RMSE.

    Args:
        lines: List of camera lines.
        intrinsic_matrix: 3x3 camera intrinsic matrix.
        distortion: Distortion coefficients (None = no distortion).
        num_samples: Number of samples per line.

    Returns:
        Overall RMSE in pixels.
    """
    if distortion is None:
        distortion = DistortionCoefficients()

    solver = DistortionSolver(SolverConfig(num_samples_per_line=num_samples))
    errors = solver.calculate_line_errors(lines, intrinsic_matrix, distortion)

    total_error = sum(e["rmse_pixels"] ** 2 * e["num_samples"] for e in errors)
    total_samples = sum(e["num_samples"] for e in errors)

    return float(np.sqrt(total_error / total_samples)) if total_samples > 0 else 0.0


def calculate_validation_rmse(
    validation_lines: list[CameraLine],
    intrinsic_matrix: np.ndarray,
    distortion: DistortionCoefficients,
    num_samples: int = 20,
) -> tuple[float, float]:
    """Calculate RMSE on hold-out validation lines.

    This should be called with lines NOT used during calibration to measure
    generalization (out-of-sample performance).

    Args:
        validation_lines: Lines not used during calibration (hold-out set).
        intrinsic_matrix: 3x3 camera intrinsic matrix.
        distortion: Distortion coefficients from calibration.
        num_samples: Points to sample per line.

    Returns:
        Tuple of (validation_rmse, baseline_rmse).
        - validation_rmse: RMSE with distortion correction applied
        - baseline_rmse: RMSE without any correction (to show improvement)
    """
    if not validation_lines:
        return 0.0, 0.0

    # Calculate baseline RMSE (no distortion correction)
    no_distortion = DistortionCoefficients()
    baseline_rmse = straightness_rmse(
        validation_lines, intrinsic_matrix, distortion=no_distortion, num_samples=num_samples
    )

    # Calculate validation RMSE (with distortion correction)
    validation_rmse = straightness_rmse(
        validation_lines, intrinsic_matrix, distortion=distortion, num_samples=num_samples
    )

    return validation_rmse, baseline_rmse
