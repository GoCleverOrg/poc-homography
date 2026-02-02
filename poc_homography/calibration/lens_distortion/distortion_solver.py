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
        """Get bounds list for scipy optimizer.

        Returns bounds in OpenCV order [k1, k2, p1, p2, k3] to match
        DistortionCoefficients.to_array() / from_array().
        """
        if self.use_radial_only:
            return [self.k1_bounds, self.k2_bounds, self.k3_bounds]
        return [
            self.k1_bounds,
            self.k2_bounds,
            self.p1_bounds,
            self.p2_bounds,
            self.k3_bounds,
        ]


@dataclass
class SolverResult:
    """Result from distortion coefficient optimization.

    Attributes:
        distortion: Optimized distortion coefficients.
        initial_error: Total straightness error before optimization.
        final_error: Total straightness error after optimization.
        rmse_per_line: RMSE for each line (list of floats).
        overall_rmse: Overall RMSE across all lines.
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
    ) -> SolverResult:
        """Solve for distortion coefficients that straighten the given lines.

        Args:
            lines: List of detected camera lines to use for calibration.
            intrinsic_matrix: 3x3 camera intrinsic matrix K.
            initial_guess: Initial distortion coefficients. Uses zeros if None.

        Returns:
            SolverResult with optimized coefficients and error metrics.

        Raises:
            ValueError: If no lines provided or intrinsic matrix is invalid.
        """
        if not lines:
            raise ValueError("At least one line required for calibration")

        if intrinsic_matrix.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix must be 3x3, got {intrinsic_matrix.shape}")

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
        logger.info(f"Overall RMSE: {overall_rmse:.4f} pixels")
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
            coeffs: Distortion coefficients [k1, k2, p1, p2, k3] (OpenCV order) or [k1, k2, k3].
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
        # Extract coefficients in OpenCV order [k1, k2, p1, p2, k3]
        k1, k2 = coeffs[0], coeffs[1]
        if len(coeffs) >= 5:
            p1, p2, k3 = coeffs[2], coeffs[3], coeffs[4]
        else:
            # Radial-only mode: coeffs = [k1, k2, k3]
            k3 = coeffs[2]
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

        convergence_tol = 1e-10
        for _ in range(max_undistort_iterations):
            r2 = x_u * x_u + y_u * y_u
            r4 = r2 * r2
            r6 = r4 * r2

            # Radial distortion factor (clamped to prevent division by zero)
            radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
            radial = np.where(
                np.abs(radial) < min_radial_factor,
                np.where(radial >= 0, min_radial_factor, -min_radial_factor),
                radial,
            )

            # Tangential distortion
            dx_tangential = 2 * p1 * x_u * y_u + p2 * (r2 + 2 * x_u * x_u)
            dy_tangential = p1 * (r2 + 2 * y_u * y_u) + 2 * p2 * x_u * y_u

            # Update estimate
            x_u_new = (x - dx_tangential) / radial
            y_u_new = (y - dy_tangential) / radial

            # Early termination on convergence
            delta = np.max(np.abs(x_u_new - x_u)) + np.max(np.abs(y_u_new - y_u))
            x_u = x_u_new
            y_u = y_u_new
            if delta < convergence_tol:
                break

        # Guard against NaN/Inf from numerical instability
        nan_mask = np.isfinite(x_u) & np.isfinite(y_u)
        x_u = np.where(nan_mask, x_u, x)
        y_u = np.where(nan_mask, y_u, y)

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
        if len(points) < 3:
            # 2 points always fit a perfect line (zero residual), providing
            # no constraint to the optimizer while diluting RMSE.
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

        if not lines:
            return errors

        for line in lines:
            samples = line.sample_points(self.config.num_samples_per_line)
            if len(samples) == 0:
                continue
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
