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

from poc_homography.calibration.lens_distortion.apply_calibration import (
    line_straightness_error as _line_straightness_error_impl,
    undistort_points as _undistort_points_impl,
)
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
    optimize_intrinsics: bool = False
    fx_bounds: tuple[float, float] = (500.0, 5000.0)
    fy_bounds: tuple[float, float] = (500.0, 5000.0)
    # Principal point bounds default to None (auto-derived from intrinsic_matrix
    # in solve()). Set explicitly to override.
    cx_bounds: tuple[float, float] | None = None
    cy_bounds: tuple[float, float] | None = None

    def get_bounds(
        self,
        image_cx: float | None = None,
        image_cy: float | None = None,
    ) -> list[tuple[float, float]]:
        """Get bounds list for scipy optimizer.

        Returns bounds in OpenCV order [k1, k2, p1, p2, k3] to match
        DistortionCoefficients.to_array() / from_array().
        When optimize_intrinsics is True, appends [fx, fy, cx, cy] bounds.

        Args:
            image_cx: Principal point X from intrinsic matrix, used to auto-derive
                cx_bounds when not explicitly set. Defaults to 960.0.
            image_cy: Principal point Y from intrinsic matrix, used to auto-derive
                cy_bounds when not explicitly set. Defaults to 540.0.
        """
        if self.use_radial_only:
            bounds = [self.k1_bounds, self.k2_bounds, self.k3_bounds]
        else:
            bounds = [
                self.k1_bounds,
                self.k2_bounds,
                self.p1_bounds,
                self.p2_bounds,
                self.k3_bounds,
            ]
        if self.optimize_intrinsics:
            cx = image_cx or 960.0
            cy = image_cy or 540.0
            cx_bounds = self.cx_bounds or (0.0, cx * 2)
            cy_bounds = self.cy_bounds or (0.0, cy * 2)
            bounds.extend(
                [
                    self.fx_bounds,
                    self.fy_bounds,
                    cx_bounds,
                    cy_bounds,
                ]
            )
        return bounds


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
    intrinsics: dict[str, float] | None = None

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

        # Filter out lines that are straight by construction (linearly
        # interpolated between endpoints).  These carry no distortion signal
        # and dilute the RMSE, leading to weak calibration results.
        original_count = len(lines)
        lines = [line for line in lines if line.has_edge_curvature()]
        filtered = original_count - len(lines)
        if filtered:
            logger.info(
                f"Filtered {filtered}/{original_count} lines with no edge "
                f"curvature (interpolated); {len(lines)} lines remain"
            )
        if not lines:
            raise ValueError(
                "No lines with edge curvature remain after filtering. "
                "All provided lines appear to be linearly interpolated "
                "between endpoints and carry no distortion signal."
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

        # When optimizing intrinsics, append [fx, fy, cx, cy] to the parameter vector
        optimize_intrinsics = self.config.optimize_intrinsics
        if optimize_intrinsics:
            x0 = np.append(x0, [fx, fy, cx, cy])

        # Calculate initial error
        initial_error = self._total_straightness_error(
            x0, line_samples, cx, cy, fx, fy, optimize_intrinsics
        )
        logger.info(f"Initial straightness error: {initial_error:.6f}")

        # Optimize
        result = minimize(
            self._total_straightness_error,
            x0,
            args=(line_samples, cx, cy, fx, fy, optimize_intrinsics),
            method="L-BFGS-B",
            bounds=self.config.get_bounds(image_cx=cx, image_cy=cy),
            options={
                "maxiter": self.config.max_iterations,
                "ftol": self.config.tolerance,
            },
        )

        # Extract optimized coefficients and intrinsics
        optimized_intrinsics = None
        if optimize_intrinsics:
            # Last 4 elements are [fx, fy, cx, cy]
            opt_fx, opt_fy, opt_cx, opt_cy = result.x[-4], result.x[-3], result.x[-2], result.x[-1]
            optimized_intrinsics = {
                "fx": float(opt_fx),
                "fy": float(opt_fy),
                "cx": float(opt_cx),
                "cy": float(opt_cy),
            }
            distortion_coeffs = result.x[:-4]
            # Use optimized intrinsics for error evaluation
            fx, fy, cx, cy = opt_fx, opt_fy, opt_cx, opt_cy
        else:
            distortion_coeffs = result.x

        if self.config.use_radial_only:
            optimized = DistortionCoefficients(
                k1=Unitless(float(distortion_coeffs[0])),
                k2=Unitless(float(distortion_coeffs[1])),
                k3=Unitless(float(distortion_coeffs[2])),
                p1=Unitless(0.0),
                p2=Unitless(0.0),
            )
        else:
            optimized = DistortionCoefficients.from_array(distortion_coeffs)

        # Calculate per-line errors with optimized coefficients
        rmse_per_line = []
        line_errors = []

        for i, samples in enumerate(line_samples):
            undistorted = self._undistort_points(samples, distortion_coeffs, cx, cy, fx, fy)
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
        if optimized_intrinsics:
            logger.info(
                f"Optimized intrinsics: fx={optimized_intrinsics['fx']:.1f}, "
                f"fy={optimized_intrinsics['fy']:.1f}, "
                f"cx={optimized_intrinsics['cx']:.1f}, "
                f"cy={optimized_intrinsics['cy']:.1f}"
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
            intrinsics=optimized_intrinsics,
        )

    def _total_straightness_error(
        self,
        coeffs: np.ndarray,
        line_samples: list[np.ndarray],
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        optimize_intrinsics: bool = False,
    ) -> float:
        """Calculate total straightness error for all lines.

        Args:
            coeffs: Distortion coefficients [k1, k2, p1, p2, k3] (OpenCV order) or [k1, k2, k3].
                When optimize_intrinsics is True, the last 4 elements are [fx, fy, cx, cy].
            line_samples: List of point arrays, one per line.
            cx, cy: Principal point coordinates (used when not optimizing intrinsics).
            fx, fy: Focal lengths (used when not optimizing intrinsics).
            optimize_intrinsics: If True, extract fx/fy/cx/cy from coeffs vector.

        Returns:
            Total sum of squared perpendicular distances.
        """
        if optimize_intrinsics:
            fx_opt, fy_opt, cx_opt, cy_opt = coeffs[-4], coeffs[-3], coeffs[-2], coeffs[-1]
            dist_coeffs = coeffs[:-4]
        else:
            fx_opt, fy_opt, cx_opt, cy_opt = fx, fy, cx, cy
            dist_coeffs = coeffs

        total_error = 0.0

        for samples in line_samples:
            undistorted = self._undistort_points(
                samples, dist_coeffs, cx_opt, cy_opt, fx_opt, fy_opt
            )
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

        Uses the Brown-Conrady distortion model via the shared utility function.

        Args:
            points: Nx2 array of (u, v) pixel coordinates.
            coeffs: Distortion coefficients in OpenCV order [k1, k2, p1, p2, k3]
                or radial-only [k1, k2, k3].
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

        return _undistort_points_impl(points, k1, k2, k3, p1, p2, fx, fy, cx, cy)

    def _line_straightness_error(self, points: np.ndarray) -> float:
        """Calculate sum of squared perpendicular distances to best-fit line.

        Uses the shared utility function for consistency.

        Args:
            points: Nx2 array of (x, y) coordinates.

        Returns:
            Sum of squared perpendicular distances from points to fitted line.
        """
        return _line_straightness_error_impl(points)

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
