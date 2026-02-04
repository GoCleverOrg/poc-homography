"""Distortion calibration using manually annotated N-point line traces.

This module provides a line-straightness-only distortion solver that works
with ``CameraLineAnnotation`` objects — N-point camera-pixel traces of
lines that should be straight in the undistorted image.

The solver optimises Brown-Conrady distortion coefficients (k1, k2, k3,
p1, p2) using scipy L-BFGS-B to minimise the sum of squared perpendicular
distances from undistorted line points to their best-fit line (SVD).

Unlike the auto-detected ``CameraLine`` approach in ``distortion_solver.py``,
this solver consumes **manually annotated** multi-point line traces from the
camera_line_annotator webapp, giving the operator control over which lines
are used and how densely they are traced.

Train/validation split
----------------------
Lines are deterministically split (by line_id hash) into a training set
used for optimisation and a validation set used for post-calibration RMSE.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from poc_homography.calibration.lens_distortion.distortion_solver import (
    SolverConfig,
    SolverResult,
)
from poc_homography.camera_parameters import DistortionCoefficients
from poc_homography.types import Unitless

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CameraLineAnnotation:
    """N-point camera-pixel trace of a line.

    Attributes:
        line_id: Line identifier.
        points: N camera-pixel points tracing the line, [(x, y), ...].
    """

    line_id: str
    points: list[tuple[float, float]]


@dataclass
class LineSplitResult:
    """Result of splitting line annotations into training and validation sets.

    Attributes:
        training_lines: Lines used for straightness optimisation.
        validation_lines: Lines used for post-calibration straightness check.
    """

    training_lines: list[CameraLineAnnotation]
    validation_lines: list[CameraLineAnnotation]


@dataclass
class AnnotatedLineSolverConfig:
    """Configuration for the annotated-line distortion solver.

    Attributes:
        train_split_ratio: Fraction of lines used for training (0.0-1.0).
        max_iterations: Maximum L-BFGS-B iterations.
        tolerance: Convergence tolerance (ftol).
        use_radial_only: If True, only optimise k1, k2, k3 (p1=p2=0).
        solver_config: Base solver config for compatibility.
    """

    train_split_ratio: float = 0.7
    max_iterations: int = 1000
    tolerance: float = 1e-10
    use_radial_only: bool = False
    solver_config: SolverConfig = field(default_factory=SolverConfig)


# ---------------------------------------------------------------------------
# Line annotation construction
# ---------------------------------------------------------------------------


def build_camera_line_annotations(
    camera_line_annotations: list[dict],
) -> list[CameraLineAnnotation]:
    """Build camera line annotations from annotation dicts.

    Reads the ``points`` array from each annotation (N-point data).
    Falls back to ``[start, end]`` if no points array is present.

    Args:
        camera_line_annotations: List of line annotation dicts with keys:
            line_id, and either ``points`` (list of [x,y]) or
            ``start_pixel_x/y``, ``end_pixel_x/y``.

    Returns:
        List of CameraLineAnnotation with N-point traces.
    """
    annotations = []

    for ann in camera_line_annotations:
        line_id = ann.get("line_id", "unknown")
        points_raw = ann.get("points")

        if points_raw and len(points_raw) >= 2:
            points = [(float(p[0]), float(p[1])) for p in points_raw]
        else:
            # Fall back to start/end endpoints
            start_x = ann.get("start_pixel_x")
            start_y = ann.get("start_pixel_y")
            end_x = ann.get("end_pixel_x")
            end_y = ann.get("end_pixel_y")
            if start_x is not None and end_x is not None:
                points = [
                    (float(start_x), float(start_y)),
                    (float(end_x), float(end_y)),
                ]
            else:
                logger.warning("Line %s has no valid points, skipping", line_id)
                continue

        annotations.append(CameraLineAnnotation(line_id=line_id, points=points))

    return annotations


def split_lines(
    lines: list[CameraLineAnnotation],
    train_ratio: float = 0.7,
) -> LineSplitResult:
    """Split line annotations into training and validation sets deterministically.

    Uses a hash of line_id for deterministic splitting.

    Args:
        lines: All available line annotations.
        train_ratio: Fraction of lines for training (0.0-1.0).

    Returns:
        LineSplitResult with training and validation sets.
    """
    if not lines:
        return LineSplitResult(training_lines=[], validation_lines=[])

    train_ratio = max(0.0, min(1.0, train_ratio))
    training = []
    validation = []

    for line in lines:
        hash_val = int(hashlib.md5(line.line_id.encode(), usedforsecurity=False).hexdigest(), 16)
        if (hash_val % 1000) / 1000.0 < train_ratio:
            training.append(line)
        else:
            validation.append(line)

    # Ensure at least one line in each set when possible
    if not training and validation:
        training.append(validation.pop(0))
    elif not validation and training:
        validation.append(training.pop(-1))

    logger.info(
        "Line split: %d training, %d validation (ratio=%.2f)",
        len(training),
        len(validation),
        train_ratio,
    )

    return LineSplitResult(training_lines=training, validation_lines=validation)


# ---------------------------------------------------------------------------
# Annotated-line distortion solver
# ---------------------------------------------------------------------------


class AnnotatedLineSolver:
    """Solves for lens distortion using line straightness on N-point traces.

    Optimises Brown-Conrady distortion coefficients so that manually
    annotated camera lines become as straight as possible after
    undistortion.
    """

    def __init__(self, config: AnnotatedLineSolverConfig | None = None) -> None:
        self.config = config or AnnotatedLineSolverConfig()

    # ---- public API -------------------------------------------------------

    def solve(
        self,
        lines: list[CameraLineAnnotation],
        intrinsic_matrix: np.ndarray,
        initial_guess: DistortionCoefficients | None = None,
    ) -> SolverResult:
        """Optimise distortion coefficients to straighten annotated lines.

        Args:
            lines: N-point camera line annotations (>= 3 points each
                are used; 2-point lines are ignored as they provide no
                straightness constraint).
            intrinsic_matrix: 3x3 camera intrinsic matrix K.
            initial_guess: Starting distortion coefficients (zeros if None).

        Returns:
            SolverResult with optimised coefficients, training/validation
            RMSE, and per-line errors.
        """
        if intrinsic_matrix is None:
            return self._fail("intrinsic_matrix is required")

        # Filter to lines with >= 3 points (2-point lines give zero error)
        usable = [la for la in lines if len(la.points) >= 3]
        if not usable:
            return self._fail(
                "No usable line annotations (need lines with >= 3 points)"
            )

        # Split into training / validation
        split = split_lines(usable, self.config.train_split_ratio)
        training_lines = split.training_lines
        validation_lines = split.validation_lines

        if not training_lines:
            # With very few lines the split may leave training empty.
            # Fall back to using all lines for both training and validation.
            training_lines = usable
            validation_lines = []

        fx = float(intrinsic_matrix[0, 0])
        fy = float(intrinsic_matrix[1, 1])
        cx = float(intrinsic_matrix[0, 2])
        cy = float(intrinsic_matrix[1, 2])

        # Prepare training line point arrays
        train_arrays = [
            np.array(la.points, dtype=np.float64) for la in training_lines
        ]

        # Initial guess
        if initial_guess is None:
            initial_guess = DistortionCoefficients()

        if self.config.use_radial_only:
            x0 = np.array([
                float(initial_guess.k1),
                float(initial_guess.k2),
                float(initial_guess.k3),
            ])
            bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        else:
            x0 = np.array([
                float(initial_guess.k1),
                float(initial_guess.k2),
                float(initial_guess.k3),
                float(initial_guess.p1),
                float(initial_guess.p2),
            ])
            bounds = [
                (-1.0, 1.0),   # k1
                (-1.0, 1.0),   # k2
                (-1.0, 1.0),   # k3
                (-0.1, 0.1),   # p1
                (-0.1, 0.1),   # p2
            ]

        # Compute initial error
        initial_error = self._total_straightness_error(
            x0, train_arrays, fx, fy, cx, cy,
        )
        logger.info("Initial straightness error: %.6f", initial_error)

        # Optimise
        opt_result = minimize(
            self._total_straightness_error,
            x0,
            args=(train_arrays, fx, fy, cx, cy),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": self.config.max_iterations,
                "ftol": self.config.tolerance,
            },
        )

        # Extract coefficients
        if self.config.use_radial_only:
            k1, k2, k3 = opt_result.x
            p1, p2 = 0.0, 0.0
        else:
            k1, k2, k3, p1, p2 = opt_result.x

        optimised = DistortionCoefficients(
            k1=Unitless(float(k1)),
            k2=Unitless(float(k2)),
            k3=Unitless(float(k3)),
            p1=Unitless(float(p1)),
            p2=Unitless(float(p2)),
        )

        # Compute validation RMSE on all lines (training + validation)
        all_lines = training_lines + validation_lines
        validation_rmse, line_errors, rmse_per_line = self._compute_validation_rmse(
            all_lines, optimised, intrinsic_matrix,
        )

        n_train = len(training_lines)
        n_val = len(validation_lines)

        # scipy reports success=False when the objective is already at its
        # minimum (zero iterations).  Treat near-zero error as success.
        converged = opt_result.success or opt_result.fun < 1e-12

        message = (
            f"Line straightness optimisation "
            f"{'converged' if converged else 'did not converge'} "
            f"({n_train} training, {n_val} validation lines, "
            f"RMSE: {validation_rmse:.4f} px)"
        )
        logger.info(message)

        return SolverResult(
            distortion=optimised,
            initial_error=initial_error,
            final_error=opt_result.fun,
            rmse_per_line=rmse_per_line,
            overall_rmse=validation_rmse,
            iterations=opt_result.nit,
            success=converged,
            message=message,
            line_errors=line_errors,
            intrinsics={
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            },
        )

    # ---- objective --------------------------------------------------------

    def _total_straightness_error(
        self,
        params: np.ndarray,
        line_arrays: list[np.ndarray],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> float:
        """Sum of squared perpendicular distances across all training lines."""
        if self.config.use_radial_only:
            k1, k2, k3 = params
            p1, p2 = 0.0, 0.0
        else:
            k1, k2, k3, p1, p2 = params

        total = 0.0
        for pts in line_arrays:
            undistorted = self._undistort_points_fast(
                pts, k1, k2, k3, p1, p2, fx, fy, cx, cy,
            )
            total += self._line_straightness_error(undistorted)

        return total

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _fail(message: str) -> SolverResult:
        return SolverResult(
            distortion=DistortionCoefficients(),
            initial_error=0.0,
            final_error=0.0,
            rmse_per_line=[],
            overall_rmse=0.0,
            iterations=0,
            success=False,
            message=message,
            line_errors=[],
        )

    @staticmethod
    def _undistort_points_fast(
        points: np.ndarray,
        k1: float, k2: float, k3: float,
        p1: float, p2: float,
        fx: float, fy: float, cx: float, cy: float,
    ) -> np.ndarray:
        """Undistort points using iterative Newton-Raphson."""
        x = (points[:, 0] - cx) / fx
        y = (points[:, 1] - cy) / fy

        x_u = x.copy()
        y_u = y.copy()

        for _ in range(10):
            r2 = x_u * x_u + y_u * y_u
            r4 = r2 * r2
            r6 = r4 * r2
            radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
            radial = np.where(np.abs(radial) < 1e-6, 1e-6, radial)

            dx_t = 2 * p1 * x_u * y_u + p2 * (r2 + 2 * x_u * x_u)
            dy_t = p1 * (r2 + 2 * y_u * y_u) + 2 * p2 * x_u * y_u

            x_u_new = (x - dx_t) / radial
            y_u_new = (y - dy_t) / radial

            delta = np.max(np.abs(x_u_new - x_u)) + np.max(np.abs(y_u_new - y_u))
            x_u = x_u_new
            y_u = y_u_new
            if delta < 1e-10:
                break

        nan_mask = np.isfinite(x_u) & np.isfinite(y_u)
        x_u = np.where(nan_mask, x_u, x)
        y_u = np.where(nan_mask, y_u, y)

        return np.column_stack([x_u * fx + cx, y_u * fy + cy])

    @staticmethod
    def _line_straightness_error(points: np.ndarray) -> float:
        """Sum of squared perpendicular distances to best-fit line."""
        if len(points) < 3:
            return 0.0

        # Guard against non-finite values from degenerate undistortion
        if not np.all(np.isfinite(points)):
            return 1e12

        centroid = np.mean(points, axis=0)
        centered = points - centroid

        try:
            _, _, Vt = np.linalg.svd(centered)
        except np.linalg.LinAlgError:
            return 1e12

        line_normal = np.array([-Vt[0, 1], Vt[0, 0]])
        distances = np.abs(centered @ line_normal)
        return float(np.sum(distances**2))

    def _compute_validation_rmse(
        self,
        lines: list[CameraLineAnnotation],
        distortion: DistortionCoefficients,
        intrinsic_matrix: np.ndarray,
    ) -> tuple[float, list[dict], list[float]]:
        """Compute line straightness RMSE on lines using N-point traces."""
        fx = float(intrinsic_matrix[0, 0])
        fy = float(intrinsic_matrix[1, 1])
        cx = float(intrinsic_matrix[0, 2])
        cy = float(intrinsic_matrix[1, 2])

        k1 = float(distortion.k1)
        k2 = float(distortion.k2)
        k3 = float(distortion.k3)
        p1 = float(distortion.p1)
        p2 = float(distortion.p2)

        line_errors: list[dict] = []
        rmse_per_line: list[float] = []
        all_squared_errors: list[float] = []

        for la in lines:
            pts = np.array(la.points, dtype=np.float64)
            if len(pts) < 3:
                continue

            undistorted = self._undistort_points_fast(
                pts, k1, k2, k3, p1, p2, fx, fy, cx, cy,
            )
            err = self._line_straightness_error(undistorted)
            rmse = float(np.sqrt(err / len(pts)))
            rmse_per_line.append(rmse)
            line_errors.append({
                "line_id": la.line_id,
                "rmse_pixels": rmse,
            })
            all_squared_errors.append(rmse ** 2)

        if not rmse_per_line:
            return 0.0, [], []

        overall_rmse = float(np.sqrt(np.mean(all_squared_errors)))
        return overall_rmse, line_errors, rmse_per_line
