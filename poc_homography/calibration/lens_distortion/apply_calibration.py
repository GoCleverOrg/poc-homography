"""Undistortion and measurement utilities for lens calibration.

This module provides utilities to:
- Undistort points and images using the Brown-Conrady distortion model.
- Measure line straightness for calibration quality assessment.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public undistortion and measurement utilities
# ---------------------------------------------------------------------------


def distort_points(
    points: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Apply forward distortion model to pixel points (Brown-Conrady).

    Takes undistorted pixel coordinates and computes their distorted positions.
    This is the direct forward model (no iteration needed).

    Args:
        points: Nx2 array of undistorted (u, v) pixel coordinates.
        k1, k2, k3: Radial distortion coefficients.
        p1, p2: Tangential distortion coefficients.
        fx, fy: Focal lengths.
        cx, cy: Principal point coordinates.

    Returns:
        Nx2 array of distorted (u, v) coordinates.
    """
    # Normalize to focal plane
    x = (points[:, 0] - cx) / fx
    y = (points[:, 1] - cy) / fy

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    # Forward radial distortion
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

    # Tangential distortion
    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_tangential = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # Apply distortion
    x_dist = x * radial + x_tangential
    y_dist = y * radial + y_tangential

    # Denormalize back to pixel coordinates
    return np.column_stack([x_dist * fx + cx, y_dist * fy + cy])


def undistort_points(
    points: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    max_iterations: int = 10,
    min_radial_factor: float = 1e-6,
) -> np.ndarray:
    """Apply inverse distortion model to pixel points (Brown-Conrady).

    Uses iterative Newton-Raphson to invert the forward distortion model.

    Args:
        points: Nx2 array of (u, v) pixel coordinates.
        k1, k2, k3: Radial distortion coefficients.
        p1, p2: Tangential distortion coefficients.
        fx, fy: Focal lengths.
        cx, cy: Principal point coordinates.
        max_iterations: Newton-Raphson iterations.
        min_radial_factor: Floor for radial factor to prevent division by zero.

    Returns:
        Nx2 array of undistorted (u, v) coordinates.
    """
    x = (points[:, 0] - cx) / fx
    y = (points[:, 1] - cy) / fy

    x_u = x.copy()
    y_u = y.copy()

    convergence_tol = 1e-10
    for _ in range(max_iterations):
        r2 = x_u * x_u + y_u * y_u
        r4 = r2 * r2
        r6 = r4 * r2

        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        radial = np.where(
            np.abs(radial) < min_radial_factor,
            np.where(radial >= 0, min_radial_factor, -min_radial_factor),
            radial,
        )

        dx_tangential = 2 * p1 * x_u * y_u + p2 * (r2 + 2 * x_u * x_u)
        dy_tangential = p1 * (r2 + 2 * y_u * y_u) + 2 * p2 * x_u * y_u

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

    return np.column_stack([x_u * fx + cx, y_u * fy + cy])


def undistort_image(
    image: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Undistort an image using the solver's Brown-Conrady forward model.

    For each pixel in the output (undistorted) image, applies the forward
    distortion model to find the source location in the input (distorted)
    image, then uses ``cv2.remap`` to interpolate.

    Uses float32 coordinate maps for reduced memory usage.
    """
    h, w = image.shape[:2]

    # Build coordinate grids directly as float32 for memory efficiency
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    x_norm = (x_coords - cx) / fx
    y_norm = (y_coords - cy) / fy

    r2 = x_norm * x_norm + y_norm * y_norm
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

    x_tangential = 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm * x_norm)
    y_tangential = p1 * (r2 + 2 * y_norm * y_norm) + 2 * p2 * x_norm * y_norm

    map_x = (x_norm * radial + x_tangential) * fx + cx
    map_y = (y_norm * radial + y_tangential) * fy + cy

    return cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def line_straightness_error(points: np.ndarray) -> float:
    """Calculate sum of squared perpendicular distances to best-fit line.

    Uses SVD to find the best-fit line through the points and computes the
    sum of squared perpendicular distances from each point to that line.

    This function is used by distortion solvers as the optimisation objective.

    Args:
        points: Nx2 array of (x, y) coordinates.

    Returns:
        Sum of squared perpendicular distances. Returns 0.0 for <3 points
        (since 2 points define a perfect line). Returns 1e12 for degenerate
        or non-finite inputs to penalise invalid configurations.
    """
    if len(points) < 3:
        # 2 points always fit a perfect line (zero residual)
        return 0.0

    # Guard against non-finite values from degenerate undistortion
    if not np.all(np.isfinite(points)):
        return 1e12

    # Fit line using SVD (total least squares)
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    try:
        _, _, Vt = np.linalg.svd(centered)
    except np.linalg.LinAlgError:
        return 1e12

    # Direction of the line is the first principal component
    # Normal to the line
    line_normal = np.array([-Vt[0, 1], Vt[0, 0]])

    # Perpendicular distances
    distances = np.abs(centered @ line_normal)

    return float(np.sum(distances**2))


def measure_line_straightness(pts: np.ndarray) -> dict:
    """Measure straightness of a set of 2-D points using total-least-squares.

    Args:
        pts: Nx2 array of (x, y) coordinates.

    Returns:
        Dictionary with ``rmse_pixels``, ``max_deviation_pixels``,
        ``r_squared``, and ``num_points``.
    """
    if len(pts) < 2:
        return {
            "rmse_pixels": 0.0,
            "max_deviation_pixels": 0.0,
            "r_squared": 1.0,
            "num_points": len(pts),
        }

    centroid = np.mean(pts, axis=0)
    centered = pts - centroid

    _, s, Vt = np.linalg.svd(centered)

    line_direction = Vt[0]
    line_normal = np.array([-line_direction[1], line_direction[0]])

    distances = np.abs(centered @ line_normal)

    rmse = float(np.sqrt(np.mean(distances**2)))
    max_deviation = float(np.max(distances))

    total_variance = np.sum(s**2)
    explained_variance = s[0] ** 2
    r_squared = explained_variance / total_variance if total_variance > 0 else 1.0

    return {
        "rmse_pixels": rmse,
        "max_deviation_pixels": max_deviation,
        "r_squared": float(r_squared),
        "num_points": len(pts),
    }
