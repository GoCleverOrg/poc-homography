"""Apply calibration results to camera configuration.

This module provides utilities to:
- Apply (save) calibrated distortion coefficients to YAML files.
- Undistort points and images using the Brown-Conrady distortion model.
- Measure line straightness for calibration quality assessment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from poc_homography.calibration.lens_distortion.calibration_table import (
    CameraCalibrationTable,
    ZoomCalibrationEntry,
)
from poc_homography.camera_parameters import DistortionCoefficients  # noqa: TC001

if TYPE_CHECKING:
    from poc_homography.calibration.lens_distortion.distortion_solver import SolverResult

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


@dataclass
class CalibrationApplicationResult:
    """Result of applying calibration to a camera.

    Attributes:
        camera_id: Camera that was updated.
        zoom_factor: Zoom level that was calibrated.
        previous_coefficients: Coefficients before update (if any).
        new_coefficients: Newly applied coefficients.
        calibration_file: Path to calibration file.
        success: Whether application succeeded.
        message: Status message.
    """

    camera_id: str
    zoom_factor: float
    previous_coefficients: DistortionCoefficients | None
    new_coefficients: DistortionCoefficients
    calibration_file: str
    success: bool
    message: str

    def summary(self) -> str:
        """Generate summary of the calibration application."""
        lines = [
            "Calibration Application Result",
            f"Camera: {self.camera_id}",
            f"Zoom: {self.zoom_factor}x",
            f"Status: {'Success' if self.success else 'Failed'}",
            f"File: {self.calibration_file}",
            "",
        ]

        if self.previous_coefficients:
            lines.append("Previous coefficients:")
            lines.append(f"  k1={self.previous_coefficients.k1:.6f}")
            lines.append(f"  k2={self.previous_coefficients.k2:.6f}")
            lines.append(f"  k3={self.previous_coefficients.k3:.6f}")
            lines.append("")

        lines.append("New coefficients:")
        lines.append(f"  k1={self.new_coefficients.k1:.6f}")
        lines.append(f"  k2={self.new_coefficients.k2:.6f}")
        lines.append(f"  k3={self.new_coefficients.k3:.6f}")
        lines.append(f"  p1={self.new_coefficients.p1:.6f}")
        lines.append(f"  p2={self.new_coefficients.p2:.6f}")

        return "\n".join(lines)


def _apply_undistortion(
    distortion: DistortionCoefficients,
    camera_id: str,
    zoom_factor: float,
    calibration_dir: str | Path,
    validation_rmse: float = 0.0,
    source_images: list[str] | None = None,
    num_lines_used: int = 0,
    message: str = "Successfully applied calibration coefficients",
) -> CalibrationApplicationResult:
    """Shared implementation for applying distortion coefficients to a calibration file.

    Creates or updates the calibration table for the specified camera.
    """
    calibration_dir = Path(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    calibration_file = calibration_dir / f"{camera_id}_calibration.yaml"

    if calibration_file.exists():
        table = CameraCalibrationTable.load(calibration_file)
        previous_entry = table.get_entry(zoom_factor)
        previous_coeffs = previous_entry.to_distortion_coefficients() if previous_entry else None
    else:
        table = CameraCalibrationTable(camera_id=camera_id)
        previous_coeffs = None

    entry = ZoomCalibrationEntry.from_solver_result(
        zoom_factor=zoom_factor,
        distortion=distortion,
        validation_rmse=validation_rmse,
        source_images=source_images,
        num_lines_used=num_lines_used,
    )

    table.add_entry(entry)
    table.save(calibration_file)

    logger.info(f"Applied calibration for {camera_id} at zoom {zoom_factor}x")

    return CalibrationApplicationResult(
        camera_id=camera_id,
        zoom_factor=zoom_factor,
        previous_coefficients=previous_coeffs,
        new_coefficients=distortion,
        calibration_file=str(calibration_file),
        success=True,
        message=message,
    )


def apply_solver_result(
    solver_result: SolverResult,
    camera_id: str,
    zoom_factor: float,
    calibration_dir: str | Path,
    source_images: list[str] | None = None,
) -> CalibrationApplicationResult:
    """Apply solver result to camera calibration file.

    Creates or updates the calibration table for the specified camera.

    Args:
        solver_result: Result from DistortionSolver.solve().
        camera_id: Identifier for the camera.
        zoom_factor: Zoom level that was calibrated.
        calibration_dir: Directory to store calibration files.
        source_images: Optional list of images used for calibration.

    Returns:
        CalibrationApplicationResult with status and details.
    """
    return _apply_undistortion(
        distortion=solver_result.distortion,
        camera_id=camera_id,
        zoom_factor=zoom_factor,
        calibration_dir=calibration_dir,
        validation_rmse=solver_result.overall_rmse,
        source_images=source_images,
        num_lines_used=len(solver_result.line_errors),
        message=f"Successfully applied calibration with RMSE={solver_result.overall_rmse:.4f}px",
    )


def apply_distortion_coefficients(
    distortion: DistortionCoefficients,
    camera_id: str,
    zoom_factor: float,
    calibration_dir: str | Path,
    validation_rmse: float = 0.0,
    source_images: list[str] | None = None,
    num_lines_used: int = 0,
) -> CalibrationApplicationResult:
    """Apply specific distortion coefficients to camera calibration.

    Args:
        distortion: Distortion coefficients to apply.
        camera_id: Identifier for the camera.
        zoom_factor: Zoom level for this calibration.
        calibration_dir: Directory to store calibration files.
        validation_rmse: RMSE from validation (optional).
        source_images: Images used for calibration (optional).
        num_lines_used: Number of lines used (optional).

    Returns:
        CalibrationApplicationResult with status and details.
    """
    return _apply_undistortion(
        distortion=distortion,
        camera_id=camera_id,
        zoom_factor=zoom_factor,
        calibration_dir=calibration_dir,
        validation_rmse=validation_rmse,
        source_images=source_images,
        num_lines_used=num_lines_used,
    )


def load_distortion_coefficients(
    camera_id: str,
    zoom_factor: float,
    calibration_dir: str | Path,
) -> DistortionCoefficients | None:
    """Load calibrated distortion coefficients for a camera.

    This is the preferred way to load calibration coefficients.  It replaces
    ``DistortionCoefficients.load_calibration`` to avoid a circular dependency
    from the data-model module back into the calibration subsystem.

    Loads from a calibration table file and interpolates for the
    requested zoom level if needed.

    Args:
        camera_id: Identifier for the camera.
        zoom_factor: Current zoom level.
        calibration_dir: Directory containing calibration YAML files.

    Returns:
        DistortionCoefficients if calibration exists, None otherwise.
    """
    calibration_dir_path = Path(calibration_dir)
    calibration_file = calibration_dir_path / f"{camera_id}_calibration.yaml"

    if not calibration_file.exists():
        return None

    try:
        table = CameraCalibrationTable.load(calibration_file)
        if not table.entries:
            return None
        return table.get_coefficients(zoom_factor)
    except FileNotFoundError:
        logger.debug(f"Calibration file not found: {calibration_file}")
        return None
    except Exception:
        logger.warning(f"Failed to load calibration from {calibration_file}", exc_info=True)
        return None


def get_camera_calibration(
    camera_id: str,
    zoom_factor: float,
    calibration_dir: str | Path,
) -> DistortionCoefficients | None:
    """Get calibration coefficients for a camera at a specific zoom.

    Args:
        camera_id: Identifier for the camera.
        zoom_factor: Desired zoom level.
        calibration_dir: Directory containing calibration files.

    Returns:
        DistortionCoefficients if calibration exists, None otherwise.
    """
    calibration_dir = Path(calibration_dir)
    calibration_file = calibration_dir / f"{camera_id}_calibration.yaml"

    if not calibration_file.exists():
        return None

    table = CameraCalibrationTable.load(calibration_file)

    if not table.entries:
        return None

    return table.get_coefficients(zoom_factor)


def export_calibration_summary(
    calibration_dir: str | Path,
    output_file: str | Path | None = None,
) -> str:
    """Export summary of all camera calibrations.

    Args:
        calibration_dir: Directory containing calibration files.
        output_file: Optional path to write summary. If None, just returns string.

    Returns:
        Summary string.
    """
    calibration_dir = Path(calibration_dir)
    lines = ["# Camera Calibration Summary", ""]

    if not calibration_dir.exists():
        lines.append("No calibration directory found.")
        summary = "\n".join(lines)
        if output_file:
            Path(output_file).write_text(summary)
        return summary

    calibration_files = list(calibration_dir.glob("*_calibration.yaml"))

    if not calibration_files:
        lines.append("No calibration files found.")
        summary = "\n".join(lines)
        if output_file:
            Path(output_file).write_text(summary)
        return summary

    for cal_file in sorted(calibration_files):
        table = CameraCalibrationTable.load(cal_file)
        lines.append(table.summary())
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    summary = "\n".join(lines)

    if output_file:
        Path(output_file).write_text(summary)
        logger.info(f"Exported calibration summary to {output_file}")

    return summary


def merge_calibration_tables(
    tables: list[CameraCalibrationTable],
    camera_id: str | None = None,
) -> CameraCalibrationTable:
    """Merge multiple calibration tables into one.

    Useful when calibrations were performed separately at different zoom levels.

    Args:
        tables: List of calibration tables to merge.
        camera_id: Camera ID for merged table. Uses first table's ID if None.

    Returns:
        Merged CameraCalibrationTable.

    Raises:
        ValueError: If tables list is empty.
    """
    if not tables:
        raise ValueError("At least one calibration table required")

    if camera_id is None:
        camera_id = tables[0].camera_id

    merged = CameraCalibrationTable(camera_id=camera_id)

    for table in tables:
        for zoom, entry in table.entries.items():
            if zoom in merged.entries:
                # Keep entry with lower RMSE
                if entry.validation_rmse < merged.entries[zoom].validation_rmse:
                    merged.entries[zoom] = entry
                    logger.debug(f"Replaced entry for zoom {zoom} with better RMSE")
            else:
                merged.entries[zoom] = entry

    return merged
