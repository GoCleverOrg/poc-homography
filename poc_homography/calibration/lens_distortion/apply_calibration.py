"""Apply calibration results to camera configuration.

This module provides utilities to update camera configurations with
calibrated distortion coefficients and verify the updates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from poc_homography.calibration.lens_distortion.calibration_table import (
    CameraCalibrationTable,
    ZoomCalibrationEntry,
)
from poc_homography.camera_parameters import DistortionCoefficients

if TYPE_CHECKING:
    from poc_homography.calibration.lens_distortion.distortion_solver import SolverResult

logger = logging.getLogger(__name__)


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
            f"Calibration Application Result",
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
    calibration_dir = Path(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    calibration_file = calibration_dir / f"{camera_id}_calibration.yaml"

    # Load existing table or create new one
    if calibration_file.exists():
        table = CameraCalibrationTable.load(calibration_file)
        previous_entry = table.get_entry(zoom_factor)
        previous_coeffs = previous_entry.to_distortion_coefficients() if previous_entry else None
    else:
        table = CameraCalibrationTable(camera_id=camera_id)
        previous_coeffs = None

    # Create new entry
    entry = ZoomCalibrationEntry.from_solver_result(
        zoom_factor=zoom_factor,
        distortion=solver_result.distortion,
        validation_rmse=solver_result.overall_rmse,
        source_images=source_images,
        num_lines_used=len(solver_result.line_errors),
    )

    # Add entry and save
    table.add_entry(entry)
    table.save(calibration_file)

    logger.info(f"Applied calibration for {camera_id} at zoom {zoom_factor}x")

    return CalibrationApplicationResult(
        camera_id=camera_id,
        zoom_factor=zoom_factor,
        previous_coefficients=previous_coeffs,
        new_coefficients=solver_result.distortion,
        calibration_file=str(calibration_file),
        success=True,
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
    calibration_dir = Path(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    calibration_file = calibration_dir / f"{camera_id}_calibration.yaml"

    # Load existing table or create new one
    if calibration_file.exists():
        table = CameraCalibrationTable.load(calibration_file)
        previous_entry = table.get_entry(zoom_factor)
        previous_coeffs = previous_entry.to_distortion_coefficients() if previous_entry else None
    else:
        table = CameraCalibrationTable(camera_id=camera_id)
        previous_coeffs = None

    # Create new entry
    entry = ZoomCalibrationEntry.from_solver_result(
        zoom_factor=zoom_factor,
        distortion=distortion,
        validation_rmse=validation_rmse,
        source_images=source_images,
        num_lines_used=num_lines_used,
    )

    # Add entry and save
    table.add_entry(entry)
    table.save(calibration_file)

    return CalibrationApplicationResult(
        camera_id=camera_id,
        zoom_factor=zoom_factor,
        previous_coefficients=previous_coeffs,
        new_coefficients=distortion,
        calibration_file=str(calibration_file),
        success=True,
        message="Successfully applied calibration coefficients",
    )


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
