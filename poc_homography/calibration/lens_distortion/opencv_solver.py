"""OpenCV-based distortion calibration using GCPs and line correspondences.

This module provides an alternative calibration method that uses OpenCV's
``cv2.calibrateCamera`` with fixed intrinsics. It constructs 2D-2D
correspondences from GCPs (Ground Control Points) and sampled line points,
treating map pixel coordinates as a flat Z=0 plane.

The solver fixes intrinsics (pre-computed from camera sensor specs) and
solves only for Brown-Conrady distortion coefficients (k1, k2, k3, p1, p2).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from poc_homography.calibration.lens_distortion.distortion_solver import (
    SolverConfig,
    SolverResult,
)
from poc_homography.camera_parameters import DistortionCoefficients
from poc_homography.types import Unitless

if TYPE_CHECKING:
    from poc_homography.calibration.lens_distortion.models import CameraLine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GCPCorrespondence:
    """A single GCP correspondence between map and camera coordinates.

    Attributes:
        gcp_id: GCP identifier.
        map_x: X coordinate in map pixels.
        map_y: Y coordinate in map pixels.
        camera_x: X coordinate in camera image pixels.
        camera_y: Y coordinate in camera image pixels.
    """

    gcp_id: str
    map_x: float
    map_y: float
    camera_x: float
    camera_y: float


@dataclass
class LineCorrespondencePair:
    """A line with both map and camera pixel endpoints.

    Attributes:
        line_id: Line identifier.
        map_start: (x, y) start in map pixels.
        map_end: (x, y) end in map pixels.
        camera_start: (x, y) start in camera pixels.
        camera_end: (x, y) end in camera pixels.
    """

    line_id: str
    map_start: tuple[float, float]
    map_end: tuple[float, float]
    camera_start: tuple[float, float]
    camera_end: tuple[float, float]


@dataclass
class LineSplitResult:
    """Result of splitting lines into training and validation sets.

    Attributes:
        training_lines: Lines used for calibration correspondences.
        validation_lines: Lines used for post-calibration straightness check.
    """

    training_lines: list[LineCorrespondencePair]
    validation_lines: list[LineCorrespondencePair]


@dataclass
class OpenCVSolverConfig:
    """Configuration for the OpenCV-based distortion solver.

    Attributes:
        num_samples_per_line: Number of points to sample along each line.
        train_split_ratio: Fraction of lines used for training (0.0-1.0).
        solver_config: Base solver config for compatibility.
    """

    num_samples_per_line: int = 20
    train_split_ratio: float = 0.7
    solver_config: SolverConfig = field(default_factory=SolverConfig)


# ---------------------------------------------------------------------------
# Correspondence construction
# ---------------------------------------------------------------------------


def build_gcp_correspondences(
    gcp_registry: list[dict],
    camera_annotations: list[dict],
) -> list[GCPCorrespondence]:
    """Build GCP correspondences from registry and camera annotations.

    Args:
        gcp_registry: List of GCP dicts with keys: id, pixel_x, pixel_y.
        camera_annotations: List of annotation dicts with keys:
            gcp_id, pixel_x, pixel_y.

    Returns:
        List of matched GCP correspondences.
    """
    gcp_map = {g["id"]: g for g in gcp_registry}
    correspondences = []

    for ann in camera_annotations:
        gcp_id = ann["gcp_id"]
        if gcp_id not in gcp_map:
            logger.warning("GCP %s not found in registry, skipping", gcp_id)
            continue

        gcp = gcp_map[gcp_id]
        correspondences.append(
            GCPCorrespondence(
                gcp_id=gcp_id,
                map_x=float(gcp["pixel_x"]),
                map_y=float(gcp["pixel_y"]),
                camera_x=float(ann["pixel_x"]),
                camera_y=float(ann["pixel_y"]),
            )
        )

    return correspondences


def build_line_correspondences(
    line_registry: list[dict],
    camera_line_annotations: list[dict],
) -> list[LineCorrespondencePair]:
    """Build line correspondences from registry and camera annotations.

    Args:
        line_registry: List of line dicts with keys:
            line_id, start_x, start_y, end_x, end_y (map pixels).
        camera_line_annotations: List of line annotation dicts with keys:
            line_id, start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y.

    Returns:
        List of matched line correspondences.
    """
    registry_map = {line["line_id"]: line for line in line_registry}
    correspondences = []

    for ann in camera_line_annotations:
        line_id = ann["line_id"]
        if line_id not in registry_map:
            logger.warning("Line %s not found in registry, skipping", line_id)
            continue

        reg = registry_map[line_id]
        correspondences.append(
            LineCorrespondencePair(
                line_id=line_id,
                map_start=(float(reg["start_x"]), float(reg["start_y"])),
                map_end=(float(reg["end_x"]), float(reg["end_y"])),
                camera_start=(float(ann["start_pixel_x"]), float(ann["start_pixel_y"])),
                camera_end=(float(ann["end_pixel_x"]), float(ann["end_pixel_y"])),
            )
        )

    return correspondences


def split_lines(
    lines: list[LineCorrespondencePair],
    train_ratio: float = 0.7,
) -> LineSplitResult:
    """Split lines into training and validation sets deterministically.

    Uses a hash of line_id for deterministic splitting, ensuring
    reproducibility across runs.

    Args:
        lines: All available line correspondences.
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


def sample_line_correspondences(
    line: LineCorrespondencePair,
    num_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample uniformly spaced points along a line in both map and camera space.

    Args:
        line: A line correspondence with map and camera endpoints.
        num_samples: Number of points to sample.

    Returns:
        Tuple of (object_points, image_points), each shape (N, 2).
    """
    t = np.linspace(0, 1, num_samples)

    map_start = np.array(line.map_start)
    map_end = np.array(line.map_end)
    map_points = map_start + np.outer(t, map_end - map_start)

    cam_start = np.array(line.camera_start)
    cam_end = np.array(line.camera_end)
    cam_points = cam_start + np.outer(t, cam_end - cam_start)

    return map_points.astype(np.float64), cam_points.astype(np.float64)


# ---------------------------------------------------------------------------
# OpenCV solver
# ---------------------------------------------------------------------------


class OpenCVDistortionSolver:
    """Solves for lens distortion using OpenCV's calibrateCamera with fixed intrinsics.

    Uses GCP and line correspondences to construct 3D-2D point pairs,
    treating map pixel coordinates as a flat Z=0 plane. Fixes intrinsics
    and solves only for distortion coefficients.
    """

    def __init__(self, config: OpenCVSolverConfig | None = None) -> None:
        self.config = config or OpenCVSolverConfig()

    def solve(
        self,
        gcp_correspondences: list[GCPCorrespondence],
        line_correspondences: list[LineCorrespondencePair],
        intrinsic_matrix: np.ndarray,
        validation_camera_lines: list[CameraLine] | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> SolverResult:
        """Run OpenCV calibration with fixed intrinsics.

        Args:
            gcp_correspondences: GCP-based point correspondences.
            line_correspondences: Line correspondences (will be split
                into training/validation).
            intrinsic_matrix: 3x3 camera intrinsic matrix K.
            validation_camera_lines: Optional pre-built CameraLine list
                for validation RMSE. If None, validation lines from split
                are used.
            image_size: (width, height) of the camera image. If None,
                estimated from intrinsics as (2*cx, 2*cy).

        Returns:
            SolverResult with distortion coefficients, reprojection error,
            and validation RMSE.
        """
        # Split lines into training and validation
        split = split_lines(
            line_correspondences, self.config.train_split_ratio
        )

        logger.info(
            "OpenCV solver: %d GCPs, %d training lines, %d validation lines",
            len(gcp_correspondences),
            len(split.training_lines),
            len(split.validation_lines),
        )

        # Build object points and image points
        object_points_list: list[np.ndarray] = []
        image_points_list: list[np.ndarray] = []

        # Add GCP correspondences
        for gcp in gcp_correspondences:
            object_points_list.append(np.array([gcp.map_x, gcp.map_y], dtype=np.float64))
            image_points_list.append(np.array([gcp.camera_x, gcp.camera_y], dtype=np.float64))

        # Add sampled training line correspondences
        for line in split.training_lines:
            map_pts, cam_pts = sample_line_correspondences(
                line, self.config.num_samples_per_line
            )
            for i in range(len(map_pts)):
                object_points_list.append(map_pts[i])
                image_points_list.append(cam_pts[i])

        if len(object_points_list) < 6:
            return SolverResult(
                distortion=DistortionCoefficients(),
                initial_error=0.0,
                final_error=0.0,
                rmse_per_line=[],
                overall_rmse=0.0,
                iterations=0,
                success=False,
                message="Insufficient correspondences (need at least 6 points)",
                line_errors=[],
            )

        # Convert to OpenCV format: single view with all points
        # Object points: (N, 1, 3) with Z=0, dtype float32 (Point3f)
        obj_pts_2d = np.array(object_points_list, dtype=np.float64)
        obj_pts_3d = np.zeros((len(obj_pts_2d), 1, 3), dtype=np.float32)
        obj_pts_3d[:, 0, 0] = obj_pts_2d[:, 0]
        obj_pts_3d[:, 0, 1] = obj_pts_2d[:, 1]

        # Image points: (N, 1, 2), dtype float32 (Point2f)
        img_pts = np.array(image_points_list, dtype=np.float32).reshape(-1, 1, 2)

        # Image size
        if image_size is None:
            cx = intrinsic_matrix[0, 2]
            cy = intrinsic_matrix[1, 2]
            image_size = (int(cx * 2), int(cy * 2))

        # Calibration flags: fix all intrinsics, solve only distortion
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_FIX_FOCAL_LENGTH
        )

        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objectPoints=[obj_pts_3d],
                imagePoints=[img_pts],
                imageSize=image_size,
                cameraMatrix=intrinsic_matrix.copy(),
                distCoeffs=None,
                flags=flags,
            )
        except cv2.error as e:
            return SolverResult(
                distortion=DistortionCoefficients(),
                initial_error=0.0,
                final_error=0.0,
                rmse_per_line=[],
                overall_rmse=0.0,
                iterations=0,
                success=False,
                message=f"OpenCV calibrateCamera failed: {e}",
                line_errors=[],
            )

        # Extract distortion coefficients (OpenCV order: k1, k2, p1, p2, k3, ...)
        dc = dist_coeffs.flatten()
        distortion = DistortionCoefficients(
            k1=Unitless(float(dc[0]) if len(dc) > 0 else 0.0),
            k2=Unitless(float(dc[1]) if len(dc) > 1 else 0.0),
            p1=Unitless(float(dc[2]) if len(dc) > 2 else 0.0),
            p2=Unitless(float(dc[3]) if len(dc) > 3 else 0.0),
            k3=Unitless(float(dc[4]) if len(dc) > 4 else 0.0),
        )

        # Compute reprojection error
        reprojection_error = self._compute_reprojection_error(
            obj_pts_3d, img_pts, camera_matrix, dist_coeffs, rvecs, tvecs
        )

        # Compute validation RMSE on validation lines
        validation_rmse, line_errors, rmse_per_line = self._compute_validation_rmse(
            split.validation_lines,
            distortion,
            intrinsic_matrix,
            validation_camera_lines,
        )

        return SolverResult(
            distortion=distortion,
            initial_error=reprojection_error,
            final_error=validation_rmse,
            rmse_per_line=rmse_per_line,
            overall_rmse=validation_rmse,
            iterations=1,
            success=True,
            message=f"OpenCV calibration converged (RMS reprojection error: {ret:.4f} px)",
            line_errors=line_errors,
            intrinsics={
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
            },
        )

    def _compute_reprojection_error(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvecs: list[np.ndarray],
        tvecs: list[np.ndarray],
    ) -> float:
        """Compute mean reprojection error across all correspondences."""
        projected, _ = cv2.projectPoints(
            object_points, rvecs[0], tvecs[0], camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        actual = image_points.reshape(-1, 2)
        errors = np.sqrt(np.sum((projected - actual) ** 2, axis=1))
        return float(np.mean(errors))

    def _compute_validation_rmse(
        self,
        validation_lines: list[LineCorrespondencePair],
        distortion: DistortionCoefficients,
        intrinsic_matrix: np.ndarray,
        camera_lines: list[CameraLine] | None = None,
    ) -> tuple[float, list[dict], list[float]]:
        """Compute line straightness RMSE on validation lines.

        Returns:
            Tuple of (overall_rmse, line_errors, rmse_per_line).
        """
        from poc_homography.calibration.lens_distortion.apply_calibration import (
            measure_line_straightness,
            undistort_points,
        )

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

        # If camera_lines provided, use them for validation
        if camera_lines:
            for cl in camera_lines:
                pts = cl.sample_points(self.config.num_samples_per_line)
                undistorted = undistort_points(pts, k1, k2, k3, p1, p2, fx, fy, cx, cy)
                result = measure_line_straightness(undistorted)
                rmse = float(result["rmse_pixels"])
                rmse_per_line.append(rmse)
                line_errors.append({
                    "line_id": cl.line_id,
                    "rmse_pixels": rmse,
                })
                all_squared_errors.append(rmse ** 2)
        else:
            # Use validation line correspondences
            for line in validation_lines:
                # Sample camera points along the line
                t = np.linspace(0, 1, self.config.num_samples_per_line)
                cam_start = np.array(line.camera_start)
                cam_end = np.array(line.camera_end)
                cam_pts = cam_start + np.outer(t, cam_end - cam_start)
                cam_pts = cam_pts.astype(np.float64)

                undistorted = undistort_points(cam_pts, k1, k2, k3, p1, p2, fx, fy, cx, cy)
                result = measure_line_straightness(undistorted)
                rmse = float(result["rmse_pixels"])
                rmse_per_line.append(rmse)
                line_errors.append({
                    "line_id": line.line_id,
                    "rmse_pixels": rmse,
                })
                all_squared_errors.append(rmse ** 2)

        if not rmse_per_line:
            return 0.0, [], []

        overall_rmse = float(np.sqrt(np.mean(all_squared_errors)))
        return overall_rmse, line_errors, rmse_per_line
