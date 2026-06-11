"""Application-layer service for homography precision computation.

Framework-agnostic orchestration of homography computation plus metric and
error aggregation for the precision-visualization endpoints.  This module must
not import Django or anything from ``webapp/`` so it can be reused and unit
tested without HTTP machinery.

The domain raises ``ValueError``/``RuntimeError`` on computation failures; the
service propagates them unchanged so the calling view can map them to the
existing HTTP error responses.  Result types are frozen dataclasses that carry
the exact response payload via ``to_payload()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from poc_homography.calibration.lens_distortion.calibration_table import (
    load_calibration_for_camera,
)
from poc_homography.domain.vo import PixelPoint
from poc_homography.homography.map_points import MapPointHomography

if TYPE_CHECKING:
    from pathlib import Path

    from poc_homography.map_points.gcp_registry import GCPRegistry

logger = logging.getLogger(__name__)


class HomographyProjection(Protocol):
    """Minimal projection interface used by the error-aggregation services.

    Implemented by ``MapPointHomography``; declared as a protocol so unit tests
    can inject identity-like stubs without RANSAC data.
    """

    def camera_to_map(self, camera_pixel: PixelPoint, point_id: str = ...) -> Any:
        """Project a camera pixel to map coordinates (``.pixel_x``/``.pixel_y``)."""
        ...

    def map_to_camera(self, map_coord: PixelPoint) -> PixelPoint:
        """Project a map coordinate to a camera pixel (``.x``/``.y``)."""
        ...


class LineNotInRegistryError(Exception):
    """Raised when a line annotation references a line id absent from the registry.

    Carries the offending ``line_id`` so the caller can reproduce the existing
    ``f"Line {line_id} not found in line registry"`` error response.
    """

    def __init__(self, line_id: str) -> None:
        """Store the missing ``line_id`` and build the message."""
        self.line_id = line_id
        super().__init__(f"Line {line_id} not found in line registry")


def perpendicular_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Calculate perpendicular distance from point p to line defined by points a and b.

    Args:
        p: Point as numpy array [x, y].
        a: Line start point as numpy array [x, y].
        b: Line end point as numpy array [x, y].

    Returns:
        Perpendicular distance in pixels.
    """
    # Line vector
    v = b - a
    # Point vector from a to p
    w = p - a

    # Project w onto v
    c1 = np.dot(w, v)
    c2 = np.dot(v, v)

    if c2 == 0:
        # Line has zero length (a and b are the same point)
        return float(np.linalg.norm(w))

    # Perpendicular distance = |w - proj_v(w)|
    b_param = c1 / c2
    proj = a + b_param * v
    return float(np.linalg.norm(p - proj))


def build_distortion_params(
    camera_name: str,
    zoom: float,
    calibrations_dir: Path,
) -> dict[str, float]:
    """Build distortion + intrinsic kwargs for ``MapPointHomography``.

    Loads the calibration table for ``camera_name``, interpolates the
    distortion coefficients and intrinsics at ``zoom``, and returns a dict with
    keys k1,k2,k3,p1,p2,fx,fy,cx,cy.  Returns an empty dict if any piece is
    unavailable, preserving the broad try/except + debug-logging behaviour of
    the original view helper.

    Args:
        camera_name: Camera identifier used to locate the calibration table.
        zoom: Zoom factor at which to interpolate coefficients/intrinsics.
        calibrations_dir: Directory containing per-camera calibration tables.

    Returns:
        Mapping of distortion + intrinsic parameters, or ``{}`` on any failure.
    """
    try:
        table = load_calibration_for_camera(camera_name, calibrations_dir)
        if table is None:
            return {}
        coeffs = table.get_coefficients(zoom)
        intrinsics = table.get_intrinsics(zoom)
        if intrinsics is None:
            return {}
        return {
            "k1": float(coeffs.k1),
            "k2": float(coeffs.k2),
            "k3": float(coeffs.k3),
            "p1": float(coeffs.p1),
            "p2": float(coeffs.p2),
            **intrinsics,
        }
    except Exception:
        logger.debug("Failed to load distortion params for %s", camera_name, exc_info=True)
        return {}


@dataclass(frozen=True)
class GcpPrecisionResult:
    """Aggregated GCP homography precision metrics, per-point errors and overlays."""

    num_gcps: int
    num_inliers: int
    inlier_ratio: float
    mean_reproj_error: float
    max_reproj_error: float
    rmse: float
    per_point_errors: list[dict[str, Any]]
    camera_annotations: list[dict[str, Any]]
    camera_reprojected: list[dict[str, Any]]
    map_gcps: list[dict[str, Any]]
    map_projected: list[dict[str, Any]]

    def to_payload(self) -> dict[str, Any]:
        """Build the HTTP response payload (keys/rounding identical to the view)."""
        return {
            "success": True,
            "metrics": {
                "num_gcps": self.num_gcps,
                "num_inliers": self.num_inliers,
                "inlier_ratio": round(self.inlier_ratio, 2),
                "mean_reproj_error": round(self.mean_reproj_error, 2),
                "max_reproj_error": round(self.max_reproj_error, 2),
                "rmse": round(self.rmse, 2),
            },
            "per_point_errors": self.per_point_errors,
            "overlays": {
                "camera": {
                    "annotations": self.camera_annotations,
                    "reprojected_gcps": self.camera_reprojected,
                },
                "map": {
                    "gcps": self.map_gcps,
                    "projected_annotations": self.map_projected,
                },
            },
        }


def compute_gcp_precision(
    annotations: list[dict[str, Any]],
    registry: GCPRegistry,
    distortion: dict[str, float],
    ransac_threshold: float = 50.0,
    min_inlier_ratio: float = 0.5,
) -> GcpPrecisionResult:
    """Compute a GCP homography and aggregate per-point errors, overlays and metrics.

    Args:
        annotations: Camera GCP annotations (``gcp_id``, ``pixel_x``, ``pixel_y``).
        registry: GCP registry providing map coordinates per ``gcp_id``.
        distortion: Distortion/intrinsic kwargs for ``MapPointHomography``.
        ransac_threshold: RANSAC inlier threshold forwarded to the domain.
        min_inlier_ratio: Minimum inlier ratio forwarded to the domain.

    Returns:
        Aggregated result carrying the response payload.

    Raises:
        ValueError: Propagated from the domain on invalid input.
        RuntimeError: Propagated from the domain on computation failure.
    """
    homography = MapPointHomography(map_id=registry.map_id, **distortion)
    result = homography.compute_from_gcps(
        gcps=annotations,
        map_registry=registry,
        ransac_threshold=ransac_threshold,
        min_inlier_ratio=min_inlier_ratio,
    )

    per_point_errors: list[dict[str, Any]] = []
    camera_annotations: list[dict[str, Any]] = []
    camera_reprojected: list[dict[str, Any]] = []
    map_gcps: list[dict[str, Any]] = []
    map_projected: list[dict[str, Any]] = []

    for annotation in annotations:
        gcp_id = annotation["gcp_id"]

        # Get original camera pixel (annotation position)
        camera_x = annotation["pixel_x"]
        camera_y = annotation["pixel_y"]
        camera_pixel = PixelPoint.create(camera_x, camera_y)

        # Get GCP map coordinate from registry
        gcp = registry.points[gcp_id]
        map_x = gcp.pixel_x
        map_y = gcp.pixel_y

        # Project annotation to map using camera_to_map()
        projected_map = homography.camera_to_map(camera_pixel)

        # Reproject GCP back to camera using map_to_camera()
        gcp_coord = PixelPoint.create(map_x, map_y)
        reprojected_camera = homography.map_to_camera(gcp_coord)

        # Calculate per-point error (Euclidean distance between original annotation
        # and reprojected GCP)
        original = np.array([camera_x, camera_y])
        reprojected = np.array([reprojected_camera.x, reprojected_camera.y])
        error_px = float(np.linalg.norm(reprojected - original))

        # Calculate per-axis errors for camera frame
        camera_dx = reprojected_camera.x - camera_x
        camera_dy = reprojected_camera.y - camera_y

        # Calculate per-axis errors for map frame
        map_dx = projected_map.pixel_x - map_x
        map_dy = projected_map.pixel_y - map_y

        per_point_errors.append(
            {
                "gcp_id": gcp_id,
                "error_px": round(error_px, 2),
                "camera_dx": round(camera_dx, 2),
                "camera_dy": round(camera_dy, 2),
                "map_dx": round(map_dx, 2),
                "map_dy": round(map_dy, 2),
                "camera_original": [round(camera_x, 2), round(camera_y, 2)],
                "camera_reprojected": [
                    round(reprojected_camera.x, 2),
                    round(reprojected_camera.y, 2),
                ],
                "map_original": [round(map_x, 2), round(map_y, 2)],
                "map_projected": [round(projected_map.pixel_x, 2), round(projected_map.pixel_y, 2)],
            }
        )

        # Collect overlay data
        camera_annotations.append(
            {
                "gcp_id": gcp_id,
                "x": round(camera_x, 2),
                "y": round(camera_y, 2),
            }
        )
        camera_reprojected.append(
            {
                "gcp_id": gcp_id,
                "x": round(reprojected_camera.x, 2),
                "y": round(reprojected_camera.y, 2),
            }
        )
        map_gcps.append(
            {
                "gcp_id": gcp_id,
                "x": round(map_x, 2),
                "y": round(map_y, 2),
            }
        )
        map_projected.append(
            {
                "gcp_id": gcp_id,
                "x": round(projected_map.pixel_x, 2),
                "y": round(projected_map.pixel_y, 2),
            }
        )

    return GcpPrecisionResult(
        num_gcps=result.num_gcps,
        num_inliers=result.num_inliers,
        inlier_ratio=result.inlier_ratio,
        mean_reproj_error=result.mean_reproj_error,
        max_reproj_error=result.max_reproj_error,
        rmse=result.rmse,
        per_point_errors=per_point_errors,
        camera_annotations=camera_annotations,
        camera_reprojected=camera_reprojected,
        map_gcps=map_gcps,
        map_projected=map_projected,
    )


@dataclass(frozen=True)
class LineHomographyResult:
    """Line-based homography metrics plus the resulting matrix."""

    num_lines: int
    num_inliers: int
    inlier_ratio: float
    mean_perp_error: float
    max_perp_error: float
    rmse: float
    homography_matrix: np.ndarray

    def to_payload(self) -> dict[str, Any]:
        """Build the HTTP response payload (keys/rounding identical to the view)."""
        return {
            "success": True,
            "homography_source": "lines",
            "metrics": {
                "num_lines": self.num_lines,
                "num_inliers": self.num_inliers,
                "inlier_ratio": round(self.inlier_ratio, 2),
                "mean_perp_error": round(self.mean_perp_error, 2),
                "max_perp_error": round(self.max_perp_error, 2),
                "rmse": round(self.rmse, 2),
            },
            "homography_matrix": self.homography_matrix.tolist(),
        }


def compute_line_homography(
    line_annotations: list[dict[str, Any]],
    line_registry: dict[str, dict[str, float]],
    distortion: dict[str, float],
    map_id: str,
    ransac_threshold: float = 50.0,
    min_inlier_ratio: float = 0.3,
) -> LineHomographyResult:
    """Compute a line-based homography and assemble its metrics.

    Args:
        line_annotations: Camera line annotations.
        line_registry: Map-coordinate line definitions keyed by ``line_id``.
        distortion: Distortion/intrinsic kwargs for ``MapPointHomography``.
        map_id: Map identifier for the homography.
        ransac_threshold: RANSAC inlier threshold forwarded to the domain.
        min_inlier_ratio: Minimum inlier ratio forwarded to the domain.

    Returns:
        Aggregated result carrying the response payload.

    Raises:
        ValueError: Propagated from the domain on invalid input.
        RuntimeError: Propagated from the domain on computation failure.
    """
    homography = MapPointHomography(map_id=map_id, **distortion)
    result = homography.compute_from_lines(
        line_annotations=line_annotations,
        line_registry=line_registry,
        ransac_threshold=ransac_threshold,
        min_inlier_ratio=min_inlier_ratio,
    )
    return LineHomographyResult(
        num_lines=result.num_lines,
        num_inliers=result.num_inliers,
        inlier_ratio=result.inlier_ratio,
        mean_perp_error=result.mean_perp_error,
        max_perp_error=result.max_perp_error,
        rmse=result.rmse,
        homography_matrix=result.homography_matrix,
    )


@dataclass(frozen=True)
class LineErrorsResult:
    """Aggregated per-line errors and overlays for a precomputed homography."""

    num_lines: int
    mean_line_error: float
    max_line_error: float
    per_line_errors: list[dict[str, Any]]
    camera_annotations: list[dict[str, Any]]
    camera_reprojected_lines: list[dict[str, Any]]
    map_gcp_lines: list[dict[str, Any]]
    map_projected_lines: list[dict[str, Any]]

    def to_payload(self) -> dict[str, Any]:
        """Build the HTTP response payload (keys/rounding identical to the view)."""
        return {
            "success": True,
            "metrics": {
                "num_lines": self.num_lines,
                "mean_line_error": round(self.mean_line_error, 2),
                "max_line_error": round(self.max_line_error, 2),
            },
            "per_line_errors": self.per_line_errors,
            "line_overlays": {
                "camera": {
                    "annotations": self.camera_annotations,
                    "reprojected_lines": self.camera_reprojected_lines,
                },
                "map": {
                    "gcp_lines": self.map_gcp_lines,
                    "projected_lines": self.map_projected_lines,
                },
            },
        }


def compute_line_errors(
    homography: HomographyProjection,
    line_annotations: list[dict[str, Any]],
    line_registry: dict[str, dict[str, float]],
) -> LineErrorsResult:
    """Aggregate per-line errors and overlays for a precomputed homography.

    Args:
        homography: Projection object exposing ``camera_to_map``/``map_to_camera``.
        line_annotations: Camera line annotations to evaluate.
        line_registry: Map-coordinate line definitions keyed by ``line_id``.

    Returns:
        Aggregated result carrying the response payload.

    Raises:
        LineNotInRegistryError: If an annotation references an unknown ``line_id``.
    """
    per_line_errors: list[dict[str, Any]] = []
    camera_annotations: list[dict[str, Any]] = []
    camera_reprojected_lines: list[dict[str, Any]] = []
    map_gcp_lines: list[dict[str, Any]] = []
    map_projected_lines: list[dict[str, Any]] = []

    total_error = 0.0
    max_error = 0.0

    for line_annotation in line_annotations:
        line_id = line_annotation["line_id"]

        # Get line definition from registry
        if line_id not in line_registry:
            raise LineNotInRegistryError(line_id)

        line_def = line_registry[line_id]

        # Ground truth line in map coordinates (directly from line registry)
        map_start = np.array([line_def["start_x"], line_def["start_y"]])
        map_end = np.array([line_def["end_x"], line_def["end_y"]])

        # Annotated line in camera coordinates
        camera_start = np.array(
            [line_annotation["start_pixel_x"], line_annotation["start_pixel_y"]]
        )
        camera_end = np.array([line_annotation["end_pixel_x"], line_annotation["end_pixel_y"]])

        # Project camera line endpoints to map
        projected_start = homography.camera_to_map(
            PixelPoint.create(camera_start[0], camera_start[1])
        )
        projected_end = homography.camera_to_map(PixelPoint.create(camera_end[0], camera_end[1]))
        projected_start_map = np.array([projected_start.pixel_x, projected_start.pixel_y])
        projected_end_map = np.array([projected_end.pixel_x, projected_end.pixel_y])

        # Compute errors: perpendicular distance from projected points to ground truth line
        start_error = perpendicular_distance(projected_start_map, map_start, map_end)
        end_error = perpendicular_distance(projected_end_map, map_start, map_end)

        # Also compute reverse: project GCP line to camera and compare
        reprojected_start = homography.map_to_camera(PixelPoint.create(map_start[0], map_start[1]))
        reprojected_end = homography.map_to_camera(PixelPoint.create(map_end[0], map_end[1]))
        reprojected_start_camera = np.array([reprojected_start.x, reprojected_start.y])
        reprojected_end_camera = np.array([reprojected_end.x, reprojected_end.y])

        # Compute errors in camera space
        camera_start_error = perpendicular_distance(
            camera_start, reprojected_start_camera, reprojected_end_camera
        )
        camera_end_error = perpendicular_distance(
            camera_end, reprojected_start_camera, reprojected_end_camera
        )

        # Average error for this line (using camera space errors as primary metric)
        line_error = (camera_start_error + camera_end_error) / 2.0

        total_error += line_error
        max_error = max(max_error, line_error)

        per_line_errors.append(
            {
                "line_id": line_id,
                "error_px": round(line_error, 2),
                "start_error": round(camera_start_error, 2),
                "end_error": round(camera_end_error, 2),
                "map_start_error": round(start_error, 2),
                "map_end_error": round(end_error, 2),
            }
        )

        # Collect overlay data for camera frame
        camera_annotations.append(
            {
                "line_id": line_id,
                "start": [round(camera_start[0], 2), round(camera_start[1], 2)],
                "end": [round(camera_end[0], 2), round(camera_end[1], 2)],
            }
        )
        camera_reprojected_lines.append(
            {
                "line_id": line_id,
                "start": [
                    round(reprojected_start_camera[0], 2),
                    round(reprojected_start_camera[1], 2),
                ],
                "end": [round(reprojected_end_camera[0], 2), round(reprojected_end_camera[1], 2)],
            }
        )

        # Collect overlay data for map frame
        map_gcp_lines.append(
            {
                "line_id": line_id,
                "start": [round(map_start[0], 2), round(map_start[1], 2)],
                "end": [round(map_end[0], 2), round(map_end[1], 2)],
            }
        )
        map_projected_lines.append(
            {
                "line_id": line_id,
                "start": [round(projected_start_map[0], 2), round(projected_start_map[1], 2)],
                "end": [round(projected_end_map[0], 2), round(projected_end_map[1], 2)],
            }
        )

    num_lines = len(line_annotations)
    mean_line_error = total_error / num_lines if num_lines > 0 else 0.0

    return LineErrorsResult(
        num_lines=num_lines,
        mean_line_error=mean_line_error,
        max_line_error=max_error,
        per_line_errors=per_line_errors,
        camera_annotations=camera_annotations,
        camera_reprojected_lines=camera_reprojected_lines,
        map_gcp_lines=map_gcp_lines,
        map_projected_lines=map_projected_lines,
    )
