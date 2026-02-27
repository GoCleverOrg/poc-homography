"""
Business logic for homography precision computation.

Contains homography computation, metric aggregation, and data loading logic
extracted from views.py. No HTTP/Django concerns live here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import tifffile
from homography_web.frame_utils import (
    GCPS_DIR,
    LINES_DIR,
    LineAnnotationDTO,
    PointAnnotationDTO,
    extract_geotiff,
    get_frame_image_path,
    get_map_from_tenant_id,
    image_filename_to_frame,
    list_frames,
    load_annotations_for_frame,
    load_line_annotations_for_frame,
    register_invalidation_callback,
    resolve_map_for_tenant,
)
from homography_web.frame_utils import (
    get_frame_repo as _get_frame_repo,
)
from line_picker.state import Line, from_line_repo
from PIL import Image

from poc_homography.domain.vo import PixelPoint
from poc_homography.homography.map_points import MapPointHomography
from poc_homography.map_points.gcp_registry import from_gcp_repo

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ImageInfoCache(TypedDict):
    """Cache entry for image information."""

    width: int
    height: int
    filename: str
    geotransform: list[float] | None
    crs: str | None


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_line_registry_cache: dict[str, list[Line]] = {}
_image_info_cache: dict[str, ImageInfoCache] = {}

register_invalidation_callback(_line_registry_cache.clear)
register_invalidation_callback(_image_info_cache.clear)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointHomographyResult:
    """Result of GCP-based homography computation with overlays."""

    metrics: dict[str, Any]
    per_point_errors: list[dict[str, Any]]
    overlays: dict[str, Any]


@dataclass(frozen=True)
class LineHomographyResult:
    """Result of line-based homography computation."""

    metrics: dict[str, Any]
    homography_matrix: list[list[float]]


@dataclass(frozen=True)
class LineErrorResult:
    """Result of line error computation with overlays."""

    metrics: dict[str, Any]
    per_line_errors: list[dict[str, Any]]
    line_overlays: dict[str, Any]


@dataclass(frozen=True)
class TestCaseInfo:
    """Summary info for a test case listing."""

    name: str
    image: str
    annotation_count: int


@dataclass(frozen=True)
class LineTestCaseInfo:
    """Summary info for a line test case listing."""

    name: str
    image: str
    line_annotation_count: int


# ---------------------------------------------------------------------------
# Tile computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileBounds:
    """Computed tile bounds in original image coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int
    is_empty: bool


def compute_tile_bounds(
    width: int,
    height: int,
    x: int,
    y: int,
    z: int,
    size: int = 256,
) -> TileBounds:
    """Compute tile bounds in original image coordinates for an OpenSeadragon tile.

    Args:
        width: Full image width in pixels.
        height: Full image height in pixels.
        x: Tile x coordinate.
        y: Tile y coordinate.
        z: Zoom level.
        size: Tile size (default 256).

    Returns:
        TileBounds with clamped coordinates and emptiness flag.
    """
    max_level = math.ceil(math.log2(max(width, height)))
    level_scale = 2 ** (max_level - z)

    x0 = x * size * level_scale
    y0 = y * size * level_scale
    x1 = (x + 1) * size * level_scale
    y1 = (y + 1) * size * level_scale

    # Clamp to image bounds
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))

    return TileBounds(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        is_empty=(x1 <= x0 or y1 <= y0),
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def require_map_id(tenant_id: str) -> str | None:
    """Return the map ID for a tenant, or None if no map is configured."""
    entity = get_map_from_tenant_id(tenant_id)
    return entity.id if entity else None


def resolve_map_file(tenant_id: str) -> Path | None:
    """Return path to the map image file for the tenant, or None.

    Thin wrapper around :func:`resolve_map_for_tenant` that converts the
    ``RuntimeError`` (missing map / missing file) into a ``None`` return,
    which callers use to build user-friendly error responses.
    """
    try:
        _entity, path = resolve_map_for_tenant(tenant_id)
    except RuntimeError:
        return None
    return path


def load_line_registry(tenant_id: str) -> list[Line]:
    """Load and cache line registry from DDD repo."""
    if tenant_id in _line_registry_cache:
        return _line_registry_cache[tenant_id]
    map_entity = get_map_from_tenant_id(tenant_id)
    if map_entity is None:
        return []
    lines = from_line_repo(LINES_DIR, map_entity.id)
    _line_registry_cache[tenant_id] = lines
    return lines


def load_test_case_by_name(name: str, map_id: str | None = None) -> dict | None:
    """Load a test case by name from the CapturedFrame repo.

    Matches ``name`` against each frame's image stem (filename without extension).

    Args:
        name: Name of the test case (image filename stem).
        map_id: If provided, only search frames belonging to this map.

    Returns:
        Legacy-format test case dict if found, None otherwise.
    """
    for frame in list_frames(map_id):
        if frame.image_path.stem != name:
            continue
        annotations = load_annotations_for_frame(frame.id)
        if not annotations:
            continue
        tc: dict = {
            "name": frame.image_path.stem,
            "image": frame.image_path.name,
            "annotations": annotations,
            "camera_status": {
                "pan": float(frame.ptz_state.pan_raw),
                "tilt": float(frame.ptz_state.tilt_deg),
                "zoom": float(frame.ptz_state.zoom),
            },
        }
        return tc
    return None


def load_line_test_case_by_name(name: str, map_id: str | None = None) -> dict | None:
    """Load a line test case by name from the DDD repos.

    For line test cases the ``name`` convention is ``{image_stem}_lines``.
    We strip the ``_lines`` suffix to find the frame, then load line
    annotations from the LineAnnotation repo.

    Args:
        name: Name of the line test case (e.g. ``valte_102.5_20.7_1_20260115_112639_lines``).
        map_id: If provided, only search frames belonging to this map.

    Returns:
        Legacy-format line test case dict if found, None otherwise.
    """
    stem = name.removesuffix("_lines") if name.endswith("_lines") else name

    for frame in list_frames(map_id):
        if frame.image_path.stem != stem:
            continue
        line_anns = load_line_annotations_for_frame(frame.id)
        if not line_anns:
            continue
        point_anns = load_annotations_for_frame(frame.id)
        point_annotations_ref = frame.image_path.stem if point_anns else ""
        return {
            "name": name,
            "image": frame.image_path.name,
            "camera_status": {
                "pan": float(frame.ptz_state.pan_raw),
                "tilt": float(frame.ptz_state.tilt_deg),
                "zoom": float(frame.ptz_state.zoom),
            },
            "point_annotations_ref": point_annotations_ref,
            "line_annotations": line_anns,
        }
    return None


def get_camera_image_path(test_case_name: str, map_id: str | None = None) -> Path | None:
    """Get the path to the camera image for a test case.

    Args:
        test_case_name: Name of the test case.
        map_id: If provided, only search frames belonging to this map.

    Returns:
        Path to the camera image, or None if not found.
    """
    test_case = load_test_case_by_name(test_case_name, map_id)
    if test_case is None:
        return None

    image_filename = test_case.get("image")
    if not image_filename:
        return None

    frame = image_filename_to_frame(image_filename)
    if frame is None:
        return None
    return get_frame_image_path(frame)


def get_camera_info(test_case_name: str, map_id: str | None = None) -> ImageInfoCache | None:
    """Get cached camera image info, loading if necessary.

    Args:
        test_case_name: Name of the test case.
        map_id: If provided, only search frames belonging to this map.

    Returns:
        Image info dict or None if not found.
    """
    cache_key = f"camera:{test_case_name}"
    if cache_key in _image_info_cache:
        return _image_info_cache[cache_key]

    image_path = get_camera_image_path(test_case_name, map_id)
    if image_path is None or not image_path.exists():
        return None

    with Image.open(image_path) as img:
        info: ImageInfoCache = {
            "width": img.width,
            "height": img.height,
            "filename": image_path.name,
            "geotransform": None,
            "crs": None,
        }

    _image_info_cache[cache_key] = info
    return info


def get_map_info(tenant_id: str) -> ImageInfoCache | None:
    """Get cached map GeoTIFF info, loading if necessary.

    Args:
        tenant_id: Tenant identifier for map lookup.

    Returns:
        Image info dict or None if not found.
    """
    cache_key = f"map:geotiff:{tenant_id}"
    if cache_key in _image_info_cache:
        return _image_info_cache[cache_key]

    geotiff_path = resolve_map_file(tenant_id)
    if geotiff_path is None:
        return None

    with tifffile.TiffFile(geotiff_path) as tif:
        page = tif.pages[0]
        width: int = page.imagewidth  # type: ignore[union-attr]
        height: int = page.imagelength  # type: ignore[union-attr]
        geotiff = extract_geotiff(tif)

        info: ImageInfoCache = {
            "width": width,
            "height": height,
            "filename": geotiff_path.name,
            "geotransform": geotiff.geotransform.to_list() if geotiff else None,
            "crs": geotiff.crs if geotiff else None,
        }

    _image_info_cache[cache_key] = info
    return info


def list_test_cases(map_id: str | None) -> list[TestCaseInfo]:
    """List all test cases that have annotations.

    Args:
        map_id: If provided, only list frames belonging to this map.

    Returns:
        List of TestCaseInfo for each frame with annotations.
    """
    repo = _get_frame_repo()
    result = []
    for frame in list_frames(map_id):
        annotations = repo.get_annotations(frame.id)
        if not annotations:
            continue
        result.append(
            TestCaseInfo(
                name=frame.image_path.stem,
                image=frame.image_path.name,
                annotation_count=len(annotations),
            )
        )
    return result


def list_line_test_cases(map_id: str | None) -> list[LineTestCaseInfo]:
    """List all frames that have line annotations.

    Args:
        map_id: If provided, only list frames belonging to this map.

    Returns:
        List of LineTestCaseInfo for each frame with line annotations.
    """
    result = []
    for frame in list_frames(map_id):
        line_anns = load_line_annotations_for_frame(frame.id)
        if not line_anns:
            continue
        result.append(
            LineTestCaseInfo(
                name=f"{frame.image_path.stem}_lines",
                image=frame.image_path.name,
                line_annotation_count=len(line_anns),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def perpendicular_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Calculate perpendicular distance from point p to line defined by points a and b.

    Args:
        p: Point as numpy array [x, y].
        a: Line start point as numpy array [x, y].
        b: Line end point as numpy array [x, y].

    Returns:
        Perpendicular distance in pixels.
    """
    v = b - a
    w = p - a

    c1 = np.dot(w, v)
    c2 = np.dot(v, v)

    if c2 == 0:
        return float(np.linalg.norm(w))

    b_param = c1 / c2
    proj = a + b_param * v
    return float(np.linalg.norm(p - proj))


# ---------------------------------------------------------------------------
# Homography computation
# ---------------------------------------------------------------------------


def compute_point_homography(
    annotations: list[PointAnnotationDTO],
    map_id: str,
) -> PointHomographyResult:
    """Compute homography from GCP annotations and return metrics with overlays.

    Args:
        annotations: List of PointAnnotationDTO instances.
        map_id: Map identifier for GCP registry lookup.

    Returns:
        PointHomographyResult with metrics, per-point errors, and overlay data.

    Raises:
        ValueError: If fewer than 4 annotations or GCP registry cannot be loaded.
        RuntimeError: If homography computation fails.
        KeyError: If GCP registry lookup fails.
        OSError: If GCP registry file cannot be read.
    """
    registry = from_gcp_repo(GCPS_DIR, map_id)

    homography = MapPointHomography(map_id=registry.map_id)
    result = homography.compute_from_gcps(
        gcps=[a.to_dict() for a in annotations],
        map_registry=registry,
        ransac_threshold=50.0,
        min_inlier_ratio=0.5,
    )

    # Compute per-point errors and overlay data
    per_point_errors: list[dict[str, Any]] = []
    camera_annotations: list[dict[str, Any]] = []
    camera_reprojected: list[dict[str, Any]] = []
    map_gcps: list[dict[str, Any]] = []
    map_projected: list[dict[str, Any]] = []

    for annotation in annotations:
        gcp_id = annotation.gcp_id
        camera_x = annotation.pixel_x
        camera_y = annotation.pixel_y
        camera_pixel = PixelPoint.create(camera_x, camera_y)

        gcp = registry.points[gcp_id]
        map_x = gcp.pixel_x
        map_y = gcp.pixel_y

        projected_map = homography.camera_to_map(camera_pixel)

        gcp_coord = PixelPoint.create(map_x, map_y)
        reprojected_camera = homography.map_to_camera(gcp_coord)

        original = np.array([camera_x, camera_y])
        reprojected = np.array([reprojected_camera.x, reprojected_camera.y])
        error_px = float(np.linalg.norm(reprojected - original))

        camera_dx = reprojected_camera.x - camera_x
        camera_dy = reprojected_camera.y - camera_y
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
                "map_projected": [
                    round(projected_map.pixel_x, 2),
                    round(projected_map.pixel_y, 2),
                ],
            }
        )

        camera_annotations.append(
            {"gcp_id": gcp_id, "x": round(camera_x, 2), "y": round(camera_y, 2)}
        )
        camera_reprojected.append(
            {
                "gcp_id": gcp_id,
                "x": round(reprojected_camera.x, 2),
                "y": round(reprojected_camera.y, 2),
            }
        )
        map_gcps.append(
            {"gcp_id": gcp_id, "x": round(map_x, 2), "y": round(map_y, 2)}
        )
        map_projected.append(
            {
                "gcp_id": gcp_id,
                "x": round(projected_map.pixel_x, 2),
                "y": round(projected_map.pixel_y, 2),
            }
        )

    return PointHomographyResult(
        metrics={
            "num_gcps": result.num_gcps,
            "num_inliers": result.num_inliers,
            "inlier_ratio": round(result.inlier_ratio, 2),
            "mean_reproj_error": round(result.mean_reproj_error, 2),
            "max_reproj_error": round(result.max_reproj_error, 2),
            "rmse": round(result.rmse, 2),
        },
        per_point_errors=per_point_errors,
        overlays={
            "camera": {
                "annotations": camera_annotations,
                "reprojected_gcps": camera_reprojected,
            },
            "map": {
                "gcps": map_gcps,
                "projected_annotations": map_projected,
            },
        },
    )


def compute_line_homography(
    line_annotations: list[dict[str, Any]],
    line_registry: dict[str, dict[str, Any]],
    map_id: str,
) -> LineHomographyResult:
    """Compute homography from line correspondences.

    Args:
        line_annotations: List of line annotation dicts (already serialised).
        line_registry: Dict mapping line_id to line definition dicts.
        map_id: Map identifier.

    Returns:
        LineHomographyResult with metrics and homography matrix.

    Raises:
        ValueError: If computation parameters are invalid.
        RuntimeError: If homography computation fails.
    """
    homography = MapPointHomography(map_id=map_id)
    result = homography.compute_from_lines(
        line_annotations=line_annotations,
        line_registry=line_registry,
        ransac_threshold=50.0,
        min_inlier_ratio=0.3,
    )

    return LineHomographyResult(
        metrics={
            "num_lines": result.num_lines,
            "num_inliers": result.num_inliers,
            "inlier_ratio": round(result.inlier_ratio, 2),
            "mean_perp_error": round(result.mean_perp_error, 2),
            "max_perp_error": round(result.max_perp_error, 2),
            "rmse": round(result.rmse, 2),
        },
        homography_matrix=result.homography_matrix.tolist(),
    )


def compute_line_errors(
    line_annotations: list[LineAnnotationDTO],
    line_registry: dict[str, dict[str, Any]],
    homography: MapPointHomography,
) -> LineErrorResult:
    """Compute per-line errors using a pre-computed homography.

    Args:
        line_annotations: List of LineAnnotationDTO instances.
        line_registry: Dict mapping line_id to line definition dicts.
        homography: A MapPointHomography instance with a computed homography.

    Returns:
        LineErrorResult with metrics, per-line errors, and overlay data.

    Raises:
        KeyError: If a line_id from annotations is not found in the registry.
    """
    per_line_errors: list[dict[str, Any]] = []
    camera_annotations: list[dict[str, Any]] = []
    camera_reprojected_lines: list[dict[str, Any]] = []
    map_gcp_lines: list[dict[str, Any]] = []
    map_projected_lines: list[dict[str, Any]] = []

    total_error = 0.0
    max_error = 0.0

    for line_annotation in line_annotations:
        line_id = line_annotation.line_id

        if line_id not in line_registry:
            msg = f"Line {line_id} not found in line registry"
            raise KeyError(msg)

        line_def = line_registry[line_id]

        # Ground truth line in map coordinates
        map_start = np.array([line_def["start_x"], line_def["start_y"]])
        map_end = np.array([line_def["end_x"], line_def["end_y"]])

        # Annotated line in camera coordinates
        camera_start = np.array(
            [line_annotation.start_pixel_x, line_annotation.start_pixel_y]
        )
        camera_end = np.array(
            [line_annotation.end_pixel_x, line_annotation.end_pixel_y]
        )

        # Project camera line endpoints to map
        projected_start = homography.camera_to_map(
            PixelPoint.create(camera_start[0], camera_start[1])
        )
        projected_end = homography.camera_to_map(
            PixelPoint.create(camera_end[0], camera_end[1])
        )
        projected_start_map = np.array([projected_start.pixel_x, projected_start.pixel_y])
        projected_end_map = np.array([projected_end.pixel_x, projected_end.pixel_y])

        # Perpendicular distance from projected points to ground truth line
        start_error = perpendicular_distance(projected_start_map, map_start, map_end)
        end_error = perpendicular_distance(projected_end_map, map_start, map_end)

        # Reverse: project GCP line to camera and compare
        reprojected_start = homography.map_to_camera(
            PixelPoint.create(map_start[0], map_start[1])
        )
        reprojected_end = homography.map_to_camera(
            PixelPoint.create(map_end[0], map_end[1])
        )
        reprojected_start_camera = np.array([reprojected_start.x, reprojected_start.y])
        reprojected_end_camera = np.array([reprojected_end.x, reprojected_end.y])

        # Errors in camera space
        camera_start_error = perpendicular_distance(
            camera_start, reprojected_start_camera, reprojected_end_camera
        )
        camera_end_error = perpendicular_distance(
            camera_end, reprojected_start_camera, reprojected_end_camera
        )

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

        # Overlay data for camera frame
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
                "end": [
                    round(reprojected_end_camera[0], 2),
                    round(reprojected_end_camera[1], 2),
                ],
            }
        )

        # Overlay data for map frame
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

    return LineErrorResult(
        metrics={
            "num_lines": num_lines,
            "mean_line_error": round(mean_line_error, 2),
            "max_line_error": round(max_error, 2),
        },
        per_line_errors=per_line_errors,
        line_overlays={
            "camera": {
                "annotations": camera_annotations,
                "reprojected_lines": camera_reprojected_lines,
            },
            "map": {
                "gcp_lines": map_gcp_lines,
                "projected_lines": map_projected_lines,
            },
        },
    )


def build_homography_from_lines(
    line_annotations: list[dict[str, Any]],
    line_registry: dict[str, dict[str, Any]],
    map_id: str,
) -> MapPointHomography:
    """Build a MapPointHomography from line annotations.

    Args:
        line_annotations: List of line annotation dicts (already serialised).
        line_registry: Dict mapping line_id to line definition dicts.
        map_id: Map identifier.

    Returns:
        MapPointHomography with computed homography.

    Raises:
        ValueError: If computation parameters are invalid.
        RuntimeError: If homography computation fails.
    """
    homography = MapPointHomography(map_id=map_id)
    homography.compute_from_lines(
        line_annotations=line_annotations,
        line_registry=line_registry,
        ransac_threshold=50.0,
        min_inlier_ratio=0.3,
    )
    return homography


def build_homography_from_points(
    annotations: list[PointAnnotationDTO],
    map_id: str,
) -> MapPointHomography:
    """Build a MapPointHomography from point (GCP) annotations.

    Args:
        annotations: List of PointAnnotationDTO instances.
        map_id: Map identifier for GCP registry lookup.

    Returns:
        MapPointHomography with computed homography.

    Raises:
        ValueError: If computation parameters are invalid.
        RuntimeError: If homography computation fails.
        KeyError: If GCP registry lookup fails.
        OSError: If GCP registry file cannot be read.
    """
    registry = from_gcp_repo(GCPS_DIR, map_id)
    homography = MapPointHomography(map_id=registry.map_id)
    homography.compute_from_gcps(
        gcps=[a.to_dict() for a in annotations],
        map_registry=registry,
        ransac_threshold=50.0,
        min_inlier_ratio=0.5,
    )
    return homography
