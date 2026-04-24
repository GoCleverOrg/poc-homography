"""FastAPI router for homography-precision endpoints.

Ported from ``webapp/homography_precision/views.py``.
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import tifffile
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from homography_web.frame_utils import (
    CALIBRATIONS_DIR,
    DATA_MAPS_DIR,
    GCPS_DIR,
    LINES_DIR,
    extract_geotiff,
    get_frame_image_path,
    get_map_from_tenant_id,
    image_filename_to_frame,
    list_frames,
    load_annotations_for_frame,
    load_line_annotations_for_frame,
    normalize_array,
    register_invalidation_callback,
)
from homography_web.frame_utils import (
    get_frame_repo as _get_frame_repo,
)
from line_picker.state import Line, from_line_repo
from PIL import Image

from api.deps import get_current_user
from api.schemas.homography_precision import (
    CameraInfoResponse,
    ComputeHomographyFromLinesRequest,
    ComputeHomographyFromLinesResponse,
    ComputeHomographyRequest,
    ComputeHomographyResponse,
    ComputeLineErrorsRequest,
    ComputeLineErrorsResponse,
    GCPPointOut,
    GCPRegistryResponse,
    HomographyMetrics,
    LineErrorMetrics,
    LineHomographyMetrics,
    LineOut,
    LineRegistryResponse,
    LineTestCaseDetailResponse,
    LineTestCaseListResponse,
    LineTestCaseSummary,
    MapInfoResponse,
    PerLineError,
    PerPointError,
    TestCaseDetailResponse,
    TestCaseListResponse,
    TestCaseSummary,
)
from poc_homography.calibration.lens_distortion.calibration_table import load_calibration_for_camera
from poc_homography.domain.vo import PixelPoint
from poc_homography.homography.map_points import MapPointHomography
from poc_homography.infrastructure.models.user import UserModel
from poc_homography.map_points.gcp_registry import from_gcp_repo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/homography-precision", tags=["homography-precision"])

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------


class ImageInfoCache(TypedDict):
    """Cache entry for image information."""

    width: int
    height: int
    filename: str
    geotransform: list[float] | None
    crs: str | None


_line_registry_cache: dict[str, list[Line]] = {}
_image_info_cache: dict[str, ImageInfoCache] = {}

register_invalidation_callback(_line_registry_cache.clear)
register_invalidation_callback(_image_info_cache.clear)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _no_map_error() -> JSONResponse:
    """Create a 422 'no map configured' error response."""
    return JSONResponse(
        content={
            "success": False,
            "error": "No map configured for the current tenant. Upload a GeoTIFF map first.",
        },
        status_code=422,
    )


def _require_map_id(tenant_id: str) -> str | None:
    """Return the map ID for a tenant, or None if no map is configured."""
    entity = get_map_from_tenant_id(tenant_id)
    return entity.id if entity else None


def _distortion_kwargs(test_case: dict) -> dict[str, float]:
    """Build distortion kwargs for MapPointHomography from a test case.

    Extracts camera_name and zoom from the test case, loads the calibration
    table, and returns interpolated distortion + intrinsic parameters as a dict
    with keys k1,k2,k3,p1,p2,fx,fy,cx,cy.  Returns empty dict if any piece
    is unavailable.
    """
    image_filename = test_case.get("image")
    if not image_filename:
        return {}
    frame = image_filename_to_frame(image_filename)
    if frame is None:
        return {}
    zoom = test_case.get("camera_status", {}).get("zoom")
    if zoom is None:
        return {}

    try:
        table = load_calibration_for_camera(frame.camera_name, CALIBRATIONS_DIR)
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
        logger.debug("Failed to load distortion params for %s", frame.camera_name, exc_info=True)
        return {}


def _get_map_geotiff_file(tenant_id: str) -> Path | None:
    """Return path to the GeoTIFF file for the tenant's map, or None."""
    map_entity = get_map_from_tenant_id(tenant_id)
    if map_entity is None:
        return None
    resolved = DATA_MAPS_DIR / map_entity.photo.path
    if not resolved.exists():
        return None
    return resolved


def _load_line_registry(tenant_id: str) -> list[Line]:
    """Load and cache line registry from DDD repo."""
    if tenant_id in _line_registry_cache:
        return _line_registry_cache[tenant_id]
    map_entity = get_map_from_tenant_id(tenant_id)
    if map_entity is None:
        return []
    lines = from_line_repo(LINES_DIR, map_entity.id)
    _line_registry_cache[tenant_id] = lines
    return lines


def _load_test_case_by_name(name: str, map_id: str | None = None) -> dict | None:
    """Load a test case by name from the CapturedFrame repo.

    Matches ``name`` against each frame's image stem (filename without extension).
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


def _load_line_test_case_by_name(name: str, map_id: str | None = None) -> dict | None:
    """Load a line test case by name from the DDD repos.

    For line test cases the ``name`` convention is ``{image_stem}_lines``.
    We strip the ``_lines`` suffix to find the frame, then load line
    annotations from the LineAnnotation repo.
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


def _get_camera_image_path(test_case_name: str, map_id: str | None = None) -> Path | None:
    """Get the path to the camera image for a test case."""
    test_case = _load_test_case_by_name(test_case_name, map_id)
    if test_case is None:
        return None

    image_filename = test_case.get("image")
    if not image_filename:
        return None

    frame = image_filename_to_frame(image_filename)
    if frame is None:
        return None
    return get_frame_image_path(frame)


def _get_camera_info(test_case_name: str, map_id: str | None = None) -> ImageInfoCache | None:
    """Get cached camera image info, loading if necessary."""
    cache_key = f"camera:{test_case_name}"
    if cache_key in _image_info_cache:
        return _image_info_cache[cache_key]

    image_path = _get_camera_image_path(test_case_name, map_id)
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


def _get_map_info(tenant_id: str) -> ImageInfoCache | None:
    """Get cached map GeoTIFF info, loading if necessary."""
    cache_key = f"map:geotiff:{tenant_id}"
    if cache_key in _image_info_cache:
        return _image_info_cache[cache_key]

    geotiff_path = _get_map_geotiff_file(tenant_id)
    if geotiff_path is None or not geotiff_path.exists():
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


def _perpendicular_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Calculate perpendicular distance from point p to line defined by points a and b."""
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
# Tile helper (shared between camera and map tiles)
# ---------------------------------------------------------------------------


def _compute_tile_bounds(
    width: int,
    height: int,
    x: int,
    y: int,
    z: int,
    size: int,
) -> tuple[int, int, int, int]:
    """Compute clamped tile bounds in original image coordinates.

    Returns (x0, y0, x1, y1).
    """
    max_level = math.ceil(math.log2(max(width, height)))
    level_scale = 2 ** (max_level - z)

    x0 = x * size * level_scale
    y0 = y * size * level_scale
    x1 = (x + 1) * size * level_scale
    y1 = (y + 1) * size * level_scale

    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))

    return x0, y0, x1, y1


def _transparent_tile(size: int) -> bytes:
    """Return a transparent PNG tile of the given size."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/test-cases/", response_model=TestCaseListResponse)
def api_test_cases(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> TestCaseListResponse:
    """List all available test cases with annotation counts."""
    repo = _get_frame_repo()
    map_entity = get_map_from_tenant_id(tenant_id)
    map_id = map_entity.id if map_entity else None
    frames = list_frames(map_id)
    if not frames:
        raise HTTPException(status_code=404, detail="No captured frames found")

    test_cases: list[TestCaseSummary] = []
    for frame in frames:
        annotations = repo.get_annotations(frame.id)
        if not annotations:
            continue
        test_cases.append(
            TestCaseSummary(
                name=frame.image_path.stem,
                image=frame.image_path.name,
                annotation_count=len(annotations),
            )
        )

    return TestCaseListResponse(test_cases=test_cases)


@router.get("/api/test-cases/{name}/", response_model=TestCaseDetailResponse)
def api_test_case_detail(
    name: str,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> TestCaseDetailResponse:
    """Get a specific test case by name."""
    map_id = _require_map_id(tenant_id)
    tc = _load_test_case_by_name(name, map_id)
    if tc is None:
        raise HTTPException(status_code=404, detail=f"Test case not found: {name}")

    return TestCaseDetailResponse(
        name=tc.get("name", ""),
        image=tc.get("image", ""),
        annotations=tc.get("annotations", []),
    )


@router.post("/api/compute-homography/", response_model=ComputeHomographyResponse)
def api_compute_homography(
    body: ComputeHomographyRequest,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> ComputeHomographyResponse | JSONResponse:
    """Compute point-based homography with metrics, per-point errors, and overlays."""
    test_case_name = body.test_case_name

    # Load GCP registry from repository
    map_id = _require_map_id(tenant_id)

    # Load test case
    test_case = _load_test_case_by_name(test_case_name, map_id)
    if test_case is None:
        raise HTTPException(
            status_code=404, detail=f"Test case not found: {test_case_name}"
        )

    annotations = test_case.get("annotations", [])
    if len(annotations) < 4:  # noqa: PLR2004
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 4 annotations, got {len(annotations)}",
        )
    if map_id is None:
        return _no_map_error()
    try:
        registry = from_gcp_repo(GCPS_DIR, map_id)
    except (KeyError, ValueError, OSError) as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load GCP registry: {e}"
        )

    # Compute homography
    try:
        homography = MapPointHomography(map_id=registry.map_id, **_distortion_kwargs(test_case))
        result = homography.compute_from_gcps(
            gcps=annotations,
            map_registry=registry,
            ransac_threshold=50.0,
            min_inlier_ratio=0.5,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=500, detail=f"Homography computation failed: {e}"
        )

    # Compute per-point errors and overlay data
    per_point_errors: list[PerPointError] = []
    camera_annotations: list[dict] = []
    camera_reprojected: list[dict] = []
    map_gcps: list[dict] = []
    map_projected: list[dict] = []

    for annotation in annotations:
        gcp_id = annotation["gcp_id"]

        camera_x = annotation["pixel_x"]
        camera_y = annotation["pixel_y"]
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
            PerPointError(
                gcp_id=gcp_id,
                error_px=round(error_px, 2),
                camera_dx=round(camera_dx, 2),
                camera_dy=round(camera_dy, 2),
                map_dx=round(map_dx, 2),
                map_dy=round(map_dy, 2),
                camera_original=[round(camera_x, 2), round(camera_y, 2)],
                camera_reprojected=[
                    round(reprojected_camera.x, 2),
                    round(reprojected_camera.y, 2),
                ],
                map_original=[round(map_x, 2), round(map_y, 2)],
                map_projected=[round(projected_map.pixel_x, 2), round(projected_map.pixel_y, 2)],
            )
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

    return ComputeHomographyResponse(
        success=True,
        metrics=HomographyMetrics(
            num_gcps=result.num_gcps,
            num_inliers=result.num_inliers,
            inlier_ratio=round(result.inlier_ratio, 2),
            mean_reproj_error=round(result.mean_reproj_error, 2),
            max_reproj_error=round(result.max_reproj_error, 2),
            rmse=round(result.rmse, 2),
        ),
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


@router.get("/api/gcp-registry/", response_model=GCPRegistryResponse)
def api_gcp_registry(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> GCPRegistryResponse | JSONResponse:
    """Get the GCP registry for the tenant's map."""
    map_id = _require_map_id(tenant_id)
    if map_id is None:
        return _no_map_error()
    try:
        registry = from_gcp_repo(GCPS_DIR, map_id)
    except (KeyError, ValueError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to load GCP registry: {e}")

    points_dict = {
        point_id: GCPPointOut(pixel_x=point.pixel_x, pixel_y=point.pixel_y)
        for point_id, point in registry.points.items()
    }

    return GCPRegistryResponse(map_id=registry.map_id, points=points_dict)


@router.get("/api/line-test-cases/", response_model=LineTestCaseListResponse)
def api_line_test_cases(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> LineTestCaseListResponse:
    """List all available line test cases."""
    map_id = _require_map_id(tenant_id)
    test_cases: list[LineTestCaseSummary] = []
    for frame in list_frames(map_id):
        line_anns = load_line_annotations_for_frame(frame.id)
        if not line_anns:
            continue
        test_cases.append(
            LineTestCaseSummary(
                name=f"{frame.image_path.stem}_lines",
                image=frame.image_path.name,
                line_annotation_count=len(line_anns),
            )
        )

    if not test_cases:
        raise HTTPException(status_code=404, detail="No frames with line annotations found")

    return LineTestCaseListResponse(test_cases=test_cases)


@router.get("/api/line-test-cases/{name}/", response_model=LineTestCaseDetailResponse)
def api_line_test_case_detail(
    name: str,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> LineTestCaseDetailResponse:
    """Get a specific line test case by name."""
    map_id = _require_map_id(tenant_id)
    tc = _load_line_test_case_by_name(name, map_id)
    if tc is None:
        raise HTTPException(status_code=404, detail=f"Line test case not found: {name}")

    return LineTestCaseDetailResponse(
        name=tc.get("name", ""),
        image=tc.get("image", ""),
        point_annotations_ref=tc.get("point_annotations_ref", ""),
        line_annotations=tc.get("line_annotations", []),
    )


@router.get("/api/line-registry/", response_model=LineRegistryResponse)
def api_line_registry(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> LineRegistryResponse:
    """Get the line registry for the tenant's map."""
    lines = _load_line_registry(tenant_id)
    if not lines:
        raise HTTPException(status_code=404, detail="No lines found in line registry")

    return LineRegistryResponse(
        map_id=_require_map_id(tenant_id),
        lines=[
            LineOut(
                line_id=line.line_id,
                start_x=line.start_x,
                start_y=line.start_y,
                end_x=line.end_x,
                end_y=line.end_y,
            )
            for line in lines
        ],
    )


@router.post(
    "/api/compute-homography-from-lines/",
    response_model=ComputeHomographyFromLinesResponse,
)
def api_compute_homography_from_lines(
    body: ComputeHomographyFromLinesRequest,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> ComputeHomographyFromLinesResponse | JSONResponse:
    """Compute line-based homography from line correspondences."""
    test_case_name = body.test_case_name
    line_annotations = body.line_annotations

    map_id = _require_map_id(tenant_id)

    line_test_case: dict | None = None
    if test_case_name:
        line_test_case = _load_line_test_case_by_name(test_case_name, map_id)
        if line_test_case is None:
            raise HTTPException(
                status_code=404,
                detail=f"Line test case not found: {test_case_name}",
            )
        line_annotations = line_test_case.get("line_annotations", [])
    elif not line_annotations:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: test_case_name or line_annotations",
        )

    if len(line_annotations) < 2:  # noqa: PLR2004
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 line annotations, got {len(line_annotations)}",
        )

    lines = _load_line_registry(tenant_id)
    if not lines:
        raise HTTPException(status_code=500, detail="No lines found in line registry")

    line_registry = {line.line_id: line.to_dict() for line in lines}

    if map_id is None:
        return _no_map_error()
    try:
        dist_kw = _distortion_kwargs(line_test_case) if test_case_name and line_test_case else {}
        homography = MapPointHomography(map_id=map_id, **dist_kw)
        result = homography.compute_from_lines(
            line_annotations=line_annotations,
            line_registry=line_registry,
            ransac_threshold=50.0,
            min_inlier_ratio=0.3,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Line-based homography computation failed: {e}",
        )

    return ComputeHomographyFromLinesResponse(
        success=True,
        homography_source="lines",
        metrics=LineHomographyMetrics(
            num_lines=result.num_lines,
            num_inliers=result.num_inliers,
            inlier_ratio=round(result.inlier_ratio, 2),
            mean_perp_error=round(result.mean_perp_error, 2),
            max_perp_error=round(result.max_perp_error, 2),
            rmse=round(result.rmse, 2),
        ),
        homography_matrix=result.homography_matrix.tolist(),
    )


@router.post("/api/compute-line-errors/", response_model=ComputeLineErrorsResponse)
def api_compute_line_errors(
    body: ComputeLineErrorsRequest,
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> ComputeLineErrorsResponse | JSONResponse:
    """Compute line errors from a line test case.

    When ``use_line_homography`` is true, computes homography from the line
    annotations themselves instead of requiring ``point_annotations_ref``.
    """
    test_case_name = body.test_case_name
    use_line_homography = body.use_line_homography

    # Load line test case
    map_id = _require_map_id(tenant_id)
    line_test_case = _load_line_test_case_by_name(test_case_name, map_id)
    if line_test_case is None:
        raise HTTPException(
            status_code=404, detail=f"Line test case not found: {test_case_name}"
        )

    line_annotations = line_test_case.get("line_annotations", [])
    if not line_annotations:
        raise HTTPException(
            status_code=400, detail="No line annotations found in test case"
        )

    # Load line registry (needed for both approaches)
    lines = _load_line_registry(tenant_id)
    if not lines:
        raise HTTPException(
            status_code=500, detail="No lines found in line registry"
        )

    line_registry = {line.line_id: line.to_dict() for line in lines}

    # Compute homography - either from lines or from referenced point annotations
    if use_line_homography:
        if len(line_annotations) < 2:  # noqa: PLR2004
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 2 line annotations for line-based homography, got {len(line_annotations)}",
            )

        if map_id is None:
            return _no_map_error()
        try:
            homography = MapPointHomography(
                map_id=map_id, **_distortion_kwargs(line_test_case)
            )
            homography.compute_from_lines(
                line_annotations=line_annotations,
                line_registry=line_registry,
                ransac_threshold=50.0,
                min_inlier_ratio=0.3,
            )
        except (ValueError, RuntimeError) as e:
            raise HTTPException(
                status_code=500,
                detail=f"Line-based homography computation failed: {e}",
            )
    else:
        # Use point-based homography (original behaviour)
        point_annotations_ref = line_test_case.get("point_annotations_ref")
        if not point_annotations_ref:
            raise HTTPException(
                status_code=400,
                detail="No point_annotations_ref found in line test case. Use use_line_homography=true for line-based homography.",
            )

        point_test_case = _load_test_case_by_name(point_annotations_ref, map_id)
        if point_test_case is None:
            raise HTTPException(
                status_code=404,
                detail=f"Referenced point test case not found: {point_annotations_ref}",
            )

        annotations = point_test_case.get("annotations", [])
        if len(annotations) < 4:  # noqa: PLR2004
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 4 point annotations, got {len(annotations)}",
            )

        map_id_for_gcps = _require_map_id(tenant_id)
        if map_id_for_gcps is None:
            return _no_map_error()
        try:
            gcp_registry = from_gcp_repo(GCPS_DIR, map_id_for_gcps)
        except (KeyError, ValueError, OSError) as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load GCP registry: {e}"
            )

        try:
            homography = MapPointHomography(
                map_id=gcp_registry.map_id, **_distortion_kwargs(line_test_case)
            )
            homography.compute_from_gcps(
                gcps=annotations,
                map_registry=gcp_registry,
                ransac_threshold=50.0,
                min_inlier_ratio=0.5,
            )
        except (ValueError, RuntimeError) as e:
            raise HTTPException(
                status_code=500, detail=f"Homography computation failed: {e}"
            )

    # Compute per-line errors
    per_line_errors: list[PerLineError] = []
    camera_annotations_overlay: list[dict] = []
    camera_reprojected_lines: list[dict] = []
    map_gcp_lines: list[dict] = []
    map_projected_lines: list[dict] = []

    total_error = 0.0
    max_error = 0.0

    for line_annotation in line_annotations:
        line_id = line_annotation["line_id"]

        if line_id not in line_registry:
            raise HTTPException(
                status_code=400,
                detail=f"Line {line_id} not found in line registry",
            )

        line_def = line_registry[line_id]

        map_start = np.array([line_def["start_x"], line_def["start_y"]])
        map_end = np.array([line_def["end_x"], line_def["end_y"]])

        camera_start = np.array(
            [line_annotation["start_pixel_x"], line_annotation["start_pixel_y"]]
        )
        camera_end = np.array(
            [line_annotation["end_pixel_x"], line_annotation["end_pixel_y"]]
        )

        projected_start = homography.camera_to_map(
            PixelPoint.create(camera_start[0], camera_start[1])
        )
        projected_end = homography.camera_to_map(
            PixelPoint.create(camera_end[0], camera_end[1])
        )
        projected_start_map = np.array([projected_start.pixel_x, projected_start.pixel_y])
        projected_end_map = np.array([projected_end.pixel_x, projected_end.pixel_y])

        start_error = _perpendicular_distance(projected_start_map, map_start, map_end)
        end_error = _perpendicular_distance(projected_end_map, map_start, map_end)

        reprojected_start = homography.map_to_camera(
            PixelPoint.create(map_start[0], map_start[1])
        )
        reprojected_end = homography.map_to_camera(
            PixelPoint.create(map_end[0], map_end[1])
        )
        reprojected_start_camera = np.array([reprojected_start.x, reprojected_start.y])
        reprojected_end_camera = np.array([reprojected_end.x, reprojected_end.y])

        camera_start_error = _perpendicular_distance(
            camera_start, reprojected_start_camera, reprojected_end_camera
        )
        camera_end_error = _perpendicular_distance(
            camera_end, reprojected_start_camera, reprojected_end_camera
        )

        line_error = (camera_start_error + camera_end_error) / 2.0

        total_error += line_error
        max_error = max(max_error, line_error)

        per_line_errors.append(
            PerLineError(
                line_id=line_id,
                error_px=round(line_error, 2),
                start_error=round(camera_start_error, 2),
                end_error=round(camera_end_error, 2),
                map_start_error=round(start_error, 2),
                map_end_error=round(end_error, 2),
            )
        )

        camera_annotations_overlay.append(
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

    return ComputeLineErrorsResponse(
        success=True,
        metrics=LineErrorMetrics(
            num_lines=num_lines,
            mean_line_error=round(mean_line_error, 2),
            max_line_error=round(max_error, 2),
        ),
        per_line_errors=per_line_errors,
        line_overlays={
            "camera": {
                "annotations": camera_annotations_overlay,
                "reprojected_lines": camera_reprojected_lines,
            },
            "map": {
                "gcp_lines": map_gcp_lines,
                "projected_lines": map_projected_lines,
            },
        },
    )


@router.get("/api/camera-info/", response_model=CameraInfoResponse)
def api_camera_info(
    case: str = Query(...),
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> CameraInfoResponse:
    """Get camera image metadata (width, height, filename)."""
    map_id = _require_map_id(tenant_id)
    info = _get_camera_info(case, map_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Test case not found or image missing: {case}",
        )

    return CameraInfoResponse(
        width=info["width"],
        height=info["height"],
        filename=info["filename"],
    )


@router.get("/api/map-info/", response_model=MapInfoResponse)
def api_map_info(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> MapInfoResponse:
    """Get map GeoTIFF metadata (width, height, filename, geotransform, CRS)."""
    info = _get_map_info(tenant_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Map GeoTIFF not found: {_get_map_geotiff_file(tenant_id)}",
        )

    return MapInfoResponse(
        width=info["width"],
        height=info["height"],
        filename=info["filename"],
        geotransform=info["geotransform"],
        crs=info["crs"],
    )


@router.get("/api/camera-tile/")
def api_camera_tile(
    case: str = Query(...),
    x: int = Query(0),
    y: int = Query(0),
    z: int = Query(0),
    size: int = Query(256),
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> Response:
    """Get a camera image tile at specified coordinates and zoom level.

    OpenSeadragon uses a pyramid where level 0 is most zoomed out and
    the max level is full resolution.
    """
    map_id = _require_map_id(tenant_id)
    info = _get_camera_info(case, map_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Test case not found or image missing: {case}",
        )

    image_path = _get_camera_image_path(case, map_id)
    if image_path is None or not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Camera image not found for test case: {case}",
        )

    width = info["width"]
    height = info["height"]

    x0, y0, x1, y1 = _compute_tile_bounds(width, height, x, y, z, size)

    if x1 <= x0 or y1 <= y0:
        return Response(content=_transparent_tile(size), media_type="image/png")

    with Image.open(image_path) as full_img:
        img = full_img.crop((x0, y0, x1, y1))
        if img.mode != "RGB":
            img = img.convert("RGB")

    img = img.resize((size, size), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@router.get("/api/map-tile/")
def api_map_tile(
    x: int = Query(0),
    y: int = Query(0),
    z: int = Query(0),
    size: int = Query(256),
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> Response:
    """Get a map GeoTIFF tile at specified coordinates and zoom level.

    OpenSeadragon uses a pyramid where level 0 is most zoomed out and
    the max level is full resolution.
    """
    info = _get_map_info(tenant_id)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Map GeoTIFF not found: {_get_map_geotiff_file(tenant_id)}",
        )

    width = info["width"]
    height = info["height"]

    x0, y0, x1, y1 = _compute_tile_bounds(width, height, x, y, z, size)

    if x1 <= x0 or y1 <= y0:
        return Response(content=_transparent_tile(size), media_type="image/png")

    geotiff_path = _get_map_geotiff_file(tenant_id)
    with tifffile.TiffFile(geotiff_path) as tif:
        page = tif.pages[0]
        data = page.asarray()

        if data.ndim == 2:  # noqa: PLR2004
            tile_data = data[y0:y1, x0:x1]
            img = Image.fromarray(normalize_array(tile_data), mode="L")
            img = img.convert("RGB")
        elif data.ndim == 3:  # noqa: PLR2004
            if data.shape[2] >= 3:  # noqa: PLR2004
                tile_data = data[y0:y1, x0:x1, :3]
                img = Image.fromarray(normalize_array(tile_data), mode="RGB")
            else:
                tile_data = data[y0:y1, x0:x1, 0]
                img = Image.fromarray(normalize_array(tile_data), mode="L")
                img = img.convert("RGB")
        else:
            if data.shape[0] in (1, 3, 4):
                if data.shape[0] == 1:
                    tile_data = data[0, y0:y1, x0:x1]
                    img = Image.fromarray(normalize_array(tile_data), mode="L")
                    img = img.convert("RGB")
                else:
                    tile_data = np.transpose(data[:3, y0:y1, x0:x1], (1, 2, 0))
                    img = Image.fromarray(normalize_array(tile_data), mode="RGB")
            else:
                tile_data = (
                    data[y0:y1, x0:x1]
                    if data.ndim == 2  # noqa: PLR2004
                    else data[y0:y1, x0:x1, 0]
                )
                img = Image.fromarray(normalize_array(tile_data), mode="L")
                img = img.convert("RGB")

    img = img.resize((size, size), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
