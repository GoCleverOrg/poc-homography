"""
Views for homography precision visualization.

Django view functions for rendering precision visualizations and API endpoints.
"""

from __future__ import annotations

import io
import json
import logging
import math
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import tifffile
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from homography_web.frame_utils import (
    CALIBRATIONS_DIR,
    DATA_MAPS_DIR,
    GCPS_DIR,
    LINES_DIR,
    extract_geotiff,
    get_frame_image_path,
    get_map_from_tenant_id,
    get_tenant_id,
    image_filename_to_frame,
    list_frames,
    load_annotations_for_frame,
    load_line_annotations_for_frame,
    register_invalidation_callback,
)
from homography_web.frame_utils import (
    get_frame_repo as _get_frame_repo,
)
from homography_web.frame_utils import (
    normalize_array as _normalize_array,
)
from line_picker.state import Line, from_line_repo
from PIL import Image

from poc_homography.application.homography_precision import (
    LineNotInRegistryError,
    build_distortion_params,
    compute_gcp_precision,
    compute_line_errors,
    compute_line_homography,
)
from poc_homography.homography.map_points import MapPointHomography
from poc_homography.map_points.gcp_registry import from_gcp_repo

logger = logging.getLogger(__name__)


def _no_map_error() -> JsonResponse:
    """Create a fresh 'no map configured' error response."""
    return JsonResponse(
        {
            "success": False,
            "error": "No map configured for the current tenant. Upload a GeoTIFF map first.",
        },
        status=422,
    )


def _require_map_id(tenant_id: str) -> str | None:
    """Return the map ID for a tenant, or None if no map is configured."""
    entity = get_map_from_tenant_id(tenant_id)
    return entity.id if entity else None


def _distortion_kwargs(test_case: dict) -> dict[str, float]:
    """Resolve camera_name + zoom from a test case and build distortion kwargs.

    Returns an empty dict if the test case lacks an image, the frame cannot be
    resolved, or no zoom is present; otherwise delegates to the application-layer
    ``build_distortion_params`` service.
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
    return build_distortion_params(frame.camera_name, zoom, CALIBRATIONS_DIR)


def _get_map_geotiff_file(tenant_id: str) -> Path | None:
    """Return path to the GeoTIFF file for the tenant's map, or None."""
    map_entity = get_map_from_tenant_id(tenant_id)
    if map_entity is None:
        return None
    resolved = DATA_MAPS_DIR / map_entity.photo.path
    if not resolved.exists():
        return None
    return resolved


# Per-tenant cached line registry
_line_registry_cache: dict[str, list[Line]] = {}


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


# Cache for image dimensions to avoid repeatedly opening files
class ImageInfoCache(TypedDict):
    """Cache entry for image information."""

    width: int
    height: int
    filename: str
    geotransform: list[float] | None
    crs: str | None


_image_info_cache: dict[str, ImageInfoCache] = {}

register_invalidation_callback(_line_registry_cache.clear)
register_invalidation_callback(_image_info_cache.clear)


def _get_camera_image_path(test_case_name: str, map_id: str | None = None) -> Path | None:
    """Get the path to the camera image for a test case.

    Args:
        test_case_name: Name of the test case.
        map_id: If provided, only search frames belonging to this map.

    Returns:
        Path to the camera image, or None if not found.
    """
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

    image_path = _get_camera_image_path(test_case_name, map_id)
    if image_path is None or not image_path.exists():
        return None

    # Load image dimensions
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
    """Get cached map GeoTIFF info, loading if necessary.

    Args:
        tenant_id: Tenant identifier for map lookup.

    Returns:
        Image info dict or None if not found.
    """
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


def index(request: HttpRequest) -> HttpResponse:
    """Landing page for homography precision visualization."""
    return render(
        request,
        "homography_precision/index.html",
        {"title": "Homography Precision Visualization"},
    )


@require_GET
def api_test_cases(request: HttpRequest) -> JsonResponse:
    """
    List all available test cases.

    GET /api/test-cases/

    Returns:
        JSON with list of test cases:
        {"test_cases": [{"name": "...", "image": "...", "annotation_count": N}, ...]}
    """
    repo = _get_frame_repo()
    map_entity = get_map_from_tenant_id(get_tenant_id(request))
    map_id = map_entity.id if map_entity else None
    frames = list_frames(map_id)
    if not frames:
        return JsonResponse({"error": "No captured frames found"}, status=404)

    test_cases = []
    for frame in frames:
        annotations = repo.get_annotations(frame.id)
        if not annotations:
            continue
        test_cases.append(
            {
                "name": frame.image_path.stem,
                "image": frame.image_path.name,
                "annotation_count": len(annotations),
            }
        )

    return JsonResponse({"test_cases": test_cases})


@require_GET
def api_test_case_detail(request: HttpRequest, name: str) -> JsonResponse:
    """
    Get a specific test case by name.

    GET /api/test-cases/{name}/

    Returns:
        JSON with full test case data:
        {"name": "...", "image": "...", "annotations": [{...}, ...]}
    """
    map_id = _require_map_id(get_tenant_id(request))
    tc = _load_test_case_by_name(name, map_id)
    if tc is None:
        return JsonResponse(
            {"error": f"Test case not found: {name}"},
            status=404,
        )

    return JsonResponse(
        {
            "name": tc.get("name", ""),
            "image": tc.get("image", ""),
            "annotations": tc.get("annotations", []),
        }
    )


def _load_test_case_by_name(name: str, map_id: str | None = None) -> dict | None:
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
        annotations = [a.to_dict() for a in load_annotations_for_frame(frame.id)]
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

    Args:
        name: Name of the line test case (e.g. ``valte_102.5_20.7_1_20260115_112639_lines``).
        map_id: If provided, only search frames belonging to this map.

    Returns:
        Legacy-format line test case dict if found, None otherwise.
    """
    # Strip trailing _lines suffix if present
    stem = name.removesuffix("_lines") if name.endswith("_lines") else name

    for frame in list_frames(map_id):
        if frame.image_path.stem != stem:
            continue
        line_anns = [a.to_dict() for a in load_line_annotations_for_frame(frame.id)]
        if not line_anns:
            continue
        # Find the point-annotations reference (same frame, for point-based homography)
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


@csrf_exempt
@require_http_methods(["POST"])
def api_compute_homography(request: HttpRequest) -> JsonResponse:
    """
    Compute homography from a test case.

    POST /api/compute-homography/

    Request body:
        {"test_case_name": "valte_30.8_13.1_1"}

    Returns:
        JSON with homography computation results including metrics, per-point errors,
        and overlay data for visualization.
    """
    # Parse request body
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"success": False, "error": f"Invalid JSON: {e}"},
            status=400,
        )

    test_case_name = body.get("test_case_name")
    if not test_case_name:
        return JsonResponse(
            {"success": False, "error": "Missing required field: test_case_name"},
            status=400,
        )

    # Load GCP registry from repository
    tenant_id = get_tenant_id(request)
    map_id = _require_map_id(tenant_id)

    # Load test case
    test_case = _load_test_case_by_name(test_case_name, map_id)
    if test_case is None:
        return JsonResponse(
            {"success": False, "error": f"Test case not found: {test_case_name}"},
            status=404,
        )

    annotations = test_case.get("annotations", [])
    if len(annotations) < 4:
        return JsonResponse(
            {"success": False, "error": f"Need at least 4 annotations, got {len(annotations)}"},
            status=400,
        )
    if map_id is None:
        return _no_map_error()
    try:
        registry = from_gcp_repo(GCPS_DIR, map_id)
    except (KeyError, ValueError, OSError) as e:
        return JsonResponse(
            {"success": False, "error": f"Failed to load GCP registry: {e}"},
            status=500,
        )

    # Compute homography + aggregate per-point errors, overlays and metrics
    try:
        result = compute_gcp_precision(
            annotations=annotations,
            registry=registry,
            distortion=_distortion_kwargs(test_case),
            ransac_threshold=50.0,
            min_inlier_ratio=0.5,
        )
    except (ValueError, RuntimeError) as e:
        return JsonResponse(
            {"success": False, "error": f"Homography computation failed: {e}"},
            status=500,
        )

    return JsonResponse(result.to_payload())


@require_GET
def api_gcp_registry(request: HttpRequest) -> JsonResponse:
    """
    Get the GCP registry.

    GET /api/gcp-registry/

    Returns:
        JSON with registry data:
        {"map_id": "...", "points": {"PS1": {"pixel_x": ..., "pixel_y": ...}, ...}}
    """
    tenant_id = get_tenant_id(request)
    map_id = _require_map_id(tenant_id)
    if map_id is None:
        return _no_map_error()
    try:
        registry = from_gcp_repo(GCPS_DIR, map_id)
    except (KeyError, ValueError, OSError) as e:
        return JsonResponse(
            {"error": f"Failed to load GCP registry: {e}"},
            status=500,
        )

    # Convert to the expected format with points as a dictionary keyed by ID
    points_dict = {
        point_id: {"pixel_x": point.pixel_x, "pixel_y": point.pixel_y}
        for point_id, point in registry.points.items()
    }

    return JsonResponse(
        {
            "map_id": registry.map_id,
            "points": points_dict,
        }
    )


@require_GET
def api_line_test_cases(request: HttpRequest) -> JsonResponse:
    """
    List all available line test cases.

    GET /api/line-test-cases/

    Returns:
        JSON with list of line test cases:
        {"test_cases": [{"name": "...", "image": "...", "line_annotation_count": N}, ...]}
    """
    map_id = _require_map_id(get_tenant_id(request))
    test_cases = []
    for frame in list_frames(map_id):
        line_anns = load_line_annotations_for_frame(frame.id)
        if not line_anns:
            continue
        test_cases.append(
            {
                "name": f"{frame.image_path.stem}_lines",
                "image": frame.image_path.name,
                "line_annotation_count": len(line_anns),
            }
        )

    if not test_cases:
        return JsonResponse(
            {"error": "No frames with line annotations found"},
            status=404,
        )

    return JsonResponse({"test_cases": test_cases})


@require_GET
def api_line_test_case_detail(request: HttpRequest, name: str) -> JsonResponse:
    """
    Get a specific line test case by name.

    GET /api/line-test-cases/{name}/

    Returns:
        JSON with full line test case data including point_annotations_ref:
        {"name": "...", "image": "...", "point_annotations_ref": "...", "line_annotations": [{...}, ...]}
    """
    map_id = _require_map_id(get_tenant_id(request))
    tc = _load_line_test_case_by_name(name, map_id)
    if tc is None:
        return JsonResponse(
            {"error": f"Line test case not found: {name}"},
            status=404,
        )

    return JsonResponse(
        {
            "name": tc.get("name", ""),
            "image": tc.get("image", ""),
            "point_annotations_ref": tc.get("point_annotations_ref", ""),
            "line_annotations": tc.get("line_annotations", []),
        }
    )


@require_GET
def api_line_registry(request: HttpRequest) -> JsonResponse:
    """
    Get the line registry.

    GET /api/line-registry/

    Returns:
        JSON with line registry data:
        {"map_id": "...", "lines": [{"line_id": "L1", "start_x": 100.0, "start_y": 200.0, "end_x": 300.0, "end_y": 400.0}, ...]}
    """
    tenant_id = get_tenant_id(request)
    lines = _load_line_registry(tenant_id)
    if not lines:
        return JsonResponse(
            {"error": "No lines found in line registry"},
            status=404,
        )

    return JsonResponse(
        {
            "map_id": _require_map_id(tenant_id),
            "lines": [line.to_dict() for line in lines],
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def api_compute_homography_from_lines(request: HttpRequest) -> JsonResponse:
    """
    Compute homography from line correspondences.

    POST /api/compute-homography-from-lines/

    Request body:
        {"test_case_name": "valte_102.5_20.7_1_20260115_lines"}
        OR
        {"line_annotations": [...], "line_registry": {...}}

    Returns:
        JSON with homography computation results using line-based approach.
    """
    # Parse request body
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"success": False, "error": f"Invalid JSON: {e}"},
            status=400,
        )

    # Determine input mode: test case name or direct annotations
    test_case_name = body.get("test_case_name")
    line_annotations = body.get("line_annotations")

    # Load line registry from DDD repo
    tenant_id = get_tenant_id(request)
    map_id = _require_map_id(tenant_id)

    if test_case_name:
        # Load line test case
        line_test_case = _load_line_test_case_by_name(test_case_name, map_id)
        if line_test_case is None:
            return JsonResponse(
                {"success": False, "error": f"Line test case not found: {test_case_name}"},
                status=404,
            )
        line_annotations = line_test_case.get("line_annotations", [])
    elif not line_annotations:
        return JsonResponse(
            {
                "success": False,
                "error": "Missing required field: test_case_name or line_annotations",
            },
            status=400,
        )

    if len(line_annotations) < 2:
        return JsonResponse(
            {
                "success": False,
                "error": f"Need at least 2 line annotations, got {len(line_annotations)}",
            },
            status=400,
        )

    lines = _load_line_registry(tenant_id)
    if not lines:
        return JsonResponse(
            {"success": False, "error": "No lines found in line registry"},
            status=500,
        )

    line_registry = {line.line_id: line.to_dict() for line in lines}

    # Compute homography from lines
    if map_id is None:
        return _no_map_error()
    try:
        dist_kw = _distortion_kwargs(line_test_case) if test_case_name else {}
        result = compute_line_homography(
            line_annotations=line_annotations,
            line_registry=line_registry,
            distortion=dist_kw,
            map_id=map_id,
            ransac_threshold=50.0,
            min_inlier_ratio=0.3,
        )
    except (ValueError, RuntimeError) as e:
        return JsonResponse(
            {"success": False, "error": f"Line-based homography computation failed: {e}"},
            status=500,
        )

    return JsonResponse(result.to_payload())


@csrf_exempt
@require_http_methods(["POST"])
def api_compute_line_errors(request: HttpRequest) -> JsonResponse:
    """
    Compute line errors from a line test case.

    POST /api/compute-line-errors/

    Request body:
        {"test_case_name": "valte_30.8_13.1_1_20260112_lines"}
        OR
        {"test_case_name": "...", "use_line_homography": true}

    When use_line_homography is true, computes homography from the line annotations
    themselves instead of requiring point_annotations_ref.

    Returns:
        JSON with line error computation results including per-line errors and overlay data.
    """
    # Parse request body
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"success": False, "error": f"Invalid JSON: {e}"},
            status=400,
        )

    test_case_name = body.get("test_case_name")
    use_line_homography = body.get("use_line_homography", False)

    if not test_case_name:
        return JsonResponse(
            {"success": False, "error": "Missing required field: test_case_name"},
            status=400,
        )

    # Load line test case
    tenant_id = get_tenant_id(request)
    map_id = _require_map_id(tenant_id)
    line_test_case = _load_line_test_case_by_name(test_case_name, map_id)
    if line_test_case is None:
        return JsonResponse(
            {"success": False, "error": f"Line test case not found: {test_case_name}"},
            status=404,
        )

    line_annotations = line_test_case.get("line_annotations", [])
    if not line_annotations:
        return JsonResponse(
            {"success": False, "error": "No line annotations found in test case"},
            status=400,
        )

    # Load line registry from DDD repo (needed for both approaches)
    lines = _load_line_registry(tenant_id)
    if not lines:
        return JsonResponse(
            {"success": False, "error": "No lines found in line registry"},
            status=500,
        )

    line_registry = {line.line_id: line.to_dict() for line in lines}

    # Compute homography - either from lines or from referenced point annotations
    homography_source = "lines" if use_line_homography else "points"

    if use_line_homography:
        # Compute homography from line annotations
        if len(line_annotations) < 2:
            return JsonResponse(
                {
                    "success": False,
                    "error": f"Need at least 2 line annotations for line-based homography, got {len(line_annotations)}",
                },
                status=400,
            )

        map_id = _require_map_id(tenant_id)
        if map_id is None:
            return _no_map_error()
        try:
            homography = MapPointHomography(map_id=map_id, **_distortion_kwargs(line_test_case))
            line_result = homography.compute_from_lines(
                line_annotations=line_annotations,
                line_registry=line_registry,
                ransac_threshold=50.0,
                min_inlier_ratio=0.3,
            )
        except (ValueError, RuntimeError) as e:
            return JsonResponse(
                {"success": False, "error": f"Line-based homography computation failed: {e}"},
                status=500,
            )
    else:
        # Use point-based homography (original behavior)
        point_annotations_ref = line_test_case.get("point_annotations_ref")
        if not point_annotations_ref:
            return JsonResponse(
                {
                    "success": False,
                    "error": "No point_annotations_ref found in line test case. Use use_line_homography=true for line-based homography.",
                },
                status=400,
            )

        # Load the referenced point test case
        point_test_case = _load_test_case_by_name(point_annotations_ref, map_id)
        if point_test_case is None:
            return JsonResponse(
                {
                    "success": False,
                    "error": f"Referenced point test case not found: {point_annotations_ref}",
                },
                status=404,
            )

        annotations = point_test_case.get("annotations", [])
        if len(annotations) < 4:
            return JsonResponse(
                {
                    "success": False,
                    "error": f"Need at least 4 point annotations, got {len(annotations)}",
                },
                status=400,
            )

        # Load GCP registry from repository
        map_id_for_gcps = _require_map_id(tenant_id)
        if map_id_for_gcps is None:
            return _no_map_error()
        try:
            gcp_registry = from_gcp_repo(GCPS_DIR, map_id_for_gcps)
        except (KeyError, ValueError, OSError) as e:
            return JsonResponse(
                {"success": False, "error": f"Failed to load GCP registry: {e}"},
                status=500,
            )

        # Compute homography from point annotations
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
            return JsonResponse(
                {"success": False, "error": f"Homography computation failed: {e}"},
                status=500,
            )

    # Compute per-line errors (line_registry already loaded above)
    try:
        line_errors = compute_line_errors(
            homography=homography,
            line_annotations=line_annotations,
            line_registry=line_registry,
        )
    except LineNotInRegistryError as e:
        return JsonResponse(
            {"success": False, "error": f"Line {e.line_id} not found in line registry"},
            status=400,
        )

    return JsonResponse(line_errors.to_payload())


@require_GET
def api_camera_info(request: HttpRequest) -> JsonResponse:
    """
    Get camera image metadata.

    GET /api/camera-info/?case=valte_30.8_13.1_1

    Query parameters:
        case: Test case name

    Returns:
        JSON with image info: {"width": N, "height": N, "filename": "..."}
    """
    test_case_name = request.GET.get("case")
    if not test_case_name:
        return JsonResponse(
            {"error": "Missing required parameter: case"},
            status=400,
        )

    map_id = _require_map_id(get_tenant_id(request))
    info = _get_camera_info(test_case_name, map_id)
    if info is None:
        return JsonResponse(
            {"error": f"Test case not found or image missing: {test_case_name}"},
            status=404,
        )

    return JsonResponse(
        {
            "width": info["width"],
            "height": info["height"],
            "filename": info["filename"],
        }
    )


@require_GET
def api_map_info(request: HttpRequest) -> JsonResponse:
    """
    Get map GeoTIFF metadata.

    GET /api/map-info/

    Returns:
        JSON with image info: {"width": N, "height": N, "filename": "...", "geotransform": [...], "crs": "..."}
    """
    tenant_id = get_tenant_id(request)
    info = _get_map_info(tenant_id)
    if info is None:
        return JsonResponse(
            {"error": f"Map GeoTIFF not found: {_get_map_geotiff_file(tenant_id)}"},
            status=404,
        )

    return JsonResponse(
        {
            "width": info["width"],
            "height": info["height"],
            "filename": info["filename"],
            "geotransform": info["geotransform"],
            "crs": info["crs"],
        }
    )


@require_GET
def api_camera_tile(request: HttpRequest) -> HttpResponse:
    """
    Get a camera image tile at specified coordinates and zoom level.

    GET /api/camera-tile/?case=valte_30.8_13.1_1&x=0&y=0&z=0&size=256

    Query parameters:
        case: Test case name
        x: Tile x coordinate
        y: Tile y coordinate
        z: Zoom level
        size: Tile size (default: 256)

    OpenSeadragon uses a pyramid where:
    - Level 0 = most zoomed out (fewest tiles)
    - Max level = full resolution (most tiles)

    At level z, the image appears at resolution: original / 2^(max_level - z)
    """
    # Parse test case name
    test_case_name = request.GET.get("case")
    if not test_case_name:
        return JsonResponse(
            {"error": "Missing required parameter: case"},
            status=400,
        )

    # Parse tile parameters
    try:
        x = int(request.GET.get("x", 0))
        y = int(request.GET.get("y", 0))
        z = int(request.GET.get("z", 0))
        size = int(request.GET.get("size", 256))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid tile parameters"}, status=400)

    # Get image info and path
    map_id = _require_map_id(get_tenant_id(request))
    info = _get_camera_info(test_case_name, map_id)
    if info is None:
        return JsonResponse(
            {"error": f"Test case not found or image missing: {test_case_name}"},
            status=404,
        )

    image_path = _get_camera_image_path(test_case_name, map_id)
    if image_path is None or not image_path.exists():
        return JsonResponse(
            {"error": f"Camera image not found for test case: {test_case_name}"},
            status=404,
        )

    width = info["width"]
    height = info["height"]

    # Calculate max level for the pyramid
    max_level = math.ceil(math.log2(max(width, height)))

    # At level z, each pixel in the tile grid corresponds to 2^(max_level-z) original pixels
    level_scale = 2 ** (max_level - z)

    # Calculate bounds in original image coordinates
    # Each tile covers (size * level_scale) original pixels
    x0 = x * size * level_scale
    y0 = y * size * level_scale
    x1 = (x + 1) * size * level_scale
    y1 = (y + 1) * size * level_scale

    # Clamp to image bounds
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))

    if x1 <= x0 or y1 <= y0:
        # Return transparent tile
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return HttpResponse(buffer.getvalue(), content_type="image/png")

    # Read from PNG using PIL
    with Image.open(image_path) as full_img:
        # Crop to tile region
        img = full_img.crop((x0, y0, x1, y1))
        if img.mode != "RGB":
            img = img.convert("RGB")

    # Resize to tile size
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return HttpResponse(buffer.getvalue(), content_type="image/png")


@require_GET
def api_map_tile(request: HttpRequest) -> HttpResponse:
    """
    Get a map GeoTIFF tile at specified coordinates and zoom level.

    GET /api/map-tile/?x=0&y=0&z=0&size=256

    Query parameters:
        x: Tile x coordinate
        y: Tile y coordinate
        z: Zoom level
        size: Tile size (default: 256)

    OpenSeadragon uses a pyramid where:
    - Level 0 = most zoomed out (fewest tiles)
    - Max level = full resolution (most tiles)

    At level z, the image appears at resolution: original / 2^(max_level - z)
    """
    # Parse tile parameters
    try:
        x = int(request.GET.get("x", 0))
        y = int(request.GET.get("y", 0))
        z = int(request.GET.get("z", 0))
        size = int(request.GET.get("size", 256))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid tile parameters"}, status=400)

    # Get map info
    tenant_id = get_tenant_id(request)
    info = _get_map_info(tenant_id)
    if info is None:
        return JsonResponse(
            {"error": f"Map GeoTIFF not found: {_get_map_geotiff_file(tenant_id)}"},
            status=404,
        )

    width = info["width"]
    height = info["height"]

    # Calculate max level for the pyramid
    max_level = math.ceil(math.log2(max(width, height)))

    # At level z, each pixel in the tile grid corresponds to 2^(max_level-z) original pixels
    level_scale = 2 ** (max_level - z)

    # Calculate bounds in original image coordinates
    # Each tile covers (size * level_scale) original pixels
    x0 = x * size * level_scale
    y0 = y * size * level_scale
    x1 = (x + 1) * size * level_scale
    y1 = (y + 1) * size * level_scale

    # Clamp to image bounds
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))

    if x1 <= x0 or y1 <= y0:
        # Return transparent tile
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return HttpResponse(buffer.getvalue(), content_type="image/png")

    # Read from TIFF using tifffile
    with tifffile.TiffFile(_get_map_geotiff_file(tenant_id)) as tif:
        page = tif.pages[0]
        data = page.asarray()

        # Extract the tile region
        if data.ndim == 2:
            tile_data = data[y0:y1, x0:x1]
            img = Image.fromarray(_normalize_array(tile_data), mode="L")
            img = img.convert("RGB")
        elif data.ndim == 3:
            if data.shape[2] >= 3:
                tile_data = data[y0:y1, x0:x1, :3]
                img = Image.fromarray(_normalize_array(tile_data), mode="RGB")
            else:
                tile_data = data[y0:y1, x0:x1, 0]
                img = Image.fromarray(_normalize_array(tile_data), mode="L")
                img = img.convert("RGB")
        else:
            if data.shape[0] in (1, 3, 4):
                if data.shape[0] == 1:
                    tile_data = data[0, y0:y1, x0:x1]
                    img = Image.fromarray(_normalize_array(tile_data), mode="L")
                    img = img.convert("RGB")
                else:
                    tile_data = np.transpose(data[:3, y0:y1, x0:x1], (1, 2, 0))
                    img = Image.fromarray(_normalize_array(tile_data), mode="RGB")
            else:
                tile_data = data[y0:y1, x0:x1] if data.ndim == 2 else data[y0:y1, x0:x1, 0]
                img = Image.fromarray(_normalize_array(tile_data), mode="L")
                img = img.convert("RGB")

    # Resize to tile size
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return HttpResponse(buffer.getvalue(), content_type="image/png")
