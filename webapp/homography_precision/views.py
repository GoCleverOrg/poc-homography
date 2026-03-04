"""
Views for homography precision visualization.

Django view functions for rendering precision visualizations and API endpoints.
All business logic is delegated to the services module; views handle only HTTP
concerns (request parsing, JSON serialization, error responses).
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from homography_web.frame_utils import (
    get_map_from_tenant_id,
    get_tenant_id,
)
from homography_web.frame_utils import (
    normalize_array as _normalize_array,
)
from PIL import Image

from . import services as svc

if TYPE_CHECKING:
    from homography_web.dtos import LineAnnotationDTO, PointAnnotationDTO

# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------


def _no_map_error() -> JsonResponse:
    """Create a fresh 'no map configured' error response."""
    return JsonResponse(
        {
            "success": False,
            "error": "No map configured for the current tenant. Upload a GeoTIFF map first.",
        },
        status=422,
    )


# ---------------------------------------------------------------------------
# Page views
# ---------------------------------------------------------------------------


def index(request: HttpRequest) -> HttpResponse:
    """Landing page for homography precision visualization."""
    return render(
        request,
        "homography_precision/index.html",
        {"title": "Homography Precision Visualization"},
    )


# ---------------------------------------------------------------------------
# Test case listing / detail APIs
# ---------------------------------------------------------------------------


@require_GET
def api_test_cases(request: HttpRequest) -> JsonResponse:
    """
    List all available test cases.

    GET /api/test-cases/

    Returns:
        JSON with list of test cases:
        {"test_cases": [{"name": "...", "image": "...", "annotation_count": N}, ...]}
    """
    map_entity = get_map_from_tenant_id(get_tenant_id(request))
    map_id = map_entity.id if map_entity else None
    cases = svc.list_test_cases(map_id)
    if not cases:
        return JsonResponse({"error": "No captured frames found"}, status=404)

    return JsonResponse(
        {
            "test_cases": [
                {
                    "name": tc.name,
                    "image": tc.image,
                    "annotation_count": tc.annotation_count,
                }
                for tc in cases
            ]
        }
    )


@require_GET
def api_test_case_detail(request: HttpRequest, name: str) -> JsonResponse:
    """
    Get a specific test case by name.

    GET /api/test-cases/{name}/

    Returns:
        JSON with full test case data:
        {"name": "...", "image": "...", "annotations": [{...}, ...]}
    """
    map_id = svc.require_map_id(get_tenant_id(request))
    tc = svc.load_test_case_by_name(name, map_id)
    if tc is None:
        return JsonResponse(
            {"error": f"Test case not found: {name}"},
            status=404,
        )

    annotations = tc.get("annotations", [])
    return JsonResponse(
        {
            "name": tc.get("name", ""),
            "image": tc.get("image", ""),
            "annotations": [a.to_dict() for a in annotations],
        }
    )


# ---------------------------------------------------------------------------
# GCP / Line registry APIs
# ---------------------------------------------------------------------------


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
    map_id = svc.require_map_id(tenant_id)
    if map_id is None:
        return _no_map_error()
    try:
        points_dict = svc.load_gcp_registry(map_id)
    except (KeyError, ValueError, OSError) as e:
        return JsonResponse(
            {"error": f"Failed to load GCP registry: {e}"},
            status=500,
        )

    return JsonResponse(
        {
            "map_id": map_id,
            "points": points_dict,
        }
    )


@require_GET
def api_line_registry(request: HttpRequest) -> JsonResponse:
    """
    Get the line registry.

    GET /api/line-registry/

    Returns:
        JSON with line registry data:
        {"map_id": "...", "lines": [{"line_id": "L1", ...}, ...]}
    """
    tenant_id = get_tenant_id(request)
    lines = svc.load_line_registry(tenant_id)
    if not lines:
        return JsonResponse(
            {"error": "No lines found in line registry"},
            status=404,
        )

    return JsonResponse(
        {
            "map_id": svc.require_map_id(tenant_id),
            "lines": [line.to_dict() for line in lines],
        }
    )


# ---------------------------------------------------------------------------
# Line test case listing / detail APIs
# ---------------------------------------------------------------------------


@require_GET
def api_line_test_cases(request: HttpRequest) -> JsonResponse:
    """
    List all available line test cases.

    GET /api/line-test-cases/

    Returns:
        JSON with list of line test cases:
        {"test_cases": [{"name": "...", "image": "...", "line_annotation_count": N}, ...]}
    """
    map_id = svc.require_map_id(get_tenant_id(request))
    cases = svc.list_line_test_cases(map_id)
    if not cases:
        return JsonResponse(
            {"error": "No frames with line annotations found"},
            status=404,
        )

    return JsonResponse(
        {
            "test_cases": [
                {
                    "name": tc.name,
                    "image": tc.image,
                    "line_annotation_count": tc.line_annotation_count,
                }
                for tc in cases
            ]
        }
    )


@require_GET
def api_line_test_case_detail(request: HttpRequest, name: str) -> JsonResponse:
    """
    Get a specific line test case by name.

    GET /api/line-test-cases/{name}/

    Returns:
        JSON with full line test case data including point_annotations_ref.
    """
    map_id = svc.require_map_id(get_tenant_id(request))
    tc = svc.load_line_test_case_by_name(name, map_id)
    if tc is None:
        return JsonResponse(
            {"error": f"Line test case not found: {name}"},
            status=404,
        )

    line_anns = tc.get("line_annotations", [])
    return JsonResponse(
        {
            "name": tc.get("name", ""),
            "image": tc.get("image", ""),
            "point_annotations_ref": tc.get("point_annotations_ref", ""),
            "line_annotations": [a.to_dict() for a in line_anns],
        }
    )


# ---------------------------------------------------------------------------
# Homography computation APIs
# ---------------------------------------------------------------------------


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

    tenant_id = get_tenant_id(request)
    map_id = svc.require_map_id(tenant_id)
    if map_id is None:
        return _no_map_error()

    test_case = svc.load_test_case_by_name(test_case_name, map_id)
    if test_case is None:
        return JsonResponse(
            {"success": False, "error": f"Test case not found: {test_case_name}"},
            status=404,
        )

    annotations: list[PointAnnotationDTO] = test_case.get("annotations", [])
    if len(annotations) < 4:
        return JsonResponse(
            {"success": False, "error": f"Need at least 4 annotations, got {len(annotations)}"},
            status=400,
        )

    try:
        result = svc.compute_point_homography(annotations, map_id)
    except (KeyError, ValueError, OSError) as e:
        return JsonResponse(
            {"success": False, "error": f"Failed to load GCP registry: {e}"},
            status=500,
        )
    except RuntimeError as e:
        return JsonResponse(
            {"success": False, "error": f"Homography computation failed: {e}"},
            status=500,
        )

    return JsonResponse(
        {
            "success": True,
            "metrics": result.metrics,
            "per_point_errors": result.per_point_errors,
            "overlays": result.overlays,
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
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"success": False, "error": f"Invalid JSON: {e}"},
            status=400,
        )

    test_case_name = body.get("test_case_name")
    line_annotations = body.get("line_annotations")

    tenant_id = get_tenant_id(request)
    map_id = svc.require_map_id(tenant_id)
    if map_id is None:
        return _no_map_error()

    if test_case_name:
        line_test_case = svc.load_line_test_case_by_name(test_case_name, map_id)
        if line_test_case is None:
            return JsonResponse(
                {"success": False, "error": f"Line test case not found: {test_case_name}"},
                status=404,
            )
        line_annotations = [
            a.to_dict() if hasattr(a, "to_dict") else a
            for a in line_test_case.get("line_annotations", [])
        ]
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

    lines = svc.load_line_registry(tenant_id)
    if not lines:
        return JsonResponse(
            {"success": False, "error": "No lines found in line registry"},
            status=500,
        )

    line_registry = {line.line_id: line.to_dict() for line in lines}

    try:
        result = svc.compute_line_homography(line_annotations, line_registry, map_id)
    except (ValueError, RuntimeError) as e:
        return JsonResponse(
            {"success": False, "error": f"Line-based homography computation failed: {e}"},
            status=500,
        )

    return JsonResponse(
        {
            "success": True,
            "homography_source": "lines",
            "metrics": result.metrics,
            "homography_matrix": result.homography_matrix,
        }
    )


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

    tenant_id = get_tenant_id(request)
    map_id = svc.require_map_id(tenant_id)
    if map_id is None:
        return _no_map_error()
    line_test_case = svc.load_line_test_case_by_name(test_case_name, map_id)
    if line_test_case is None:
        return JsonResponse(
            {"success": False, "error": f"Line test case not found: {test_case_name}"},
            status=404,
        )

    line_annotations: list[LineAnnotationDTO] = line_test_case.get("line_annotations", [])
    if not line_annotations:
        return JsonResponse(
            {"success": False, "error": "No line annotations found in test case"},
            status=400,
        )

    # Load line registry (needed for both approaches)
    lines = svc.load_line_registry(tenant_id)
    if not lines:
        return JsonResponse(
            {"success": False, "error": "No lines found in line registry"},
            status=500,
        )

    line_registry = {line.line_id: line.to_dict() for line in lines}

    # Build homography - either from lines or from referenced point annotations
    if use_line_homography:
        if len(line_annotations) < 2:
            return JsonResponse(
                {
                    "success": False,
                    "error": (
                        "Need at least 2 line annotations for line-based homography, "
                        f"got {len(line_annotations)}"
                    ),
                },
                status=400,
            )

        try:
            homography = svc.build_homography_from_lines(
                [a.to_dict() for a in line_annotations], line_registry, map_id
            )
        except (ValueError, RuntimeError) as e:
            return JsonResponse(
                {"success": False, "error": f"Line-based homography computation failed: {e}"},
                status=500,
            )
    else:
        point_annotations_ref = line_test_case.get("point_annotations_ref")
        if not point_annotations_ref:
            return JsonResponse(
                {
                    "success": False,
                    "error": (
                        "No point_annotations_ref found in line test case. "
                        "Use use_line_homography=true for line-based homography."
                    ),
                },
                status=400,
            )

        point_test_case = svc.load_test_case_by_name(point_annotations_ref, map_id)
        if point_test_case is None:
            return JsonResponse(
                {
                    "success": False,
                    "error": f"Referenced point test case not found: {point_annotations_ref}",
                },
                status=404,
            )

        pt_annotations: list[PointAnnotationDTO] = point_test_case.get("annotations", [])
        if len(pt_annotations) < 4:
            return JsonResponse(
                {
                    "success": False,
                    "error": f"Need at least 4 point annotations, got {len(pt_annotations)}",
                },
                status=400,
            )

        try:
            homography = svc.build_homography_from_points(pt_annotations, map_id)
        except (KeyError, ValueError, OSError) as e:
            return JsonResponse(
                {"success": False, "error": f"Failed to load GCP registry: {e}"},
                status=500,
            )
        except RuntimeError as e:
            return JsonResponse(
                {"success": False, "error": f"Homography computation failed: {e}"},
                status=500,
            )

    # Compute per-line errors using the built homography
    try:
        result = svc.compute_line_errors(line_annotations, line_registry, homography)
    except KeyError as e:
        return JsonResponse(
            {"success": False, "error": str(e)},
            status=400,
        )

    return JsonResponse(
        {
            "success": True,
            "metrics": result.metrics,
            "per_line_errors": result.per_line_errors,
            "line_overlays": result.line_overlays,
        }
    )


# ---------------------------------------------------------------------------
# Image info APIs
# ---------------------------------------------------------------------------


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

    map_id = svc.require_map_id(get_tenant_id(request))
    info = svc.get_camera_info(test_case_name, map_id)
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
    info = svc.get_map_info(tenant_id)
    if info is None:
        return JsonResponse(
            {"error": f"No map image found for tenant: {tenant_id}"},
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


# ---------------------------------------------------------------------------
# Tile serving APIs
# ---------------------------------------------------------------------------


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
    test_case_name = request.GET.get("case")
    if not test_case_name:
        return JsonResponse(
            {"error": "Missing required parameter: case"},
            status=400,
        )

    try:
        x = int(request.GET.get("x", 0))
        y = int(request.GET.get("y", 0))
        z = int(request.GET.get("z", 0))
        size = int(request.GET.get("size", 256))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid tile parameters"}, status=400)

    map_id = svc.require_map_id(get_tenant_id(request))
    info = svc.get_camera_info(test_case_name, map_id)
    if info is None:
        return JsonResponse(
            {"error": f"Test case not found or image missing: {test_case_name}"},
            status=404,
        )

    image_path = svc.get_camera_image_path(test_case_name, map_id)
    if image_path is None or not image_path.exists():
        return JsonResponse(
            {"error": f"Camera image not found for test case: {test_case_name}"},
            status=404,
        )

    bounds = svc.compute_tile_bounds(info["width"], info["height"], x, y, z, size)

    if bounds.is_empty:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return HttpResponse(buffer.getvalue(), content_type="image/png")

    with Image.open(image_path) as full_img:
        img = full_img.crop((bounds.x0, bounds.y0, bounds.x1, bounds.y1))
        if img.mode != "RGB":
            img = img.convert("RGB")

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
    try:
        x = int(request.GET.get("x", 0))
        y = int(request.GET.get("y", 0))
        z = int(request.GET.get("z", 0))
        size = int(request.GET.get("size", 256))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid tile parameters"}, status=400)

    tenant_id = get_tenant_id(request)
    map_path = svc.resolve_map_file(tenant_id)
    if map_path is None:
        return JsonResponse(
            {"error": f"No map image found for tenant: {tenant_id}"},
            status=404,
        )

    info = svc.get_map_info(tenant_id)
    if info is None:
        return JsonResponse(
            {"error": f"No map image found for tenant: {tenant_id}"},
            status=404,
        )

    width = info["width"]
    height = info["height"]

    bounds = svc.compute_tile_bounds(width, height, x, y, z, size)

    if bounds.is_empty:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return HttpResponse(buffer.getvalue(), content_type="image/png")

    with tifffile.TiffFile(map_path) as tif:
        page = tif.pages[0]
        data = page.asarray()

        if data.ndim == 2:
            tile_data = data[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
            img = Image.fromarray(_normalize_array(tile_data), mode="L")
            img = img.convert("RGB")
        elif data.ndim == 3:
            if data.shape[2] >= 3:
                tile_data = data[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1, :3]
                img = Image.fromarray(_normalize_array(tile_data), mode="RGB")
            else:
                tile_data = data[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1, 0]
                img = Image.fromarray(_normalize_array(tile_data), mode="L")
                img = img.convert("RGB")
        else:
            if data.shape[0] in (1, 3, 4):
                if data.shape[0] == 1:
                    tile_data = data[0, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
                    img = Image.fromarray(_normalize_array(tile_data), mode="L")
                    img = img.convert("RGB")
                else:
                    tile_data = np.transpose(
                        data[:3, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1], (1, 2, 0)
                    )
                    img = Image.fromarray(_normalize_array(tile_data), mode="RGB")
            else:
                tile_data = data[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1, 0]
                img = Image.fromarray(_normalize_array(tile_data), mode="L")
                img = img.convert("RGB")

    img = img.resize((size, size), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return HttpResponse(buffer.getvalue(), content_type="image/png")
