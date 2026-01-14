"""
Views for homography precision visualization.

Django view functions for rendering precision visualizations and API endpoints.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import tifffile
import yaml
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from PIL import Image

from poc_homography.homography.map_points import MapPointHomography
from poc_homography.map_points import MapPointRegistry
from poc_homography.pixel_point import PixelPoint

# Test data paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"
ANNOTATIONS_FILE = TEST_DATA_DIR / "valte_annotations.yaml"
GCP_REGISTRY_FILE = TEST_DATA_DIR / "Cartografia_valencia_gcps.yaml"
MAP_GEOTIFF_FILE = PROJECT_ROOT / "Cartografia_valencia.tif"


# Cache for image dimensions to avoid repeatedly opening files
class ImageInfoCache(TypedDict):
    """Cache entry for image information."""

    width: int
    height: int
    filename: str
    geotransform: list[float] | None
    crs: str | None


_image_info_cache: dict[str, ImageInfoCache] = {}


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize array values to 0-255 range for display.

    Args:
        arr: Input array.

    Returns:
        Normalized uint8 array.
    """
    if arr.dtype == np.uint8:
        return arr

    # Handle floating point and other types
    arr = arr.astype(np.float64)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    if max_val - min_val > 0:
        arr = (arr - min_val) / (max_val - min_val) * 255
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.uint8)


def _extract_geotransform(tif: tifffile.TiffFile) -> tuple[list[float] | None, str | None]:
    """Extract GeoTIFF geotransform and CRS info from TIFF tags.

    Args:
        tif: Open tifffile TiffFile object.

    Returns:
        Tuple of (geotransform, crs_string).
        geotransform is 6-element list [origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height]
        or None if not available.
    """
    page = tif.pages[0]
    # type: ignore needed because tifffile types are incomplete
    tags = {tag.name: tag for tag in page.tags.values()}  # type: ignore[union-attr]

    geotransform = None
    crs = None

    # Try to extract GeoTIFF metadata
    if "ModelPixelScaleTag" in tags and "ModelTiepointTag" in tags:
        # Common GeoTIFF format: pixel scale + tiepoint
        try:
            scale = tags["ModelPixelScaleTag"].value
            tiepoint = tags["ModelTiepointTag"].value

            # GDAL-style geotransform: [originX, pixelWidth, rotationX, originY, rotationY, pixelHeight]
            # tiepoint = [I, J, K, X, Y, Z] where (I,J,K) is pixel coord and (X,Y,Z) is map coord
            origin_x = tiepoint[3] - tiepoint[0] * scale[0]
            origin_y = tiepoint[4] + tiepoint[1] * scale[1]

            geotransform = [
                float(origin_x),  # GT[0]: origin X
                float(scale[0]),  # GT[1]: pixel width
                0.0,  # GT[2]: rotation (typically 0)
                float(origin_y),  # GT[3]: origin Y
                0.0,  # GT[4]: rotation (typically 0)
                -float(scale[1]),  # GT[5]: pixel height (negative for north-up)
            ]
        except (IndexError, TypeError, ValueError):
            pass

    elif "ModelTransformationTag" in tags:
        # Alternative: 4x4 transformation matrix
        try:
            matrix = tags["ModelTransformationTag"].value
            geotransform = [
                float(matrix[3]),  # origin X
                float(matrix[0]),  # pixel width
                float(matrix[1]),  # rotation
                float(matrix[7]),  # origin Y
                float(matrix[4]),  # rotation
                float(matrix[5]),  # pixel height
            ]
        except (IndexError, TypeError, ValueError):
            pass

    # Try to get CRS info from GeoKeyDirectoryTag
    if "GeoKeyDirectoryTag" in tags:
        try:
            geo_keys = tags["GeoKeyDirectoryTag"].value
            # Look for ProjectedCSTypeGeoKey (3072) or GeographicTypeGeoKey (2048)
            for i in range(4, len(geo_keys), 4):
                key_id = geo_keys[i]
                if key_id == 3072:  # ProjectedCSTypeGeoKey
                    epsg = geo_keys[i + 3]
                    crs = f"EPSG:{epsg}"
                    break
                elif key_id == 2048:  # GeographicTypeGeoKey
                    epsg = geo_keys[i + 3]
                    crs = f"EPSG:{epsg}"
        except (IndexError, TypeError, ValueError):
            pass

    return geotransform, crs


def _get_camera_image_path(test_case_name: str) -> Path | None:
    """Get the path to the camera image for a test case.

    Args:
        test_case_name: Name of the test case.

    Returns:
        Path to the camera image, or None if not found.
    """
    test_case = _load_test_case_by_name(test_case_name)
    if test_case is None:
        return None

    image_filename = test_case.get("image")
    if not image_filename:
        return None

    return TEST_DATA_DIR / image_filename


def _get_camera_info(test_case_name: str) -> ImageInfoCache | None:
    """Get cached camera image info, loading if necessary.

    Args:
        test_case_name: Name of the test case.

    Returns:
        Image info dict or None if not found.
    """
    cache_key = f"camera:{test_case_name}"
    if cache_key in _image_info_cache:
        return _image_info_cache[cache_key]

    image_path = _get_camera_image_path(test_case_name)
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


def _get_map_info() -> ImageInfoCache | None:
    """Get cached map GeoTIFF info, loading if necessary.

    Returns:
        Image info dict or None if not found.
    """
    cache_key = "map:geotiff"
    if cache_key in _image_info_cache:
        return _image_info_cache[cache_key]

    if not MAP_GEOTIFF_FILE.exists():
        return None

    with tifffile.TiffFile(MAP_GEOTIFF_FILE) as tif:
        page = tif.pages[0]
        width: int = page.imagewidth  # type: ignore[union-attr]
        height: int = page.imagelength  # type: ignore[union-attr]
        geotransform, crs = _extract_geotransform(tif)

        info: ImageInfoCache = {
            "width": width,
            "height": height,
            "filename": MAP_GEOTIFF_FILE.name,
            "geotransform": geotransform,
            "crs": crs,
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
    if not ANNOTATIONS_FILE.exists():
        return JsonResponse(
            {"error": f"Annotations file not found: {ANNOTATIONS_FILE}"},
            status=404,
        )

    try:
        with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return JsonResponse(
            {"error": f"Failed to parse annotations YAML: {e}"},
            status=500,
        )

    if not data or "test_cases" not in data:
        return JsonResponse(
            {"error": "No test_cases found in annotations file"},
            status=404,
        )

    test_cases = [
        {
            "name": tc.get("name", ""),
            "image": tc.get("image", ""),
            "annotation_count": len(tc.get("annotations", [])),
        }
        for tc in data["test_cases"]
    ]

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
    if not ANNOTATIONS_FILE.exists():
        return JsonResponse(
            {"error": f"Annotations file not found: {ANNOTATIONS_FILE}"},
            status=404,
        )

    try:
        with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return JsonResponse(
            {"error": f"Failed to parse annotations YAML: {e}"},
            status=500,
        )

    if not data or "test_cases" not in data:
        return JsonResponse(
            {"error": "No test_cases found in annotations file"},
            status=404,
        )

    for tc in data["test_cases"]:
        if tc.get("name") == name:
            return JsonResponse(
                {
                    "name": tc.get("name", ""),
                    "image": tc.get("image", ""),
                    "annotations": tc.get("annotations", []),
                }
            )

    return JsonResponse(
        {"error": f"Test case not found: {name}"},
        status=404,
    )


def _load_test_case_by_name(name: str) -> dict | None:
    """Load a test case by name from the annotations file.

    Args:
        name: Name of the test case to load.

    Returns:
        The test case dictionary if found, None otherwise.
    """
    if not ANNOTATIONS_FILE.exists():
        return None

    try:
        with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        return None

    if not data or "test_cases" not in data:
        return None

    for tc in data["test_cases"]:
        if tc.get("name") == name:
            return tc

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

    # Load test case
    test_case = _load_test_case_by_name(test_case_name)
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

    # Load GCP registry
    if not GCP_REGISTRY_FILE.exists():
        return JsonResponse(
            {"success": False, "error": f"GCP registry file not found: {GCP_REGISTRY_FILE}"},
            status=500,
        )

    try:
        registry = MapPointRegistry.load(GCP_REGISTRY_FILE)
    except (yaml.YAMLError, KeyError, ValueError) as e:
        return JsonResponse(
            {"success": False, "error": f"Failed to load GCP registry: {e}"},
            status=500,
        )

    # Compute homography
    try:
        homography = MapPointHomography(map_id=registry.map_id)
        result = homography.compute_from_gcps(
            gcps=annotations,
            map_registry=registry,
            ransac_threshold=50.0,
            min_inlier_ratio=0.5,
        )
    except (ValueError, RuntimeError) as e:
        return JsonResponse(
            {"success": False, "error": f"Homography computation failed: {e}"},
            status=500,
        )

    # Compute per-point errors and overlay data
    per_point_errors = []
    camera_annotations = []
    camera_reprojected = []
    map_gcps = []
    map_projected = []

    for annotation in annotations:
        gcp_id = annotation["gcp_id"]

        # Get original camera pixel (annotation position)
        camera_x = annotation["pixel_x"]
        camera_y = annotation["pixel_y"]
        camera_pixel = PixelPoint(camera_x, camera_y)

        # Get GCP map coordinate from registry
        gcp = registry.points[gcp_id]
        map_x = gcp.pixel_x
        map_y = gcp.pixel_y

        # Project annotation to map using camera_to_map()
        projected_map = homography.camera_to_map(camera_pixel)

        # Reproject GCP back to camera using map_to_camera()
        gcp_coord = PixelPoint(map_x, map_y)
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

        per_point_errors.append({
            "gcp_id": gcp_id,
            "error_px": round(error_px, 2),
            "camera_dx": round(camera_dx, 2),
            "camera_dy": round(camera_dy, 2),
            "map_dx": round(map_dx, 2),
            "map_dy": round(map_dy, 2),
            "camera_original": [round(camera_x, 2), round(camera_y, 2)],
            "camera_reprojected": [round(reprojected_camera.x, 2), round(reprojected_camera.y, 2)],
            "map_original": [round(map_x, 2), round(map_y, 2)],
            "map_projected": [round(projected_map.pixel_x, 2), round(projected_map.pixel_y, 2)],
        })

        # Collect overlay data
        camera_annotations.append({
            "gcp_id": gcp_id,
            "x": round(camera_x, 2),
            "y": round(camera_y, 2),
        })
        camera_reprojected.append({
            "gcp_id": gcp_id,
            "x": round(reprojected_camera.x, 2),
            "y": round(reprojected_camera.y, 2),
        })
        map_gcps.append({
            "gcp_id": gcp_id,
            "x": round(map_x, 2),
            "y": round(map_y, 2),
        })
        map_projected.append({
            "gcp_id": gcp_id,
            "x": round(projected_map.pixel_x, 2),
            "y": round(projected_map.pixel_y, 2),
        })

    return JsonResponse({
        "success": True,
        "metrics": {
            "num_gcps": result.num_gcps,
            "num_inliers": result.num_inliers,
            "inlier_ratio": round(result.inlier_ratio, 2),
            "mean_reproj_error": round(result.mean_reproj_error, 2),
            "max_reproj_error": round(result.max_reproj_error, 2),
            "rmse": round(result.rmse, 2),
        },
        "per_point_errors": per_point_errors,
        "overlays": {
            "camera": {
                "annotations": camera_annotations,
                "reprojected_gcps": camera_reprojected,
            },
            "map": {
                "gcps": map_gcps,
                "projected_annotations": map_projected,
            },
        },
    })


@require_GET
def api_gcp_registry(request: HttpRequest) -> JsonResponse:
    """
    Get the GCP registry.

    GET /api/gcp-registry/

    Returns:
        JSON with registry data:
        {"map_id": "...", "points": {"PS1": {"pixel_x": ..., "pixel_y": ...}, ...}}
    """
    if not GCP_REGISTRY_FILE.exists():
        return JsonResponse(
            {"error": f"GCP registry file not found: {GCP_REGISTRY_FILE}"},
            status=404,
        )

    try:
        registry = MapPointRegistry.load(GCP_REGISTRY_FILE)
    except (yaml.YAMLError, KeyError, ValueError) as e:
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

    info = _get_camera_info(test_case_name)
    if info is None:
        return JsonResponse(
            {"error": f"Test case not found or image missing: {test_case_name}"},
            status=404,
        )

    return JsonResponse({
        "width": info["width"],
        "height": info["height"],
        "filename": info["filename"],
    })


@require_GET
def api_map_info(request: HttpRequest) -> JsonResponse:
    """
    Get map GeoTIFF metadata.

    GET /api/map-info/

    Returns:
        JSON with image info: {"width": N, "height": N, "filename": "...", "geotransform": [...], "crs": "..."}
    """
    info = _get_map_info()
    if info is None:
        return JsonResponse(
            {"error": f"Map GeoTIFF not found: {MAP_GEOTIFF_FILE}"},
            status=404,
        )

    return JsonResponse({
        "width": info["width"],
        "height": info["height"],
        "filename": info["filename"],
        "geotransform": info["geotransform"],
        "crs": info["crs"],
    })


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
    info = _get_camera_info(test_case_name)
    if info is None:
        return JsonResponse(
            {"error": f"Test case not found or image missing: {test_case_name}"},
            status=404,
        )

    image_path = _get_camera_image_path(test_case_name)
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
    info = _get_map_info()
    if info is None:
        return JsonResponse(
            {"error": f"Map GeoTIFF not found: {MAP_GEOTIFF_FILE}"},
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
    with tifffile.TiffFile(MAP_GEOTIFF_FILE) as tif:
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
