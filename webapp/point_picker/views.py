"""Views for the point picker Django app."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import tifffile
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from PIL import Image

from .state import get_state, get_tag_from_id, normalize_array
from .validation import (
    validate_add_point_request,
    validate_export_request,
    validate_import_request,
    validate_update_point_request,
)


def index(request: HttpRequest) -> HttpResponse:
    """Serve the main HTML page."""
    return render(request, "point_picker/index.html")


@require_GET
def api_image_info(request: HttpRequest) -> JsonResponse:
    """Get image metadata."""
    state = get_state()
    return JsonResponse(
        {
            "width": state.width,
            "height": state.height,
            "geotransform": state.geotiff.geotransform.to_dict() if state.geotiff else None,
            "crs": state.geotiff.crs if state.geotiff else None,
            "filename": state.geotiff_path.name,
        }
    )


@require_GET
def api_image_tile(request: HttpRequest) -> HttpResponse:
    """Get an image tile at specified coordinates and zoom level.

    OpenSeadragon uses a pyramid where:
    - Level 0 = most zoomed out (fewest tiles)
    - Max level = full resolution (most tiles)

    At level z, the image appears at resolution: original / 2^(max_level - z)
    """
    # Parse query parameters
    try:
        x = int(request.GET.get("x", 0))
        y = int(request.GET.get("y", 0))
        z = int(request.GET.get("z", 0))
        size = int(request.GET.get("size", 256))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid tile parameters"}, status=400)

    state = get_state()

    # Calculate max level for the pyramid
    max_level = math.ceil(math.log2(max(state.width, state.height)))

    # At level z, each pixel in the tile grid corresponds to 2^(max_level-z) original pixels
    level_scale = 2 ** (max_level - z)

    # Calculate bounds in original image coordinates
    # Each tile covers (size * level_scale) original pixels
    x0 = x * size * level_scale
    y0 = y * size * level_scale
    x1 = (x + 1) * size * level_scale
    y1 = (y + 1) * size * level_scale

    # Clamp to image bounds
    x0 = max(0, min(x0, state.width))
    y0 = max(0, min(y0, state.height))
    x1 = max(0, min(x1, state.width))
    y1 = max(0, min(y1, state.height))

    if x1 <= x0 or y1 <= y0:
        # Return transparent tile
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return HttpResponse(buffer.getvalue(), content_type="image/png")

    # Read the tile based on file type
    suffix = state.geotiff_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        # Read from TIFF using tifffile
        with tifffile.TiffFile(state.geotiff_path) as tif:
            page = tif.pages[0]
            data = page.asarray()

            # Extract the tile region
            if data.ndim == 2:
                tile_data = data[y0:y1, x0:x1]
                img = Image.fromarray(normalize_array(tile_data), mode="L")
                img = img.convert("RGB")
            elif data.ndim == 3:
                if data.shape[2] >= 3:
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
                    tile_data = data[y0:y1, x0:x1] if data.ndim == 2 else data[y0:y1, x0:x1, 0]
                    img = Image.fromarray(normalize_array(tile_data), mode="L")
                    img = img.convert("RGB")
    else:
        # Read from other formats (PNG, JPG, etc.) using PIL
        with Image.open(state.geotiff_path) as full_img:
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
def api_image_full(request: HttpRequest) -> HttpResponse:
    """Get the full image scaled to max_size."""
    try:
        max_size = int(request.GET.get("max_size", 2048))
    except (TypeError, ValueError):
        max_size = 2048

    state = get_state()

    suffix = state.geotiff_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(state.geotiff_path) as tif:
            page = tif.pages[0]
            data = page.asarray()

            # Convert to image
            if data.ndim == 2:
                img = Image.fromarray(normalize_array(data), mode="L")
                img = img.convert("RGB")
            elif data.ndim == 3:
                if data.shape[2] >= 3:
                    img = Image.fromarray(normalize_array(data[:, :, :3]), mode="RGB")
                else:
                    img = Image.fromarray(normalize_array(data[:, :, 0]), mode="L")
                    img = img.convert("RGB")
            else:
                if data.shape[0] in (1, 3, 4):
                    if data.shape[0] == 1:
                        img = Image.fromarray(normalize_array(data[0]), mode="L")
                        img = img.convert("RGB")
                    else:
                        img_array = np.transpose(data[:3], (1, 2, 0))
                        img = Image.fromarray(normalize_array(img_array), mode="RGB")
                else:
                    img = Image.fromarray(normalize_array(data), mode="L")
                    img = img.convert("RGB")
    else:
        # Load other formats with PIL using context manager to prevent resource leak
        with Image.open(state.geotiff_path) as pil_img:
            img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img.copy()

    # Scale to max_size while preserving aspect ratio
    ratio = min(max_size / img.width, max_size / img.height)
    if ratio < 1:
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return HttpResponse(buffer.getvalue(), content_type="image/png")


@csrf_exempt
@require_http_methods(["GET", "POST"])
def api_points(request: HttpRequest) -> JsonResponse:
    """Get all points (GET) or add a new point (POST)."""
    state = get_state()

    if request.method == "GET":
        return JsonResponse(
            {
                "map_id": state.registry.map_id,
                "points": [
                    {
                        "id": pid,
                        "pixel_x": p.pixel_x,
                        "pixel_y": p.pixel_y,
                        "tag": get_tag_from_id(pid),
                    }
                    for pid, p in state.registry.points.items()
                ],
            }
        )

    # POST - add a new point
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    error = validate_add_point_request(data)
    if error:
        return JsonResponse({"error": error}, status=422)

    tag = data.get("tag", "extra")
    pixel_x = float(data["pixel_x"])
    pixel_y = float(data["pixel_y"])
    point_id = data.get("id")

    point_id = state.add_point(tag, pixel_x, pixel_y, point_id=point_id)
    point = state.registry.points[point_id]

    return JsonResponse(
        {
            "id": point_id,
            "pixel_x": point.pixel_x,
            "pixel_y": point.pixel_y,
            "tag": get_tag_from_id(point_id),
        }
    )


@csrf_exempt
@require_http_methods(["PUT", "DELETE"])
def api_point_detail(request: HttpRequest, point_id: str) -> JsonResponse:
    """Update (PUT) or delete (DELETE) a specific point."""
    state = get_state()

    if request.method == "DELETE":
        try:
            state.delete_point(point_id)
            return JsonResponse({"deleted": point_id})
        except KeyError:
            return JsonResponse({"error": f"Point not found: {point_id}"}, status=404)

    # PUT - update point coordinates
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    error = validate_update_point_request(data)
    if error:
        return JsonResponse({"error": error}, status=422)

    try:
        state.update_point(point_id, float(data["pixel_x"]), float(data["pixel_y"]))
        point = state.registry.points[point_id]
        return JsonResponse(
            {
                "id": point_id,
                "pixel_x": point.pixel_x,
                "pixel_y": point.pixel_y,
                "tag": get_tag_from_id(point_id),
            }
        )
    except KeyError:
        return JsonResponse({"error": f"Point not found: {point_id}"}, status=404)


@require_GET
def api_next_id(request: HttpRequest) -> JsonResponse:
    """Get the next ID for a tag category."""
    tag = request.GET.get("tag")
    if not tag:
        return JsonResponse({"error": "Missing required parameter: tag"}, status=400)

    state = get_state()
    try:
        next_id = state.get_next_id(tag)
        return JsonResponse({"tag": tag, "next_id": next_id})
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)


@require_GET
def api_geo_coords(request: HttpRequest) -> JsonResponse:
    """Convert pixel coordinates to geographic coordinates."""
    try:
        pixel_x = float(request.GET.get("pixel_x", 0))
        pixel_y = float(request.GET.get("pixel_y", 0))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid coordinate parameters"}, status=400)

    state = get_state()
    coords = state.get_geo_coords(pixel_x, pixel_y)
    if coords:
        return JsonResponse(
            {
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "easting": coords[0],
                "northing": coords[1],
                "crs": state.geotiff.crs if state.geotiff else None,
            }
        )
    return JsonResponse(
        {
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "easting": None,
            "northing": None,
            "crs": None,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def api_export(request: HttpRequest) -> JsonResponse:
    """Export points to YAML file."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    error = validate_export_request(data)
    if error:
        return JsonResponse({"error": error}, status=422)

    state = get_state()
    path = Path(data.get("path", "")) if data.get("path") else Path(f"{state.map_id}_points.yaml")
    state.save_registry(path)
    return JsonResponse({"exported": str(path), "count": len(state.registry.points)})


@csrf_exempt
@require_http_methods(["POST"])
def api_import(request: HttpRequest) -> JsonResponse:
    """Import points from YAML file."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    error = validate_import_request(data)
    if error:
        return JsonResponse({"error": error}, status=422)

    state = get_state()
    path = Path(data["path"])
    if not path.exists():
        return JsonResponse({"error": f"File not found: {path}"}, status=404)

    state.load_registry(path)
    return JsonResponse(
        {
            "imported": str(path),
            "count": len(state.registry.points),
            "map_id": state.registry.map_id,
        }
    )
