"""Views for the line picker Django app."""

from __future__ import annotations

import io
import json
import math
import numpy as np
import tifffile
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from homography_web.frame_utils import LINES_DIR, normalize_array
from PIL import Image

from .state import from_line_repo, get_state, list_line_map_ids, save_to_line_repo


def index(request: HttpRequest) -> HttpResponse:
    """Serve the main HTML page."""
    return render(request, "line_picker/index.html")


@require_GET
def api_image_info(request: HttpRequest) -> JsonResponse:
    """Get image metadata."""
    state = get_state()
    return JsonResponse(
        {
            "width": state.width,
            "height": state.height,
            "geotransform": state.geotiff.geotransform.to_list() if state.geotiff else None,
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
def api_lines(request: HttpRequest) -> JsonResponse:
    """Get all lines (GET) or add a new line (POST).

    Lines are defined by their own pixel coordinate endpoints, independent of GCPs.
    """
    state = get_state()

    if request.method == "GET":
        # Build line data with pixel coordinates
        lines_data = []
        for line in state.lines:
            lines_data.append(
                {
                    "line_id": line.line_id,
                    "start_x": line.start_x,
                    "start_y": line.start_y,
                    "end_x": line.end_x,
                    "end_y": line.end_y,
                }
            )

        return JsonResponse(
            {
                "map_id": state.map_id,
                "lines": lines_data,
            }
        )

    # POST - add a new line with coordinates
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Validate required fields
    required = ["start_x", "start_y", "end_x", "end_y"]
    missing = [f for f in required if f not in data]
    if missing:
        return JsonResponse({"error": f"Missing required fields: {missing}"}, status=422)

    # Validate coordinate types
    try:
        start_x = float(data["start_x"])
        start_y = float(data["start_y"])
        end_x = float(data["end_x"])
        end_y = float(data["end_y"])
    except (TypeError, ValueError):
        return JsonResponse({"error": "Coordinates must be numeric"}, status=422)

    line_id = data.get("line_id")

    try:
        line_id = state.add_line(start_x, start_y, end_x, end_y, line_id=line_id)

        return JsonResponse(
            {
                "line_id": line_id,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            }
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=422)


@csrf_exempt
@require_http_methods(["PUT", "DELETE"])
def api_line_detail(request: HttpRequest, line_id: str) -> JsonResponse:
    """Update (PUT) or delete (DELETE) a specific line.

    PUT updates the coordinate endpoints of the line.
    DELETE removes the line.
    """
    state = get_state()

    if request.method == "DELETE":
        try:
            state.delete_line(line_id)
            return JsonResponse({"deleted": line_id})
        except KeyError:
            return JsonResponse({"error": f"Line not found: {line_id}"}, status=404)

    # PUT - update line endpoints
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Validate at least one coordinate field is provided
    coord_fields = ["start_x", "start_y", "end_x", "end_y"]
    if not any(f in data for f in coord_fields):
        return JsonResponse(
            {
                "error": "Must provide at least one coordinate field (start_x, start_y, end_x, end_y)"
            },
            status=422,
        )

    # Get the existing line
    line = state.get_line(line_id)
    if line is None:
        return JsonResponse({"error": f"Line not found: {line_id}"}, status=404)

    # Prepare new coordinates (keep existing if not provided)
    try:
        new_start_x = float(data["start_x"]) if "start_x" in data else line.start_x
        new_start_y = float(data["start_y"]) if "start_y" in data else line.start_y
        new_end_x = float(data["end_x"]) if "end_x" in data else line.end_x
        new_end_y = float(data["end_y"]) if "end_y" in data else line.end_y
    except (TypeError, ValueError):
        return JsonResponse({"error": "Coordinates must be numeric"}, status=422)

    # Validate different endpoints
    if new_start_x == new_end_x and new_start_y == new_end_y:
        return JsonResponse({"error": "Start and end points must be different"}, status=422)

    # Update the line in place
    line.start_x = new_start_x
    line.start_y = new_start_y
    line.end_x = new_end_x
    line.end_y = new_end_y

    return JsonResponse(
        {
            "line_id": line_id,
            "start_x": new_start_x,
            "start_y": new_start_y,
            "end_x": new_end_x,
            "end_y": new_end_y,
        }
    )


@require_GET
def api_next_line_id(request: HttpRequest) -> JsonResponse:
    """Get the next auto-generated line ID."""
    state = get_state()
    next_id = state.get_next_id()
    return JsonResponse({"next_id": next_id})


@require_GET
def api_geo_coords(request: HttpRequest) -> JsonResponse:
    """Convert pixel coordinates to geographic coordinates.

    Query parameters:
        pixel_x: X coordinate in pixels
        pixel_y: Y coordinate in pixels

    Returns:
        easting and northing in map coordinates (if geotransform available)
    """
    state = get_state()

    try:
        pixel_x = float(request.GET.get("pixel_x", 0))
        pixel_y = float(request.GET.get("pixel_y", 0))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid coordinate parameters"}, status=400)

    if state.geotiff is None:
        return JsonResponse(
            {
                "easting": None,
                "northing": None,
                "crs": None,
                "message": "No geotransform available",
            }
        )

    easting, northing = state.geotiff.pixel_to_geo(pixel_x, pixel_y)

    return JsonResponse(
        {
            "easting": float(easting),
            "northing": float(northing),
            "crs": state.geotiff.crs,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def api_export(request: HttpRequest) -> JsonResponse:
    """Save lines to the DDD line repository."""
    state = get_state()
    save_to_line_repo(state.lines, state.map_id, LINES_DIR)
    return JsonResponse({"saved": True, "count": len(state.lines)})


@csrf_exempt
@require_http_methods(["POST"])
def api_import(request: HttpRequest) -> JsonResponse:
    """Load lines from the DDD line repository for a given map_id."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    map_id = data.get("map_id")
    if not map_id:
        return JsonResponse({"error": "map_id is required"}, status=400)

    state = get_state()
    repo_lines = from_line_repo(LINES_DIR, map_id)
    state.lines = repo_lines
    return JsonResponse(
        {"map_id": map_id, "count": len(state.lines)}
    )


@require_GET
def api_registries(request: HttpRequest) -> JsonResponse:
    """List available map IDs from the line repository."""
    return JsonResponse({"map_ids": list_line_map_ids(LINES_DIR)})
