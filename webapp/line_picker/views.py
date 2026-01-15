"""Views for the line picker Django app."""

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

from .state import get_state
from .validation import (
    validate_add_line_request,
    validate_export_request,
    validate_import_request,
)


def normalize_array(arr: np.ndarray) -> np.ndarray:
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


def get_tag_from_id(point_id: str) -> str:
    """Extract tag from point ID prefix.

    Args:
        point_id: Point ID (e.g., "PS1", "AR2").

    Returns:
        Tag abbreviation (e.g., "PS", "AR").
    """
    # Extract prefix by finding where the digits start
    for i, char in enumerate(point_id):
        if char.isdigit():
            return point_id[:i]
    return point_id


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
            "geotransform": state.geotransform,
            "crs": state.crs,
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


@require_GET
def api_gcps(request: HttpRequest) -> JsonResponse:
    """Get all GCPs from the loaded registry (read-only).

    Returns GCP data with pixel coordinates for displaying as clickable markers.
    """
    state = get_state()

    return JsonResponse(
        {
            "map_id": state.gcp_registry.map_id,
            "points": [
                {
                    "id": pid,
                    "pixel_x": p.pixel_x,
                    "pixel_y": p.pixel_y,
                    "tag": get_tag_from_id(pid),
                }
                for pid, p in state.gcp_registry.points.items()
            ],
        }
    )


@csrf_exempt
@require_http_methods(["GET", "POST"])
def api_lines(request: HttpRequest) -> JsonResponse:
    """Get all lines (GET) or add a new line (POST).

    GET response includes pixel coordinates of the GCP endpoints for rendering.
    """
    state = get_state()

    if request.method == "GET":
        # Build line data with pixel coordinates
        lines_data = []
        for line in state.lines:
            start_point = state.gcp_registry.points.get(line.start_gcp)
            end_point = state.gcp_registry.points.get(line.end_gcp)

            # Skip lines with missing GCPs (shouldn't happen but defensive)
            if start_point is None or end_point is None:
                continue

            lines_data.append(
                {
                    "line_id": line.line_id,
                    "start_gcp": line.start_gcp,
                    "end_gcp": line.end_gcp,
                    "start_pixel": {"x": start_point.pixel_x, "y": start_point.pixel_y},
                    "end_pixel": {"x": end_point.pixel_x, "y": end_point.pixel_y},
                }
            )

        return JsonResponse(
            {
                "map_id": state.map_id,
                "lines": lines_data,
            }
        )

    # POST - add a new line
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    error = validate_add_line_request(data)
    if error:
        return JsonResponse({"error": error}, status=422)

    start_gcp = data["start_gcp"]
    end_gcp = data["end_gcp"]
    line_id = data.get("line_id")

    try:
        line_id = state.add_line(start_gcp, end_gcp, line_id=line_id)

        # Get the created line and include pixel coordinates
        start_point = state.gcp_registry.points[start_gcp]
        end_point = state.gcp_registry.points[end_gcp]

        return JsonResponse(
            {
                "line_id": line_id,
                "start_gcp": start_gcp,
                "end_gcp": end_gcp,
                "start_pixel": {"x": start_point.pixel_x, "y": start_point.pixel_y},
                "end_pixel": {"x": end_point.pixel_x, "y": end_point.pixel_y},
            }
        )
    except KeyError as e:
        return JsonResponse({"error": str(e)}, status=404)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=422)


@csrf_exempt
@require_http_methods(["PUT", "DELETE"])
def api_line_detail(request: HttpRequest, line_id: str) -> JsonResponse:
    """Update (PUT) or delete (DELETE) a specific line.

    PUT updates the start_gcp and/or end_gcp of the line.
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

    # Validate at least one field is provided
    if "start_gcp" not in data and "end_gcp" not in data:
        return JsonResponse({"error": "Must provide start_gcp and/or end_gcp"}, status=422)

    # Get the existing line
    line = state.get_line(line_id)
    if line is None:
        return JsonResponse({"error": f"Line not found: {line_id}"}, status=404)

    # Prepare new endpoints (keep existing if not provided)
    new_start_gcp = data.get("start_gcp", line.start_gcp)
    new_end_gcp = data.get("end_gcp", line.end_gcp)

    # Validate types
    if not isinstance(new_start_gcp, str):
        return JsonResponse({"error": "start_gcp must be a string"}, status=422)
    if not isinstance(new_end_gcp, str):
        return JsonResponse({"error": "end_gcp must be a string"}, status=422)

    # Validate GCPs exist
    if new_start_gcp not in state.gcp_registry.points:
        return JsonResponse({"error": f"Start GCP not found: {new_start_gcp}"}, status=404)
    if new_end_gcp not in state.gcp_registry.points:
        return JsonResponse({"error": f"End GCP not found: {new_end_gcp}"}, status=404)

    # Validate different endpoints
    if new_start_gcp == new_end_gcp:
        return JsonResponse({"error": "Start and end GCP must be different"}, status=422)

    # Update the line in place
    line.start_gcp = new_start_gcp
    line.end_gcp = new_end_gcp

    # Return updated line with pixel coordinates
    start_point = state.gcp_registry.points[new_start_gcp]
    end_point = state.gcp_registry.points[new_end_gcp]

    return JsonResponse(
        {
            "line_id": line_id,
            "start_gcp": new_start_gcp,
            "end_gcp": new_end_gcp,
            "start_pixel": {"x": start_point.pixel_x, "y": start_point.pixel_y},
            "end_pixel": {"x": end_point.pixel_x, "y": end_point.pixel_y},
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

    if state.geotransform is None:
        return JsonResponse(
            {
                "easting": None,
                "northing": None,
                "crs": None,
                "message": "No geotransform available",
            }
        )

    # Apply geotransform: [origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height]
    gt = state.geotransform
    easting = gt[0] + pixel_x * gt[1] + pixel_y * gt[2]
    northing = gt[3] + pixel_x * gt[4] + pixel_y * gt[5]

    return JsonResponse(
        {
            "easting": easting,
            "northing": northing,
            "crs": state.crs,
        }
    )


def _validate_safe_path(path: Path, allowed_base: Path) -> Path | None:
    """Validate that a path is within the allowed base directory.

    Args:
        path: The path to validate (can be relative or absolute).
        allowed_base: The base directory that the path must be within.

    Returns:
        The resolved absolute path if safe, None if path traversal detected.
    """
    # Resolve to absolute path
    if not path.is_absolute():
        resolved = (allowed_base / path).resolve()
    else:
        resolved = path.resolve()

    # Check that the resolved path is within the allowed base
    try:
        resolved.relative_to(allowed_base.resolve())
        return resolved
    except ValueError:
        return None


@csrf_exempt
@require_http_methods(["POST"])
def api_export(request: HttpRequest) -> JsonResponse:
    """Export lines to YAML file."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    error = validate_export_request(data)
    if error:
        return JsonResponse({"error": error}, status=422)

    state = get_state()
    path = Path(data.get("path", "")) if data.get("path") else Path(f"{state.map_id}_lines.yaml")

    # Validate path is within allowed directory (same directory as GCP registry or image)
    allowed_base = (
        state.gcp_registry_path.parent if state.gcp_registry_path else state.geotiff_path.parent
    )
    safe_path = _validate_safe_path(path, allowed_base)
    if safe_path is None:
        return JsonResponse(
            {"error": "Invalid path: path traversal not allowed"},
            status=400,
        )

    state.save_lines(safe_path)
    return JsonResponse({"exported": str(safe_path), "count": len(state.lines)})


@csrf_exempt
@require_http_methods(["POST"])
def api_import(request: HttpRequest) -> JsonResponse:
    """Import lines from YAML file."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    error = validate_import_request(data)
    if error:
        return JsonResponse({"error": error}, status=422)

    state = get_state()
    path = Path(data["path"])

    # Validate path is within allowed directory (same directory as GCP registry or image)
    allowed_base = (
        state.gcp_registry_path.parent if state.gcp_registry_path else state.geotiff_path.parent
    )
    safe_path = _validate_safe_path(path, allowed_base)
    if safe_path is None:
        return JsonResponse(
            {"error": "Invalid path: path traversal not allowed"},
            status=400,
        )

    if not safe_path.exists():
        return JsonResponse({"error": f"File not found: {safe_path}"}, status=404)

    try:
        state.load_lines(safe_path)
        return JsonResponse(
            {
                "imported": str(safe_path),
                "count": len(state.lines),
                "map_id": state.map_id,
            }
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=422)
    except KeyError as e:
        return JsonResponse({"error": str(e)}, status=404)
