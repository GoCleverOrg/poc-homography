"""
Views for GCP capture and visualization.

Django view wrappers that use GroundControlPointCollection for persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from poc_homography.map_points import GroundControlPointCollection, MapPoint

# Project root and data directories
# Path: views.py -> gcp/ -> webapp/ -> project_root/
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# Default map points file at project root (can also use data/gcps/ for custom files)
MAP_POINTS_FILE = PROJECT_ROOT / "valte_map_points.yaml"


def _ensure_data_dir() -> None:
    """Ensure data directory exists."""
    gcps_dir = DATA_DIR / "gcps"
    gcps_dir.mkdir(parents=True, exist_ok=True)


def _resolve_safe_file_path(filename: str) -> Path | None:
    """
    Safely resolve a filename to a path within the allowed gcps directory.

    Prevents path traversal attacks by:
    1. Extracting only the filename (stripping any directory components)
    2. Constructing the path within DATA_DIR/gcps
    3. Validating the resolved path is still within DATA_DIR

    Args:
        filename: User-provided filename (may contain malicious path components)

    Returns:
        Safe resolved Path within gcps directory, or None if invalid
    """
    # Extract just the filename, stripping any directory components
    safe_name = Path(filename).name

    # Reject empty filenames or hidden files
    if not safe_name or safe_name.startswith("."):
        return None

    # Construct path within allowed directory
    gcps_dir = DATA_DIR / "gcps"
    file_path = (gcps_dir / safe_name).resolve()

    # Verify the resolved path is still within DATA_DIR
    try:
        file_path.relative_to(DATA_DIR.resolve())
    except ValueError:
        return None

    return file_path


def _load_registry() -> GroundControlPointCollection:
    """Load GroundControlPointCollection from disk or return empty registry."""
    if MAP_POINTS_FILE.exists():
        return GroundControlPointCollection.load(MAP_POINTS_FILE)
    return GroundControlPointCollection(map_id="default", points={})


def _save_registry(registry: GroundControlPointCollection) -> None:
    """Save GroundControlPointCollection to disk."""
    _ensure_data_dir()
    registry.save(MAP_POINTS_FILE)


def _point_to_dict(point_id: str, point: MapPoint) -> dict[str, Any]:
    """Convert a MapPoint to dict with its id included."""
    return {"id": point_id, **point.to_dict()}


def index(request: HttpRequest) -> HttpResponse:
    """Landing page with links to tools."""
    return render(request, "gcp/index.html", {"title": "Homography GCP Tools"})


def gcp_capture(request: HttpRequest) -> HttpResponse:
    """GCP capture interface for marking points on satellite map."""
    # Load existing points to display
    registry = _load_registry()
    points_data = [_point_to_dict(pid, p) for pid, p in registry.points.items()]

    context: dict[str, Any] = {
        "title": "GCP Capture Tool",
        "map_id": registry.map_id,
        "points": points_data,
    }
    return render(request, "gcp/gcp_capture.html", context)


def debug_map(request: HttpRequest) -> HttpResponse:
    """
    Debug visualization for MapPoint data.

    Displays MapPoints from storage for verification.
    Since MapPoints use pixel coordinates (not GPS), this shows a
    list view rather than a geographic map.
    """
    # Load registry from file parameter or default
    # Uses safe path resolution to prevent path traversal attacks
    file_param = request.GET.get("file")
    if file_param:
        file_path = _resolve_safe_file_path(file_param)
        if file_path and file_path.exists():
            registry = GroundControlPointCollection.load(file_path)
        else:
            registry = _load_registry()
    else:
        registry = _load_registry()

    points_data = [_point_to_dict(pid, p) for pid, p in registry.points.items()]

    context: dict[str, Any] = {
        "map_id": registry.map_id,
        "points": points_data,
        "point_count": len(points_data),
    }

    return render(request, "gcp/debug_map.html", context)


@require_GET
def api_get_gcps(request: HttpRequest) -> JsonResponse:
    """
    API endpoint to get MapPoints from storage.

    Returns:
        JSON with success status and list of MapPoint data.
    """
    registry = _load_registry()
    points = [_point_to_dict(pid, p) for pid, p in registry.points.items()]

    return JsonResponse(
        {
            "success": True,
            "map_id": registry.map_id,
            "points": points,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def api_save_gcps(request: HttpRequest) -> JsonResponse:
    """
    API endpoint to save MapPoints to storage.

    Expects JSON body with:
        - map_id: str - Map identifier
        - points: list - List of point objects with id, pixel_x, pixel_y, map_id
    """
    data = json.loads(request.body)
    map_id = data.get("map_id", "default")
    points_data = data.get("points", [])

    # Build MapPoints from data
    points: dict[str, MapPoint] = {}
    for p in points_data:
        point_id = str(p["id"])  # Extract id
        point = MapPoint.from_dict(p)  # MapPoint ignores the id field
        points[point_id] = point  # Use id as key

    # Create and save registry
    registry = GroundControlPointCollection(map_id=map_id, points=points)
    _save_registry(registry)

    return JsonResponse(
        {
            "success": True,
            "message": f"Saved {len(points)} MapPoints",
            "map_id": map_id,
        }
    )


@csrf_exempt
@require_http_methods(["DELETE"])
def api_delete_gcp(request: HttpRequest, gcp_id: int) -> JsonResponse:
    """
    API endpoint to delete a specific MapPoint.

    Args:
        gcp_id: ID of the MapPoint to delete (passed as int but used as string key)
    """
    point_id = str(gcp_id)
    registry = _load_registry()

    if point_id not in registry.points:
        return JsonResponse(
            {"success": False, "error": f"MapPoint {point_id} not found"},
            status=404,
        )

    # Create new registry without the deleted point
    new_points = {k: v for k, v in registry.points.items() if k != point_id}
    new_registry = GroundControlPointCollection(map_id=registry.map_id, points=new_points)
    _save_registry(new_registry)

    return JsonResponse(
        {
            "success": True,
            "message": f"Deleted MapPoint {point_id}",
        }
    )
