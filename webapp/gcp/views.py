"""
Views for GCP visualization.

Django view wrappers that use GCPRegistry for persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from poc_homography.map_points import MapPoint, GCPRegistry

# Project root and data directories
# Path: views.py -> gcp/ -> webapp/ -> project_root/
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# Default map points file at project root (can also use data/gcps/ for custom files)
MAP_POINTS_FILE = PROJECT_ROOT / "valte_map_points.yaml"


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


def _load_registry() -> GCPRegistry:
    """Load GCPRegistry from disk or return empty registry."""
    if MAP_POINTS_FILE.exists():
        return GCPRegistry.load(MAP_POINTS_FILE)
    return GCPRegistry(map_id="default", points={})


def _point_to_dict(point_id: str, point: MapPoint) -> dict[str, Any]:
    """Convert a MapPoint to dict with its id included."""
    return {"id": point_id, **point.to_dict()}


def index(request: HttpRequest) -> HttpResponse:
    """Landing page with links to tools."""
    return render(request, "gcp/index.html", {"title": "Homography GCP Tools"})


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
            registry = GCPRegistry.load(file_path)
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
