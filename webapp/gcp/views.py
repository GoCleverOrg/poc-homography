"""
Views for GCP visualization.

Django view wrappers that use the DDD repository for persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render

from poc_homography.map_points import GCPRegistry, MapPoint
from poc_homography.map_points.gcp_registry import from_gcp_repo, list_map_ids

# Project root and data directories
# Path: views.py -> gcp/ -> webapp/ -> project_root/
PROJECT_ROOT = Path(__file__).parent.parent.parent
GCPS_DIR = PROJECT_ROOT / "data" / "gcps"


def _load_registry(map_id: str | None = None) -> GCPRegistry:
    """Load GCPRegistry from repository.

    Args:
        map_id: Map identifier. If None, uses the first available map.

    Returns:
        Loaded GCPRegistry, or empty registry if no maps exist.
    """
    if not GCPS_DIR.exists():
        return GCPRegistry(map_id="default", points={})

    if map_id is None:
        available = list_map_ids(GCPS_DIR)
        if not available:
            return GCPRegistry(map_id="default", points={})
        map_id = available[0]

    return from_gcp_repo(GCPS_DIR, map_id)


def _point_to_dict(point_id: str, point: MapPoint) -> dict[str, Any]:
    """Convert a MapPoint to dict with its id included."""
    return {"id": point_id, **point.to_dict()}


def index(request: HttpRequest) -> HttpResponse:
    """Landing page with links to tools."""
    return render(request, "gcp/index.html", {"title": "Homography GCP Tools"})


def debug_map(request: HttpRequest) -> HttpResponse:
    """
    Debug visualization for MapPoint data.

    Displays MapPoints from repository for verification.
    Since MapPoints use pixel coordinates (not GPS), this shows a
    list view rather than a geographic map.
    """
    map_id = request.GET.get("map_id")
    registry = _load_registry(map_id)

    points_data = [_point_to_dict(pid, p) for pid, p in registry.points.items()]

    context: dict[str, Any] = {
        "map_id": registry.map_id,
        "points": points_data,
        "point_count": len(points_data),
    }

    return render(request, "gcp/debug_map.html", context)


def api_map_ids(request: HttpRequest) -> JsonResponse:
    """Return available map IDs from the GCP repository."""
    if not GCPS_DIR.exists():
        return JsonResponse({"map_ids": []})
    return JsonResponse({"map_ids": list_map_ids(GCPS_DIR)})
