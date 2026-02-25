"""
Views for GCP visualization.

Django view wrappers that use the DDD repository for persistence.
"""

from __future__ import annotations

from typing import Any

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from homography_web.frame_utils import (
    GCPS_DIR,
    get_map_repo,
    get_tenant_repo,
)

from poc_homography.map_points import GCPRegistry, MapPoint
from poc_homography.map_points.gcp_registry import from_gcp_repo, list_map_ids


def _load_registry(map_id: str) -> GCPRegistry:
    """Load GCPRegistry from repository.

    Args:
        map_id: Map identifier (required — no cross-tenant fallback).

    Returns:
        Loaded GCPRegistry, or empty registry if directory or map not found.
    """
    if not GCPS_DIR.exists():
        return GCPRegistry(map_id=map_id, points={})

    return from_gcp_repo(GCPS_DIR, map_id)


def _point_to_dict(point_id: str, point: MapPoint) -> dict[str, Any]:
    """Convert a MapPoint to dict with its id included."""
    return {"id": point_id, **point.to_dict()}


def index(request: HttpRequest) -> HttpResponse:
    """Landing page with links to tools and tenant selector."""
    tenants = get_tenant_repo().get_all()
    current_tenant_id = request.GET.get("tenant_id") or (tenants[0].id if tenants else "")

    # Get maps for the current tenant
    maps = get_map_repo().get_by_tenant(current_tenant_id)

    tenant_maps = [m.to_dict() for m in maps.values()]
    return render(
        request,
        "gcp/index.html",
        {
            "title": "Homography GCP Tools",
            "tenants": [t.to_dict() for t in tenants],
            "current_tenant_id": current_tenant_id,
            "tenant_maps": tenant_maps,
            "has_maps": bool(tenant_maps),
        },
    )


def debug_map(request: HttpRequest) -> HttpResponse:
    """
    Debug visualization for MapPoint data.

    Displays MapPoints from repository for verification.
    Since MapPoints use pixel coordinates (not GPS), this shows a
    list view rather than a geographic map.
    """
    map_id = request.GET.get("map_id", "")
    if not map_id:
        return render(request, "gcp/debug_map.html", {"map_id": "", "points": [], "point_count": 0})
    registry = _load_registry(map_id)

    points_data = [_point_to_dict(pid, p) for pid, p in registry.points.items()]

    context: dict[str, Any] = {
        "map_id": registry.map_id,
        "points": points_data,
        "point_count": len(points_data),
    }

    return render(request, "gcp/debug_map.html", context)


@require_GET
def api_tenants(request: HttpRequest) -> JsonResponse:
    """Return available tenants from the DDD repository."""
    tenants = get_tenant_repo().get_all()
    return JsonResponse(
        {"tenants": [t.to_dict() for t in tenants]},
    )


@require_GET
def api_tenant_maps(request: HttpRequest, tenant_id: str) -> JsonResponse:
    """Return maps belonging to a specific tenant."""
    maps = get_map_repo().get_by_tenant(tenant_id)
    return JsonResponse(
        {"maps": [m.to_dict() for m in maps.values()]},
    )


@require_GET
def api_map_ids(request: HttpRequest) -> JsonResponse:
    """Return available map IDs from the GCP repository."""
    if not GCPS_DIR.exists():
        return JsonResponse({"map_ids": []})
    return JsonResponse({"map_ids": list_map_ids(GCPS_DIR)})
