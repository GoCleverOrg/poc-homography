"""
Views for GCP visualization.

Django view wrappers that use the DDD repository for persistence.
"""

from __future__ import annotations

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from homography_web.frame_utils import (
    GCPS_DIR,
    get_map_repo,
    get_tenant_id,
    get_tenant_repo,
)

from poc_homography.map_points.gcp_registry import list_map_ids


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
    """Return available map IDs from the GCP repository, filtered by tenant."""
    if not GCPS_DIR.exists():
        return JsonResponse({"map_ids": []})
    tenant_id = get_tenant_id(request)
    tenant_maps = get_map_repo().get_by_tenant(tenant_id)
    tenant_map_ids = set(tenant_maps)
    all_ids = list_map_ids(GCPS_DIR)
    return JsonResponse({"map_ids": [mid for mid in all_ids if mid in tenant_map_ids]})
