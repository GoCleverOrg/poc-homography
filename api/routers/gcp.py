"""FastAPI router for GCP-related endpoints (tenants, maps, map IDs)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query

from api.deps import get_current_user
from api.schemas.gcp import (
    MapIdsResponse,
    MapListResponse,
    TenantListResponse,
)
from poc_homography.infrastructure.models.user import UserModel
from poc_homography.infrastructure.repositories import RepoYamlMap, RepoYamlTenant
from poc_homography.map_points.gcp_registry import list_map_ids

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
_GCPS_DIR = _DATA_DIR / "gcps"

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/gcp", tags=["gcp"])


@router.get("/api/tenants/", response_model=TenantListResponse)
def get_tenants(
    user: UserModel = Depends(get_current_user),
) -> TenantListResponse:
    """List all tenants."""
    repo = RepoYamlTenant(_DATA_DIR / "tenants")
    tenants = repo.get_all()
    return TenantListResponse(tenants=[t.to_dict() for t in tenants])


@router.get("/api/tenants/{tenant_id}/maps/", response_model=MapListResponse)
def get_tenant_maps(
    tenant_id: str,
    user: UserModel = Depends(get_current_user),
) -> MapListResponse:
    """List maps for a tenant."""
    repo = RepoYamlMap(_DATA_DIR / "maps")
    maps = repo.get_by_tenant(tenant_id)
    return MapListResponse(maps=[m.to_dict() for m in maps.values()])


@router.get("/api/map-ids/", response_model=MapIdsResponse)
def get_map_ids(
    tenant_id: str = Query(...),
    user: UserModel = Depends(get_current_user),
) -> MapIdsResponse:
    """List map IDs filtered by tenant, intersected with available GCPs."""
    if not _GCPS_DIR.exists():
        return MapIdsResponse(map_ids=[])

    tenant_maps = RepoYamlMap(_DATA_DIR / "maps").get_by_tenant(tenant_id)
    tenant_map_ids = set(tenant_maps)
    all_ids = list_map_ids(_GCPS_DIR)
    return MapIdsResponse(map_ids=[mid for mid in all_ids if mid in tenant_map_ids])
