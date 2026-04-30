"""FastAPI router for GCP-related endpoints (tenants, maps, map IDs)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.deps import get_current_user, get_db_session
from api.schemas.gcp import (
    MapIdsResponse,
    MapListResponse,
    TenantListResponse,
)
from poc_homography.infrastructure.models.user import UserModel
from poc_homography.infrastructure.repositories import (
    RepoPostgresGroundControlPoint,
    RepoPostgresMap,
    RepoPostgresTenant,
)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/gcp", tags=["gcp"])


@router.get("/api/tenants/", response_model=TenantListResponse)
def get_tenants(
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> TenantListResponse:
    """List all tenants."""
    repo = RepoPostgresTenant(session)
    tenants = repo.get_all()
    return TenantListResponse(tenants=[t.to_dict() for t in tenants])


@router.get("/api/tenants/{tenant_id}/maps/", response_model=MapListResponse)
def get_tenant_maps(
    tenant_id: str,
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> MapListResponse:
    """List maps for a tenant."""
    repo = RepoPostgresMap(session)
    maps = repo.get_by_tenant(tenant_id)
    return MapListResponse(maps=[m.to_dict() for m in maps.values()])


@router.get("/api/map-ids/", response_model=MapIdsResponse)
def get_map_ids(
    tenant_id: str = Query(...),
    session: Session = Depends(get_db_session),
    user: UserModel = Depends(get_current_user),
) -> MapIdsResponse:
    """List map IDs filtered by tenant, intersected with available GCPs."""
    tenant_maps = RepoPostgresMap(session).get_by_tenant(tenant_id)
    tenant_map_ids = set(tenant_maps)

    gcp_repo = RepoPostgresGroundControlPoint(session)
    all_gcp_map_ids = sorted({gcp.map_id for gcp in gcp_repo.get_all()})
    return MapIdsResponse(map_ids=[mid for mid in all_gcp_map_ids if mid in tenant_map_ids])
