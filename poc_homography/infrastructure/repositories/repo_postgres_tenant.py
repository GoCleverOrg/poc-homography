"""PostgreSQL-backed Tenant repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poc_homography.domain.entities.tenant import Tenant
from poc_homography.infrastructure.models.tenant import TenantModel
from poc_homography.infrastructure.repositories.base import RepoPostgres

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.infrastructure.database import Base


class RepoPostgresTenant(RepoPostgres[Tenant]):
    """Repository for Tenant entities stored in PostgreSQL."""

    def __init__(self, session: Session) -> None:
        super().__init__(session, TenantModel, Tenant)

    # -- serialisation overrides ------------------------------------------
    # Tenant.to_dict() nests location as {"location": {"lat": ..., "lon": ...}}
    # and omits empty optional fields, but the ORM model has flat columns
    # (description, location_lat, location_lon).

    def _entity_to_row(self, entity: Tenant) -> dict[str, Any]:
        return {
            "id": entity.id,
            "name": entity.name,
            "description": entity.description,
            "location_lat": entity.location_lat,
            "location_lon": entity.location_lon,
        }

    def _row_to_entity(self, row: Base) -> Tenant:
        return Tenant(
            id=row.id,  # type: ignore[attr-defined]
            name=row.name,  # type: ignore[attr-defined]
            description=row.description,  # type: ignore[attr-defined]
            location_lat=row.location_lat,  # type: ignore[attr-defined]
            location_lon=row.location_lon,  # type: ignore[attr-defined]
        )
