"""PostgreSQL-backed GroundControlPoint repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poc_homography.domain.entities.ground_control_point import GroundControlPoint
from poc_homography.infrastructure.models.ground_control_point import (
    GroundControlPointModel,
)
from poc_homography.infrastructure.repositories.base import (
    MixinRepoMapFilterPostgres,
    RepoPostgres,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class RepoPostgresGroundControlPoint(
    MixinRepoMapFilterPostgres,
    RepoPostgres[GroundControlPoint],
):
    """Repository for GroundControlPoint entities stored in PostgreSQL.

    Composite identity: ``{map_id}/{name}`` stored in the ``id`` column.
    The ``map_point`` value object is persisted as JSONB.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, GroundControlPointModel, GroundControlPoint)

    def _entity_to_row(self, entity: GroundControlPoint) -> dict[str, Any]:
        return {
            "id": entity.id,
            "name": entity.name,
            "map_id": entity.map_id,
            "map_point": entity.map_point.to_dict(),
        }
