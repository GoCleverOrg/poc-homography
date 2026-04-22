"""PostgreSQL-backed Line repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from poc_homography.domain.entities.line import Line
from poc_homography.infrastructure.models.line import LineModel
from poc_homography.infrastructure.repositories.base import (
    MixinRepoMapFilter,
    RepoPostgres,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class RepoPostgresLine(
    MixinRepoMapFilter,
    RepoPostgres[Line],
):
    """Repository for Line entities stored in PostgreSQL.

    Composite identity: ``{map_id}/{name}`` stored in the ``id`` column.
    The ``start`` and ``end`` PixelPoint value objects are persisted as JSONB.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, LineModel, Line)

    def _entity_to_row(self, entity: Line) -> dict[str, Any]:
        return {
            "id": entity.id,
            "name": entity.name,
            "map_id": entity.map_id,
            "start": entity.start.to_dict(),
            "end": entity.end.to_dict(),
        }

    def get_by_map_id(self, map_id: str) -> list[Line]:
        """Return all lines belonging to *map_id*."""
        stmt = select(self._model_cls).where(LineModel.map_id == map_id)
        rows = self._session.execute(stmt).scalars().all()
        return [self._row_to_entity(row) for row in rows]
