"""PostgreSQL-backed Map repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.domain.entities.map import Map
from poc_homography.infrastructure.models.map import MapModel
from poc_homography.infrastructure.repositories.base import (
    MixinRepoTenantFilter,
    RepoPostgres,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class RepoPostgresMap(MixinRepoTenantFilter, RepoPostgres[Map]):
    """Repository for Map entities stored in PostgreSQL.

    The ``photo`` and ``geotiff`` value objects are stored as JSONB columns.
    ``Map.to_dict()`` already produces the correct shape (``{"photo": {...},
    "geotiff": {...}}``) so no serialisation override is needed.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, MapModel, Map)
