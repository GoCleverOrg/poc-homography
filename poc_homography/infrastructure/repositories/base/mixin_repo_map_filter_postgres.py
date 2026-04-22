"""Mixin for PostgreSQL repositories of entities that belong to a map."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.entities.entity import Entity


class MixinRepoMapFilterPostgres:
    """Mixin providing get_by_map() for Postgres-backed repositories."""

    def _filter_by(self, field_name: str, value: object) -> dict[str, Any]:
        raise NotImplementedError

    def get_by_map(self, map_id: str) -> dict[str, Entity]:
        return self._filter_by("map_id", map_id)
