"""Mixin for repositories of entities that belong to a map."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.entities.entity import Entity


class MixinRepoMapFilter:
    """Mixin providing get_by_map() for entities with a map_id property."""

    def _filter_by(self, field_name: str, value: object) -> dict[str, Any]:
        raise NotImplementedError

    def get_by_map(self, map_id: str) -> dict[str, "Entity"]:
        """Retrieve all entities for a specific map."""
        return self._filter_by("map_id", map_id)
