"""Mixin for repositories of entities that belong to a tenant."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.entities.entity import Entity


class MixinRepoTenantFilter:
    """Mixin providing get_by_tenant() for entities with a tenant_id property."""

    def _filter_by(self, field_name: str, value: object) -> dict[str, Any]:
        raise NotImplementedError

    def get_by_tenant(self, tenant_id: str) -> dict[str, "Entity"]:
        """Retrieve all entities for a specific tenant."""
        return self._filter_by("tenant_id", tenant_id)
