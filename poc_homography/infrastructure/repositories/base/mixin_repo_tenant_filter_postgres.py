"""Mixin for PostgreSQL repositories of entities that belong to a tenant."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.entities.entity import Entity


class MixinRepoTenantFilterPostgres:
    """Mixin providing get_by_tenant() for Postgres-backed repositories."""

    def _filter_by(self, field_name: str, value: object) -> dict[str, Any]:
        raise NotImplementedError

    def get_by_tenant(self, tenant_id: str) -> dict[str, Entity]:
        return self._filter_by("tenant_id", tenant_id)
