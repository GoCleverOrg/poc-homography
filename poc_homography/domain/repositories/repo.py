"""Base repository protocol for domain entities.

This module provides a generic repository protocol that defines the
standard CRUD operations for domain entities.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from poc_homography.domain.entities.entity import Entity

T = TypeVar("T", bound=Entity)


class Repo(Protocol[T]):
    """Generic repository protocol for entity persistence.

    Provides the standard CRUD interface that all repositories implement.
    Specific repository protocols can extend this with entity-specific
    operations like ``get_by_map()`` or filtered queries.

    Type Parameters:
        T: The entity type this repository manages.
    """

    def get(self, entity_id: str) -> T | None:
        """Retrieve an entity by its unique identifier."""
        ...

    def save(self, entity: T) -> None:
        """Save an entity (create or update)."""
        ...

    def delete(self, entity_id: str) -> bool:
        """Delete an entity by its unique identifier."""
        ...

    def exists(self, entity_id: str) -> bool:
        """Check if an entity exists."""
        ...

    def get_all(self) -> list[T]:
        """Return all entities in the repository."""
        ...
