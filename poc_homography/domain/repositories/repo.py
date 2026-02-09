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
    operations like `get_by_map()` or `get_all()`.

    Type Parameters:
        T: The entity type this repository manages.

    Example:
        class UserRepository(Repository[User], Protocol):
            def get_by_email(self, email: str) -> User | None: ...
    """

    def get(self, entity_id: str) -> T | None:
        """Retrieve an entity by its unique identifier.

        Args:
            entity_id: Unique identifier for the entity.

        Returns:
            The entity if found, None otherwise.
        """
        ...

    def save(self, entity: T) -> None:
        """Save an entity (create or update).

        Args:
            entity: The entity to save.
        """
        ...

    def delete(self, entity_id: str) -> bool:
        """Delete an entity by its unique identifier.

        Args:
            entity_id: Unique identifier for the entity.

        Returns:
            True if the entity was deleted, False if it didn't exist.
        """
        ...

    def exists(self, entity_id: str) -> bool:
        """Check if an entity exists.

        Args:
            entity_id: Unique identifier for the entity.

        Returns:
            True if the entity exists, False otherwise.
        """
        ...
