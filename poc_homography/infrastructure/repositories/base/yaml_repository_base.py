"""Base class for YAML-backed repositories with caching.

This module provides a reusable foundation for repositories that store
domain entities in YAML files with in-memory caching.
"""

from abc import ABC
from pathlib import Path
from typing import Generic, TypeVar

import yaml

from poc_homography.domain.entities.entity import Entity

TEntity = TypeVar("TEntity", bound=Entity)


class YamlRepositoryBase(ABC, Generic[TEntity]):
    """Abstract base class for YAML-backed single-entity-per-file repositories.

    Provides common functionality:
    - Directory management and creation
    - In-memory caching with cache-first reads
    - ID-to-path conversion with configurable separators
    - YAML read/write with consistent formatting
    - Standard CRUD operations (get, save, delete, exists)

    Subclasses only need to provide the entity class at construction time.
    The entity class must implement the Entity protocol (id, to_dict, from_dict).

    Example usage:
        class YamlUserRepository(YamlRepositoryBase[User]):
            def __init__(self, data_dir: Path) -> None:
                super().__init__(data_dir, User)
    """

    def __init__(
        self,
        data_dir: Path,
        entity_cls: type[TEntity],
        *,
        create_dir: bool = True,
        id_separator: str = "/",
        filename_separator: str = "__",
    ) -> None:
        """Initialize the repository.

        Args:
            data_dir: Directory for storing YAML files.
            entity_cls: The entity class (must have from_dict class method).
            create_dir: If True, create data_dir if it doesn't exist.
            id_separator: Character used in entity IDs (e.g., "/" in "map_id/name").
            filename_separator: Replacement for id_separator in filenames.
        """
        self._data_dir = Path(data_dir)
        self._entity_cls = entity_cls
        if create_dir:
            self._data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, TEntity] = {}
        self._id_sep = id_separator
        self._filename_sep = filename_separator

    # ─── Path Resolution ───────────────────────────────────────────────

    def _id_to_path(self, entity_id: str) -> Path:
        """Convert entity ID to filesystem path.

        Args:
            entity_id: The entity's unique identifier.

        Returns:
            Path to the YAML file for this entity.
        """
        filename = entity_id.replace(self._id_sep, self._filename_sep)
        return self._data_dir / f"{filename}.yaml"

    def _path_to_id(self, path: Path) -> str:
        """Convert filesystem path back to entity ID.

        Args:
            path: Path to a YAML file.

        Returns:
            The entity ID derived from the filename.
        """
        return path.stem.replace(self._filename_sep, self._id_sep)

    # ─── YAML I/O ──────────────────────────────────────────────────────

    def _read_yaml(self, path: Path) -> dict | None:
        """Read YAML file safely.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed dictionary, or None if file is empty or doesn't exist.
        """
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if data else None

    def _write_yaml(self, path: Path, data: dict) -> None:
        """Write data to YAML with consistent formatting.

        Args:
            path: Path to the YAML file.
            data: Dictionary to serialize.
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # ─── Entity Conversion (Default Implementations) ───────────────────

    def _get_entity_id(self, entity: TEntity) -> str:
        """Extract the unique ID from an entity.

        Default implementation uses entity.id property.

        Args:
            entity: The domain entity.

        Returns:
            The entity's unique identifier.
        """
        return entity.id

    def _entity_to_dict(self, entity: TEntity) -> dict:
        """Convert domain entity to dictionary for YAML storage.

        Default implementation uses entity.to_dict() method.

        Args:
            entity: The domain entity to serialize.

        Returns:
            Dictionary representation suitable for YAML.
        """
        return entity.to_dict()

    def _dict_to_entity(self, data: dict) -> TEntity | None:
        """Reconstruct domain entity from YAML dictionary.

        Default implementation uses entity_cls.from_dict() class method.

        Args:
            data: Dictionary loaded from YAML.

        Returns:
            The reconstructed entity, or None if data is invalid.
        """
        return self._entity_cls.from_dict(data)

    # ─── CRUD Operations ───────────────────────────────────────────────

    def get(self, entity_id: str) -> TEntity | None:
        """Retrieve entity by ID, using cache if available.

        Args:
            entity_id: Unique identifier for the entity.

        Returns:
            The entity if found, None otherwise.
        """
        if entity_id in self._cache:
            return self._cache[entity_id]

        path = self._id_to_path(entity_id)
        data = self._read_yaml(path)
        if data is None:
            return None

        entity = self._dict_to_entity(data)
        if entity:
            self._cache[entity_id] = entity
        return entity

    def save(self, entity: TEntity) -> None:
        """Save entity to YAML and update cache.

        Args:
            entity: The entity to save.
        """
        entity_id = self._get_entity_id(entity)
        path = self._id_to_path(entity_id)
        data = self._entity_to_dict(entity)
        self._write_yaml(path, data)
        self._cache[entity_id] = entity

    def delete(self, entity_id: str) -> bool:
        """Delete entity file and remove from cache.

        Args:
            entity_id: Unique identifier for the entity.

        Returns:
            True if the entity was deleted, False if it didn't exist.
        """
        path = self._id_to_path(entity_id)
        if not path.exists():
            return False

        path.unlink()
        self._cache.pop(entity_id, None)
        return True

    def exists(self, entity_id: str) -> bool:
        """Check if entity exists in cache or filesystem.

        Args:
            entity_id: Unique identifier for the entity.

        Returns:
            True if the entity exists, False otherwise.
        """
        if entity_id in self._cache:
            return True
        return self._id_to_path(entity_id).exists()

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()

    # ─── Batch Operations ──────────────────────────────────────────────

    def get_all(self) -> list[TEntity]:
        """Load all entities from the data directory.

        Returns:
            List of all entities found in YAML files.
        """
        entities = []
        for path in self._data_dir.glob("*.yaml"):
            entity_id = self._path_to_id(path)
            entity = self.get(entity_id)
            if entity:
                entities.append(entity)
        return entities

    def get_by_prefix(self, prefix: str) -> dict[str, TEntity]:
        """Get all entities whose IDs start with a prefix.

        Args:
            prefix: The ID prefix to filter by.

        Returns:
            Dictionary mapping entity IDs to entities.
        """
        result: dict[str, TEntity] = {}
        safe_prefix = prefix.replace(self._id_sep, self._filename_sep)

        for path in self._data_dir.glob(f"{safe_prefix}*.yaml"):
            entity_id = self._path_to_id(path)
            entity = self.get(entity_id)
            if entity:
                result[entity_id] = entity
        return result
