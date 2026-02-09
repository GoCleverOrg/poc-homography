"""Base class for YAML-backed repositories."""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Generic, TypeVar

import yaml

from poc_homography.domain.entities.entity import Entity

TEntity = TypeVar("TEntity", bound=Entity)


class RepoYaml(ABC, Generic[TEntity]):
    """Abstract base for YAML-backed single-entity-per-file repositories."""

    def __init__(
        self,
        data_dir: Path,
        entity_cls: type[TEntity],
        *,
        create_dir: bool = True,
        id_separator: str = "/",
        filename_separator: str = "__",
    ) -> None:
        self._data_dir = Path(data_dir)
        self._entity_cls = entity_cls
        if create_dir:
            self._data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, TEntity] = {}
        self._id_sep = id_separator
        self._filename_sep = filename_separator

    def _id_to_path(self, entity_id: str) -> Path:
        filename = entity_id.replace(self._id_sep, self._filename_sep)
        return self._data_dir / f"{filename}.yaml"

    def _path_to_id(self, path: Path) -> str:
        return path.stem.replace(self._filename_sep, self._id_sep)

    def _read_yaml(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if data else None

    def _write_yaml(self, path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _get_entity_id(self, entity: TEntity) -> str:
        return entity.id

    def _entity_to_dict(self, entity: TEntity) -> dict:
        return entity.to_dict()

    def _dict_to_entity(self, data: dict) -> TEntity | None:
        return self._entity_cls.from_dict(data)

    def get(self, entity_id: str) -> TEntity | None:
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
        entity_id = self._get_entity_id(entity)
        path = self._id_to_path(entity_id)
        data = self._entity_to_dict(entity)
        self._write_yaml(path, data)
        self._cache[entity_id] = entity

    def delete(self, entity_id: str) -> bool:
        path = self._id_to_path(entity_id)
        if not path.exists():
            return False

        path.unlink()
        self._cache.pop(entity_id, None)
        return True

    def exists(self, entity_id: str) -> bool:
        if entity_id in self._cache:
            return True
        return self._id_to_path(entity_id).exists()

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_all(self) -> list[TEntity]:
        entities = []
        for path in self._data_dir.glob("*.yaml"):
            entity_id = self._path_to_id(path)
            entity = self.get(entity_id)
            if entity:
                entities.append(entity)
        return entities

    def _filter_by(self, field_name: str, value: object) -> dict[str, TEntity]:
        return {
            self._get_entity_id(e): e for e in self.get_all() if getattr(e, field_name) == value
        }
