"""Base class for PostgreSQL-backed repositories."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from sqlalchemy import inspect, select

from poc_homography.domain.entities.entity import Entity

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.infrastructure.database import Base

TEntity = TypeVar("TEntity", bound=Entity)


class RepoPostgres(ABC, Generic[TEntity]):
    """Abstract base for PostgreSQL-backed repositories using SQLAlchemy 2.0."""

    def __init__(
        self,
        session: Session,
        model_cls: type[Base],
        entity_cls: type[TEntity],
    ) -> None:
        self._session = session
        self._model_cls = model_cls
        self._entity_cls = entity_cls

    def _entity_to_row(self, entity: TEntity) -> dict[str, Any]:
        return entity.to_dict()

    def _row_to_entity(self, row: Base) -> TEntity:
        mapper = inspect(type(row))
        data = {col.key: getattr(row, col.key) for col in mapper.column_attrs}
        return self._entity_cls.from_dict(data)

    def get(self, entity_id: str) -> TEntity | None:
        row = self._session.get(self._model_cls, entity_id)
        if row is None:
            return None
        return self._row_to_entity(row)

    def save(self, entity: TEntity) -> None:
        data = self._entity_to_row(entity)
        row = self._session.get(self._model_cls, entity.id)
        if row is None:
            row = self._model_cls(**data)
            self._session.add(row)
        else:
            for key, value in data.items():
                setattr(row, key, value)
        self._session.flush()

    def delete(self, entity_id: str) -> bool:
        row = self._session.get(self._model_cls, entity_id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True

    def exists(self, entity_id: str) -> bool:
        return self._session.get(self._model_cls, entity_id) is not None

    def get_all(self) -> list[TEntity]:
        stmt = select(self._model_cls)
        rows = self._session.execute(stmt).scalars().all()
        return [self._row_to_entity(row) for row in rows]

    def _filter_by(self, field_name: str, value: object) -> dict[str, TEntity]:
        column = getattr(self._model_cls, field_name)
        stmt = select(self._model_cls).where(column == value)
        rows = self._session.execute(stmt).scalars().all()
        return {entity.id: entity for entity in (self._row_to_entity(row) for row in rows)}
