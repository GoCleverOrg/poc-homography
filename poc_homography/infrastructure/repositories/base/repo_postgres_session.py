"""Base class for PostgreSQL-backed session repositories.

Session entities (DiagnosticSession, StressTestSession, SurveySession) live
in the webapp layer as untyped ``Any`` rather than domain ``Entity``
subclasses, so they cannot use ``RepoPostgres[T]``.  This base class
extracts the shared CRUD and paginated-query logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.orm import Session

    from poc_homography.infrastructure.database import Base

logger = logging.getLogger(__name__)


class RepoPostgresSession(ABC):
    """Abstract base for PostgreSQL-backed session repositories."""

    def __init__(
        self,
        session: Session,
        model_cls: type[Base],
        entity_factory: Callable[[dict[str, Any]], Any],
    ) -> None:
        self._session = session
        self._model_cls = model_cls
        self._entity_factory = entity_factory

    # ------------------------------------------------------------------
    # Extraction hooks (override per session type)
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def _extract_tenant_id(data: dict[str, Any]) -> str: ...

    @staticmethod
    @abstractmethod
    def _extract_camera_id(data: dict[str, Any]) -> str: ...

    @staticmethod
    @abstractmethod
    def _extract_created_date(data: dict[str, Any]) -> datetime: ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_iso_or_now(raw: object) -> datetime:
        """Parse an ISO datetime string, falling back to UTC now."""
        if raw:
            return datetime.fromisoformat(str(raw))
        return datetime.now(tz=timezone.utc)

    def _row_to_entity(self, row: Base) -> Any:
        return self._entity_factory(row.data)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> Any | None:
        """Load a session by ID, or ``None`` if not found."""
        row = self._session.get(self._model_cls, session_id)
        if row is None:
            return None
        return self._row_to_entity(row)

    def save(self, entity: Any) -> bool:
        """Persist a session entity.  Returns ``True`` on success."""
        try:
            data: dict[str, Any] = entity.to_dict()
            session_id = entity.id
            tenant_id = self._extract_tenant_id(data)
            camera_id = self._extract_camera_id(data)
            created_date = self._extract_created_date(data)

            row = self._session.get(self._model_cls, session_id)
            if row is None:
                row = self._model_cls(
                    id=session_id,
                    tenant_id=tenant_id,
                    camera_id=camera_id,
                    created_date=created_date,
                    data=data,
                )
                self._session.add(row)
            else:
                row.tenant_id = tenant_id  # type: ignore[attr-defined]
                row.camera_id = camera_id  # type: ignore[attr-defined]
                row.created_date = created_date  # type: ignore[attr-defined]
                row.data = data  # type: ignore[attr-defined]
            self._session.flush()
            return True
        except Exception:
            logger.exception(
                "Failed to save %s %s",
                self._model_cls.__name__,  # type: ignore[attr-defined]
                entity.id,
            )
            return False

    def delete(self, session_id: str) -> tuple[bool, str | None]:
        """Delete a session by ID.  Returns ``(success, error_message)``."""
        row = self._session.get(self._model_cls, session_id)
        if row is None:
            return False, "Session not found"
        try:
            self._session.delete(row)
            self._session.flush()
            return True, None
        except Exception as exc:
            logger.exception(
                "Failed to delete %s %s",
                self._model_cls.__name__,  # type: ignore[attr-defined]
                session_id,
            )
            return False, str(exc)

    def exists(self, session_id: str) -> bool:
        return self._session.get(self._model_cls, session_id) is not None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_all(
        self,
        *,
        tenant_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Any], int]:
        """List sessions with optional filters.

        Returns ``(sessions_page, total_count)``.
        """
        conditions: list[Any] = []
        if tenant_id:
            conditions.append(self._model_cls.tenant_id == tenant_id)  # type: ignore[attr-defined]
        if camera_id:
            conditions.append(self._model_cls.camera_id == camera_id)  # type: ignore[attr-defined]

        # Total count
        count_stmt = select(func.count()).select_from(self._model_cls)
        for cond in conditions:
            count_stmt = count_stmt.where(cond)
        total: int = self._session.execute(count_stmt).scalar_one()

        # Paginated rows, newest first
        stmt = (
            select(self._model_cls)
            .order_by(self._model_cls.created_date.desc())  # type: ignore[attr-defined]
            .limit(limit)
            .offset(offset)
        )
        for cond in conditions:
            stmt = stmt.where(cond)

        rows = self._session.execute(stmt).scalars().all()
        return [self._row_to_entity(row) for row in rows], total
