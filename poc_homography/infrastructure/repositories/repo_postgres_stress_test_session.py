"""PostgreSQL-backed StressTestSession repository.

This repository does NOT extend ``RepoPostgres[T]`` because session entities
live in the webapp layer (untyped ``Any``) rather than as domain ``Entity``
subclasses.  It mirrors the public interface of
``RepoYamlStressTestSession`` while storing session manifests as JSONB with
indexed columns for tenant_id, camera_id, and created_date.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from poc_homography.infrastructure.models.stress_test_session import (
    StressTestSessionModel,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class RepoPostgresStressTestSession:
    """Repository for StressTestSession manifests stored in PostgreSQL."""

    def __init__(
        self,
        session: Session,
        entity_factory: Callable[[dict[str, Any]], Any],
    ) -> None:
        self._session = session
        self._entity_factory = entity_factory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tenant_id(data: dict[str, Any]) -> str:
        """Extract tenant_id from the manifest dict.

        StressTestSession.to_dict() stores it at the top level as
        ``{"tenant_id": "..."}``.
        """
        return str(data.get("tenant_id", ""))

    @staticmethod
    def _extract_camera_id(data: dict[str, Any]) -> str:
        """Extract camera_id from the manifest dict.

        StressTestSession.to_dict() stores it at the top level as
        ``{"camera_id": "..."}``.
        """
        return str(data.get("camera_id", ""))

    @staticmethod
    def _extract_created_date(data: dict[str, Any]) -> datetime:
        """Extract created_date from the manifest dict.

        StressTestSession.to_dict() stores it at the top level as
        ``{"created_at": "<iso>"}``.
        """
        raw = data.get("created_at")
        if raw:
            return datetime.fromisoformat(str(raw))
        return datetime.now()

    def _row_to_entity(self, row: StressTestSessionModel) -> Any:
        return self._entity_factory(row.data)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> Any | None:
        """Load a StressTestSession by ID.

        Returns the entity or ``None`` if not found.
        """
        row = self._session.get(StressTestSessionModel, session_id)
        if row is None:
            return None
        return self._row_to_entity(row)

    def save(self, entity: Any) -> None:
        """Persist a StressTestSession entity."""
        data: dict[str, Any] = entity.to_dict()
        session_id = entity.id
        tenant_id = self._extract_tenant_id(data)
        camera_id = self._extract_camera_id(data)
        created_date = self._extract_created_date(data)

        row = self._session.get(StressTestSessionModel, session_id)
        if row is None:
            row = StressTestSessionModel(
                id=session_id,
                tenant_id=tenant_id,
                camera_id=camera_id,
                created_date=created_date,
                data=data,
            )
            self._session.add(row)
        else:
            row.tenant_id = tenant_id
            row.camera_id = camera_id
            row.created_date = created_date
            row.data = data
        self._session.flush()

    def delete(self, session_id: str) -> tuple[bool, str | None]:
        """Delete a session by ID.

        Returns ``(success, error_message)``.
        """
        row = self._session.get(StressTestSessionModel, session_id)
        if row is None:
            return False, "Session not found"
        try:
            self._session.delete(row)
            self._session.flush()
            return True, None
        except Exception as exc:
            logger.exception("Failed to delete stress test session %s", session_id)
            return False, str(exc)

    def exists(self, session_id: str) -> bool:
        return self._session.get(StressTestSessionModel, session_id) is not None

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
        """List sessions with optional tenant/camera filter.

        Returns ``(sessions_page, total_count)``.
        """
        # -- build WHERE clause ------------------------------------------------
        conditions: list[Any] = []
        if tenant_id:
            conditions.append(StressTestSessionModel.tenant_id == tenant_id)
        if camera_id:
            conditions.append(StressTestSessionModel.camera_id == camera_id)

        # -- total count -------------------------------------------------------
        count_stmt = select(func.count()).select_from(StressTestSessionModel)
        for cond in conditions:
            count_stmt = count_stmt.where(cond)
        total: int = self._session.execute(count_stmt).scalar_one()

        # -- paginated rows, newest first --------------------------------------
        stmt = (
            select(StressTestSessionModel)
            .order_by(StressTestSessionModel.created_date.desc())
            .limit(limit)
            .offset(offset)
        )
        for cond in conditions:
            stmt = stmt.where(cond)

        rows = self._session.execute(stmt).scalars().all()
        sessions = [self._row_to_entity(row) for row in rows]
        return sessions, total
