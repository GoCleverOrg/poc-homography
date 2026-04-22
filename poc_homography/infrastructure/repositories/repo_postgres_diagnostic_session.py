"""PostgreSQL-backed DiagnosticSession repository.

This repository does NOT extend ``RepoPostgres[T]`` because session entities
live in the webapp layer (untyped ``Any``) rather than as domain ``Entity``
subclasses.  It mirrors the public interface of
``RepoYamlDiagnosticSession`` while storing session manifests as JSONB with
indexed columns for tenant_id, camera_id, and created_date.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from poc_homography.infrastructure.models.diagnostic_session import (
    DiagnosticSessionModel,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class RepoPostgresDiagnosticSession:
    """Repository for DiagnosticSession manifests stored in PostgreSQL."""

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

        DiagnosticSession.to_dict() nests tenant info as
        ``{"tenant": {"id": "..."}, ...}``.
        """
        return str(data.get("tenant", {}).get("id", ""))

    @staticmethod
    def _extract_camera_id(_data: dict[str, Any]) -> str:
        """Extract camera_id (empty string for diagnostic sessions)."""
        return ""

    @staticmethod
    def _extract_created_date(data: dict[str, Any]) -> datetime:
        """Extract created_date from the manifest dict.

        DiagnosticSession.to_dict() nests it as
        ``{"session": {"created_at": "<iso>"}}``.
        """
        raw = data.get("session", {}).get("created_at")
        if raw:
            return datetime.fromisoformat(str(raw))
        return datetime.now(tz=timezone.utc)

    def _row_to_entity(self, row: DiagnosticSessionModel) -> Any:
        return self._entity_factory(row.data)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> Any | None:
        """Load a DiagnosticSession by ID.

        Returns the entity or ``None`` if not found.
        """
        row = self._session.get(DiagnosticSessionModel, session_id)
        if row is None:
            return None
        return self._row_to_entity(row)

    def save(self, entity: Any) -> bool:
        """Persist a DiagnosticSession entity.

        Returns ``True`` on success, ``False`` otherwise.
        """
        try:
            data: dict[str, Any] = entity.to_dict()
            session_id = entity.id
            tenant_id = self._extract_tenant_id(data)
            camera_id = self._extract_camera_id(data)
            created_date = self._extract_created_date(data)

            row = self._session.get(DiagnosticSessionModel, session_id)
            if row is None:
                row = DiagnosticSessionModel(
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
            return True
        except Exception:
            logger.exception("Failed to save diagnostic session %s", entity.id)
            return False

    def delete(self, session_id: str) -> tuple[bool, str | None]:
        """Delete a session by ID.

        Returns ``(success, error_message)``.
        """
        row = self._session.get(DiagnosticSessionModel, session_id)
        if row is None:
            return False, "Session not found"
        try:
            self._session.delete(row)
            self._session.flush()
            return True, None
        except Exception as exc:
            logger.exception("Failed to delete diagnostic session %s", session_id)
            return False, str(exc)

    def exists(self, session_id: str) -> bool:
        return self._session.get(DiagnosticSessionModel, session_id) is not None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_all(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Any], int]:
        """List sessions with optional tenant filter.

        Returns ``(sessions_page, total_count)``.
        """
        # -- build WHERE clause ------------------------------------------------
        conditions: list[Any] = []
        if tenant_id:
            conditions.append(DiagnosticSessionModel.tenant_id == tenant_id)

        # -- total count -------------------------------------------------------
        count_stmt = select(func.count()).select_from(DiagnosticSessionModel)
        for cond in conditions:
            count_stmt = count_stmt.where(cond)
        total: int = self._session.execute(count_stmt).scalar_one()

        # -- paginated rows, newest first --------------------------------------
        stmt = (
            select(DiagnosticSessionModel)
            .order_by(DiagnosticSessionModel.created_date.desc())
            .limit(limit)
            .offset(offset)
        )
        for cond in conditions:
            stmt = stmt.where(cond)

        rows = self._session.execute(stmt).scalars().all()
        sessions = [self._row_to_entity(row) for row in rows]
        return sessions, total
