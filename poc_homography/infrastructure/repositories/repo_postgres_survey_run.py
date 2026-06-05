"""PostgreSQL-backed :class:`SurveyRun` repository."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from poc_homography.infrastructure.models.survey_run import SurveyRunModel
from poc_homography.infrastructure.repositories.base import RepoPostgresSession

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class RepoPostgresSurveyRun(RepoPostgresSession):
    """Repository for :class:`SurveyRun` aggregates stored in PostgreSQL.

    ``SurveyRun`` has no tenant, so :meth:`save` is overridden to persist the
    ``camera_id``, ``status``, and ``created_date`` indexed columns plus the
    full JSONB ``data`` blob. The inherited ``get`` / ``delete`` / ``exists``
    operate unchanged. :meth:`get_all` is overridden to drop the tenant filter,
    since ``survey_runs`` has no ``tenant_id`` column (the inherited
    implementation would raise ``AttributeError`` for a truthy ``tenant_id``).
    """

    def __init__(
        self,
        session: Session,
        entity_factory: Callable[[dict[str, Any]], Any],
    ) -> None:
        super().__init__(session, SurveyRunModel, entity_factory)

    @staticmethod
    def _extract_tenant_id(data: dict[str, Any]) -> str:  # noqa: ARG004 — abstract hook; SurveyRun is tenant-less
        """SurveyRun has no tenant; the ``survey_runs`` table has no column."""
        return ""

    @staticmethod
    def _extract_camera_id(data: dict[str, Any]) -> str:
        return str(data.get("camera_id", ""))

    @staticmethod
    def _extract_created_date(data: dict[str, Any]) -> datetime:
        return RepoPostgresSession._parse_iso_or_now(data.get("started_at"))

    @staticmethod
    def _extract_status(data: dict[str, Any]) -> str:
        return str(data.get("status", ""))

    def get_all(
        self,
        *,
        tenant_id: str | None = None,  # tenant-less; accepted for LSP, ignored
        camera_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Any], int]:
        """List survey runs, newest first. ``tenant_id`` is ignored.

        ``survey_runs`` has no ``tenant_id`` column, so the base filter is
        forced off to avoid an ``AttributeError`` on ``SurveyRunModel``.
        """
        return super().get_all(tenant_id=None, camera_id=camera_id, limit=limit, offset=offset)

    def save(self, entity: Any) -> bool:
        """Persist a :class:`SurveyRun`, upserting by id. Returns success.

        Overrides the base because ``survey_runs`` has a ``status`` column and
        no ``tenant_id`` column.
        """
        try:
            data: dict[str, Any] = entity.to_dict()
            row = self._session.get(self._model_cls, entity.id)
            if row is None:
                row = self._model_cls(
                    id=entity.id,
                    camera_id=self._extract_camera_id(data),
                    status=self._extract_status(data),
                    created_date=self._extract_created_date(data),
                    data=data,
                )
                self._session.add(row)
            else:
                row.camera_id = self._extract_camera_id(data)  # type: ignore[attr-defined]
                row.status = self._extract_status(data)  # type: ignore[attr-defined]
                row.created_date = self._extract_created_date(data)  # type: ignore[attr-defined]
                row.data = data  # type: ignore[attr-defined]
            self._session.flush()
            return True
        except Exception:
            logger.exception("Failed to save SurveyRun %s", entity.id)
            return False
