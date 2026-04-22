"""PostgreSQL-backed SurveySession repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poc_homography.infrastructure.models.survey_session import (
    SurveySessionModel,
)
from poc_homography.infrastructure.repositories.base import RepoPostgresSession

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from sqlalchemy.orm import Session


class RepoPostgresSurveySession(RepoPostgresSession):
    """Repository for SurveySession manifests stored in PostgreSQL."""

    def __init__(
        self,
        session: Session,
        entity_factory: Callable[[dict[str, Any]], Any],
    ) -> None:
        super().__init__(session, SurveySessionModel, entity_factory)

    @staticmethod
    def _extract_tenant_id(data: dict[str, Any]) -> str:
        return str(data.get("tenant", {}).get("id", ""))

    @staticmethod
    def _extract_camera_id(data: dict[str, Any]) -> str:
        return str(data.get("camera", {}).get("id", ""))

    @staticmethod
    def _extract_created_date(data: dict[str, Any]) -> datetime:
        raw = data.get("session", {}).get("start_time")
        return RepoPostgresSession._parse_iso_or_now(raw)

    def get_session_dir(self, session_id: str) -> None:
        """Not applicable for PostgreSQL storage.

        The YAML repo returns a filesystem path for image storage.
        The Postgres repo does not manage session directories, so this
        always returns ``None``.
        """
        return None
