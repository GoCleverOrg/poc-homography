"""PostgreSQL-backed DiagnosticSession repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poc_homography.infrastructure.models.diagnostic_session import (
    DiagnosticSessionModel,
)
from poc_homography.infrastructure.repositories.base import RepoPostgresSession

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from sqlalchemy.orm import Session


class RepoPostgresDiagnosticSession(RepoPostgresSession):
    """Repository for DiagnosticSession manifests stored in PostgreSQL."""

    def __init__(
        self,
        session: Session,
        entity_factory: Callable[[dict[str, Any]], Any],
    ) -> None:
        super().__init__(session, DiagnosticSessionModel, entity_factory)

    @staticmethod
    def _extract_tenant_id(data: dict[str, Any]) -> str:
        return str(data.get("tenant", {}).get("id", ""))

    @staticmethod
    def _extract_camera_id(_data: dict[str, Any]) -> str:
        return ""

    @staticmethod
    def _extract_created_date(data: dict[str, Any]) -> datetime:
        raw = data.get("session", {}).get("created_at")
        return RepoPostgresSession._parse_iso_or_now(raw)
