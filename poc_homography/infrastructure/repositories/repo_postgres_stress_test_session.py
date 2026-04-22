"""PostgreSQL-backed StressTestSession repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poc_homography.infrastructure.models.stress_test_session import (
    StressTestSessionModel,
)
from poc_homography.infrastructure.repositories.base import RepoPostgresSession

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from sqlalchemy.orm import Session


class RepoPostgresStressTestSession(RepoPostgresSession):
    """Repository for StressTestSession manifests stored in PostgreSQL."""

    def __init__(
        self,
        session: Session,
        entity_factory: Callable[[dict[str, Any]], Any],
    ) -> None:
        super().__init__(session, StressTestSessionModel, entity_factory)

    @staticmethod
    def _extract_tenant_id(data: dict[str, Any]) -> str:
        return str(data.get("tenant_id", ""))

    @staticmethod
    def _extract_camera_id(data: dict[str, Any]) -> str:
        return str(data.get("camera_id", ""))

    @staticmethod
    def _extract_created_date(data: dict[str, Any]) -> datetime:
        raw = data.get("created_at")
        return RepoPostgresSession._parse_iso_or_now(raw)
