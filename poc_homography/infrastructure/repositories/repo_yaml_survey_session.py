"""YAML-based SurveySession repository with date-partitioned storage.

Storage layout: ``data_dir/YYYYMMDD/<session_id>/manifest.yaml``
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger(__name__)


def _is_valid_session_id(session_id: str) -> bool:
    try:
        uuid.UUID(session_id)
        return True
    except (ValueError, TypeError):
        return False


class RepoYamlSurveySession:
    """Repository for SurveySession entities stored as date-partitioned YAML manifests."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _session_dir(self, session_id: str, date: datetime) -> Path:
        date_str = date.strftime("%Y%m%d")
        return self._data_dir / date_str / session_id

    def _find_session_dir(self, session_id: str) -> Path | None:
        if not _is_valid_session_id(session_id):
            return None
        if not self._data_dir.exists():
            return None
        for date_dir in self._data_dir.iterdir():
            if date_dir.is_dir():
                candidate = date_dir / session_id
                if candidate.is_dir() and (candidate / "manifest.yaml").exists():
                    return candidate
        return None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> Any | None:
        session_dir = self._find_session_dir(session_id)
        if session_dir is None:
            return None
        return self._load_entity(session_dir)

    def save(self, entity: Any) -> bool:
        """Persist a SurveySession.  Returns True on success."""
        try:
            session_dir = self._session_dir(entity.id, entity.start_time)
            session_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = session_dir / "manifest.yaml"
            with open(manifest_path, "w", encoding="utf-8") as f:
                yaml.dump(entity.to_dict(), f, default_flow_style=False, sort_keys=False)
            return True
        except Exception:
            logger.exception("Failed to save survey session %s", entity.id)
            return False

    def delete(self, session_id: str) -> tuple[bool, str | None]:
        if not _is_valid_session_id(session_id):
            return False, "Invalid session ID format"

        session_dir = self._find_session_dir(session_id)
        if session_dir is None:
            return False, "Session not found"

        try:
            date_dir = session_dir.parent
            shutil.rmtree(session_dir)
            if date_dir.exists() and not any(date_dir.iterdir()):
                date_dir.rmdir()
            return True, None
        except Exception as exc:
            logger.exception("Failed to delete survey session %s", session_id)
            return False, str(exc)

    def exists(self, session_id: str) -> bool:
        return self._find_session_dir(session_id) is not None

    def get_session_dir(self, session_id: str) -> Path | None:
        """Return the session directory path (for image storage)."""
        return self._find_session_dir(session_id)

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
        sessions: list[Any] = []

        if not self._data_dir.exists():
            return [], 0

        for date_dir in sorted(self._data_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            for session_dir in sorted(date_dir.iterdir(), reverse=True):
                if not session_dir.is_dir():
                    continue
                entity = self._load_entity(session_dir)
                if entity is None:
                    continue
                if tenant_id and getattr(entity.tenant, "id", None) != tenant_id:
                    continue
                if camera_id and getattr(entity.camera, "id", None) != camera_id:
                    continue
                sessions.append(entity)

        sessions.sort(key=lambda s: s.start_time, reverse=True)
        total = len(sessions)
        return sessions[offset : offset + limit], total

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_entity(session_dir: Path) -> Any | None:
        try:
            from camera_survey.models import SurveySession  # type: ignore[import-untyped]

            manifest_path = session_dir / "manifest.yaml"
            if not manifest_path.exists():
                return None
            with open(manifest_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return SurveySession.from_dict(data)
        except Exception:
            logger.warning("Failed to load survey session from %s", session_dir, exc_info=True)
            return None
