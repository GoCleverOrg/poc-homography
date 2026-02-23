"""Tests for session repo factory pattern (Arch #6).

All three session repos (Diagnostic, StressTest, Survey) accept an entity_factory
callable. These tests verify the factory is invoked on get() and that a raising
factory is handled gracefully.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import yaml

from poc_homography.infrastructure.repositories.repo_yaml_diagnostic_session import (
    RepoYamlDiagnosticSession,
)
from poc_homography.infrastructure.repositories.repo_yaml_stress_test_session import (
    RepoYamlStressTestSession,
)
from poc_homography.infrastructure.repositories.repo_yaml_survey_session import (
    RepoYamlSurveySession,
)

SESSION_ID = str(uuid.uuid4())
CREATED_AT = datetime(2024, 6, 15, 12, 0, 0)
DATE_STR = CREATED_AT.strftime("%Y%m%d")


def _write_manifest(base_dir: Path, manifest_data: dict[str, Any]) -> None:
    """Write a manifest.yaml in the date-partitioned layout."""
    session_dir = base_dir / DATE_STR / SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)
    with open(session_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f)


MANIFEST = {"id": SESSION_ID, "created_at": CREATED_AT.isoformat(), "data": "test"}


# ── DiagnosticSession ─────────────────────────────────────────────


class TestDiagnosticSessionFactory:
    def test_factory_called_on_get(self, tmp_path: Path) -> None:
        calls: list[dict[str, Any]] = []

        def factory(data: dict[str, Any]) -> dict[str, Any]:
            calls.append(data)
            return data

        _write_manifest(tmp_path, MANIFEST)
        repo = RepoYamlDiagnosticSession(tmp_path, entity_factory=factory)
        result = repo.get(SESSION_ID)

        assert len(calls) == 1
        assert calls[0]["id"] == SESSION_ID
        assert result is not None

    def test_raising_factory_returns_none(self, tmp_path: Path) -> None:
        def bad_factory(data: dict[str, Any]) -> None:
            raise ValueError("bad data")

        _write_manifest(tmp_path, MANIFEST)
        repo = RepoYamlDiagnosticSession(tmp_path, entity_factory=bad_factory)
        assert repo.get(SESSION_ID) is None

    def test_invalid_session_id_returns_none(self, tmp_path: Path) -> None:
        repo = RepoYamlDiagnosticSession(tmp_path, entity_factory=lambda d: d)
        assert repo.get("../../../etc/passwd") is None


# ── StressTestSession ─────────────────────────────────────────────


class TestStressTestSessionFactory:
    def test_factory_called_on_get(self, tmp_path: Path) -> None:
        calls: list[dict[str, Any]] = []

        def factory(data: dict[str, Any]) -> dict[str, Any]:
            calls.append(data)
            return data

        _write_manifest(tmp_path, MANIFEST)
        repo = RepoYamlStressTestSession(tmp_path, entity_factory=factory)
        result = repo.get(SESSION_ID)

        assert len(calls) == 1
        assert calls[0]["id"] == SESSION_ID
        assert result is not None

    def test_raising_factory_returns_none(self, tmp_path: Path) -> None:
        def bad_factory(data: dict[str, Any]) -> None:
            raise ValueError("bad data")

        _write_manifest(tmp_path, MANIFEST)
        repo = RepoYamlStressTestSession(tmp_path, entity_factory=bad_factory)
        assert repo.get(SESSION_ID) is None

    def test_invalid_session_id_returns_none(self, tmp_path: Path) -> None:
        repo = RepoYamlStressTestSession(tmp_path, entity_factory=lambda d: d)
        assert repo.get("not-a-uuid") is None


# ── SurveySession ────────────────────────────────────────────────


class TestSurveySessionFactory:
    def test_factory_called_on_get(self, tmp_path: Path) -> None:
        calls: list[dict[str, Any]] = []

        def factory(data: dict[str, Any]) -> dict[str, Any]:
            calls.append(data)
            return data

        _write_manifest(tmp_path, MANIFEST)
        repo = RepoYamlSurveySession(tmp_path, entity_factory=factory)
        result = repo.get(SESSION_ID)

        assert len(calls) == 1
        assert calls[0]["id"] == SESSION_ID
        assert result is not None

    def test_raising_factory_returns_none(self, tmp_path: Path) -> None:
        def bad_factory(data: dict[str, Any]) -> None:
            raise ValueError("bad data")

        _write_manifest(tmp_path, MANIFEST)
        repo = RepoYamlSurveySession(tmp_path, entity_factory=bad_factory)
        assert repo.get(SESSION_ID) is None

    def test_invalid_session_id_returns_none(self, tmp_path: Path) -> None:
        repo = RepoYamlSurveySession(tmp_path, entity_factory=lambda d: d)
        assert repo.get("../../etc/shadow") is None
