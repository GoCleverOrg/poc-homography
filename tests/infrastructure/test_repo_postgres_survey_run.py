"""Integration tests for RepoPostgresSurveyRun (skipped without DATABASE_URL).

These exercise the JSONB round-trip and the indexed ``camera_id`` / ``status``
filtering against a real PostgreSQL database. The ``db_session`` fixture rolls
back after each test, so no permanent data is written.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus
from poc_homography.infrastructure.repositories.repo_postgres_survey_run import (
    RepoPostgresSurveyRun,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

_TS = datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc)


def _make_run(run_id: str, *, status: SurveyRunStatus = SurveyRunStatus.RUNNING) -> SurveyRun:
    return SurveyRun(
        run_id=run_id,
        camera_id="cam01",
        phases=frozenset({SurveyPhase.MAIN_SURVEY, SurveyPhase.VALIDATION}),
        started_at=_TS,
        finished_at=None,
        status=status,
    )


@pytest.mark.integration
class TestRepoPostgresSurveyRun:
    def test_save_and_get_round_trip(self, db_session: Session) -> None:
        repo = RepoPostgresSurveyRun(db_session, SurveyRun.from_dict)
        run = _make_run("pg-run-0001")
        assert repo.save(run) is True

        loaded = repo.get("pg-run-0001")
        assert loaded is not None
        assert loaded == run
        assert loaded.to_dict() == run.to_dict()

    def test_get_nonexistent(self, db_session: Session) -> None:
        repo = RepoPostgresSurveyRun(db_session, SurveyRun.from_dict)
        assert repo.get("missing") is None

    def test_upsert_updates_status(self, db_session: Session) -> None:
        repo = RepoPostgresSurveyRun(db_session, SurveyRun.from_dict)
        repo.save(_make_run("pg-run-0002", status=SurveyRunStatus.RUNNING))
        repo.save(_make_run("pg-run-0002", status=SurveyRunStatus.COMPLETED))

        loaded = repo.get("pg-run-0002")
        assert loaded is not None
        assert loaded.status is SurveyRunStatus.COMPLETED

    def test_delete(self, db_session: Session) -> None:
        repo = RepoPostgresSurveyRun(db_session, SurveyRun.from_dict)
        repo.save(_make_run("pg-run-0003"))
        success, error = repo.delete("pg-run-0003")
        assert success is True
        assert error is None
        assert repo.get("pg-run-0003") is None

    def test_get_all_filters_by_camera(self, db_session: Session) -> None:
        repo = RepoPostgresSurveyRun(db_session, SurveyRun.from_dict)
        repo.save(_make_run("pg-run-0004"))
        runs, total = repo.get_all(camera_id="cam01")
        assert total >= 1
        assert all(r.camera_id == "cam01" for r in runs)
