"""Tests for plan_config persistence on the SurveyRun repos.

The YAML test is a pure-filesystem unit test (no DB). The Postgres test is an
``integration`` test gated on ``DATABASE_URL`` by the infra conftest; the
``db_session`` fixture rolls back after each test.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus
from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig
from poc_homography.infrastructure.repositories.repo_postgres_survey_run import (
    RepoPostgresSurveyRun,
)
from poc_homography.infrastructure.repositories.repo_yaml_survey_run import (
    RepoYamlSurveyRun,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.orm import Session

_TS = datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc)


def _make_config() -> SurveyPlanConfig:
    return SurveyPlanConfig(
        enabled_phases=frozenset({1, 4, 5, 9}),
        phase_pan_range={4: (-30.0, 30.0), 5: (-90.0, 90.0)},
        phase_zoom_range={6: (1.0, 12.0)},
        grid_overlap_pct={4: 70.0, 5: 60.0},
        burst_frame_count={2: 5, 7: 4},
        jitter_burst_duration_s=15.0,
        jitter_pose_count=3,
        zoom_levels=[1.0, 2.0, 4.0],
        repeat_count={2: 2, 7: 5},
        holdout_fraction=0.25,
    )


def _make_run(run_id: str) -> SurveyRun:
    return SurveyRun(
        run_id=run_id,
        camera_id="cam01",
        phases=frozenset({SurveyPhase.MAIN_SURVEY, SurveyPhase.VALIDATION}),
        started_at=_TS,
        finished_at=None,
        status=SurveyRunStatus.RUNNING,
    )


def test_yaml_plan_config_round_trip(tmp_path: Path) -> None:
    repo = RepoYamlSurveyRun(
        data_dir=tmp_path / "survey_runs",
        frames_dir=tmp_path / "survey",
    )
    cfg = _make_config()
    run_id = "run-0001"

    assert repo.save_plan_config(run_id, cfg) is True

    sidecar = tmp_path / "survey" / run_id / "plan_config.yaml"
    assert sidecar.exists()

    assert repo.load_plan_config(run_id) == cfg


def test_yaml_load_plan_config_unknown_raises_key_error(tmp_path: Path) -> None:
    repo = RepoYamlSurveyRun(
        data_dir=tmp_path / "survey_runs",
        frames_dir=tmp_path / "survey",
    )
    with pytest.raises(KeyError):
        repo.load_plan_config("does-not-exist")


@pytest.mark.integration
class TestRepoPostgresSurveyRunPlanConfig:
    def test_plan_config_round_trip(self, db_session: Session) -> None:
        repo = RepoPostgresSurveyRun(db_session, SurveyRun.from_dict)
        run = _make_run("pg-plan-0001")
        assert repo.save(run) is True

        cfg = _make_config()
        assert repo.save_plan_config(run.run_id, cfg) is True
        assert repo.load_plan_config(run.run_id) == cfg

    def test_load_plan_config_unknown_raises_key_error(self, db_session: Session) -> None:
        repo = RepoPostgresSurveyRun(db_session, SurveyRun.from_dict)
        with pytest.raises(KeyError):
            repo.load_plan_config("missing-run")
