"""Tests for RepoYamlSurveyRun persistence and frame grouping queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from tests.domain.survey.builders import make_frame_record, make_survey_run

from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.infrastructure.repositories.repo_yaml_survey_run import (
    RepoYamlSurveyRun,
)
from poc_homography.types import Unitless

if TYPE_CHECKING:
    from pathlib import Path

    from poc_homography.domain.entities.survey.frame_record import FrameRecord


def _write_frame(frames_dir: Path, frame: FrameRecord) -> None:
    """Persist a FrameRecord under data/survey/{run}/{camera}/frames/."""
    target = (
        frames_dir
        / frame.capture.run_id
        / frame.camera.camera_id
        / "frames"
        / f"{frame.capture.capture_id}.yaml"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.dump(frame.to_dict(), handle, default_flow_style=False, sort_keys=False)


def _make_repo(tmp_path: Path) -> RepoYamlSurveyRun:
    return RepoYamlSurveyRun(tmp_path / "survey_runs", tmp_path / "survey")


class TestRepoYamlSurveyRunPersistence:
    def test_save_and_get(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        run = make_survey_run(run_id="run-0001")
        repo.save(run)
        assert (tmp_path / "survey_runs" / "run-0001.yaml").exists()

    def test_round_trip_via_fresh_repo(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        run = make_survey_run(run_id="run-0001")
        repo.save(run)
        fresh = _make_repo(tmp_path)
        loaded = fresh.get("run-0001")
        assert loaded is not None
        assert loaded == run
        assert loaded.to_dict() == run.to_dict()

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert repo.get("does-not-exist") is None


class TestRepoYamlSurveyRunGroupingQueries:
    def _seed(self, tmp_path: Path) -> RepoYamlSurveyRun:
        frames_dir = tmp_path / "survey"
        _write_frame(
            frames_dir,
            make_frame_record(
                capture_id="c1",
                run_id="runA",
                camera_id="cam01",
                phase=SurveyPhase.MAIN_SURVEY,
                reported_zoom=2.0,
            ),
        )
        _write_frame(
            frames_dir,
            make_frame_record(
                capture_id="c2",
                run_id="runA",
                camera_id="cam02",
                phase=SurveyPhase.DENSE_NADIR,
                reported_zoom=8.0,
                burst_id="burstX",
                frame_index=1,
            ),
        )
        _write_frame(
            frames_dir,
            make_frame_record(
                capture_id="c3",
                run_id="runB",
                camera_id="cam01",
                phase=SurveyPhase.MAIN_SURVEY,
                reported_zoom=12.0,
            ),
        )
        return _make_repo(tmp_path)

    def test_by_run(self, tmp_path: Path) -> None:
        repo = self._seed(tmp_path)
        assert {f.id for f in repo.get_frames_by_run("runA")} == {"c1", "c2"}
        assert {f.id for f in repo.get_frames_by_run("runB")} == {"c3"}

    def test_by_camera(self, tmp_path: Path) -> None:
        repo = self._seed(tmp_path)
        assert {f.id for f in repo.get_frames_by_camera("cam01")} == {"c1", "c3"}

    def test_by_phase(self, tmp_path: Path) -> None:
        repo = self._seed(tmp_path)
        assert {f.id for f in repo.get_frames_by_phase(SurveyPhase.MAIN_SURVEY)} == {"c1", "c3"}
        assert {f.id for f in repo.get_frames_by_phase(SurveyPhase.DENSE_NADIR)} == {"c2"}

    def test_by_zoom_range(self, tmp_path: Path) -> None:
        repo = self._seed(tmp_path)
        result = repo.get_frames_by_zoom_range(Unitless(5.0), Unitless(10.0))
        assert {f.id for f in result} == {"c2"}

    def test_by_burst(self, tmp_path: Path) -> None:
        repo = self._seed(tmp_path)
        assert {f.id for f in repo.get_frames_by_burst("burstX")} == {"c2"}

    def test_empty_frames_dir_returns_empty(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert repo.get_frames_by_run("any") == []
