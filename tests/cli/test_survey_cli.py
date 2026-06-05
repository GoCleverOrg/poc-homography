"""Tests for the ``hom survey`` CLI sub-app."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import yaml
from typer.testing import CliRunner

import poc_homography.cli.survey as survey_cli
from poc_homography.cli.main import app
from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig

runner = CliRunner()


@pytest.mark.integration
def test_survey_run_streams_progress(tmp_path):
    """`survey run` against the real default service prints sessions + progress."""
    plan = SurveyPlanConfig(enabled_phases=frozenset({1, 2}))
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan.to_dict()))

    result = runner.invoke(
        app,
        ["survey", "run", "--plan", str(plan_path), "--cameras", "cam-a,cam-b"],
    )

    assert result.exit_code == 0, result.output
    assert "run_id:" in result.output
    assert "cam-a" in result.output
    assert "cam-b" in result.output
    assert "phase=" in result.output


def test_survey_run_bad_plan(tmp_path, monkeypatch):
    """`survey run` exits 1 on an unsupported schema_version."""
    plan_path = tmp_path / "bad.yaml"
    plan_path.write_text(yaml.safe_dump({"schema_version": "999"}))

    result = runner.invoke(
        app,
        ["survey", "run", "--plan", str(plan_path), "--cameras", "cam-a"],
    )

    assert result.exit_code == 1
    assert "Error" in result.output


def test_survey_status(monkeypatch):
    """`survey status` prints per-camera phase/frame/status."""
    fake = MagicMock()
    fake.get_status.return_value = {
        "run_id": "run-1",
        "cameras": {
            "cam-a": {
                "session_id": "sess-a",
                "phase": "camera_inventory",
                "frame_count": 4,
                "status": "running",
            }
        },
    }
    monkeypatch.setattr(survey_cli, "_survey_run_service", fake)

    result = runner.invoke(app, ["survey", "status", "--run-id", "run-1"])

    assert result.exit_code == 0, result.output
    assert "run-1" in result.output
    assert "cam-a" in result.output
    assert "phase=camera_inventory" in result.output
    assert "frames=4" in result.output
    assert "status=running" in result.output


def test_survey_status_not_found(monkeypatch):
    """`survey status` exits 1 when the run is unknown."""
    fake = MagicMock()
    fake.get_status.return_value = None
    monkeypatch.setattr(survey_cli, "_survey_run_service", fake)

    result = runner.invoke(app, ["survey", "status", "--run-id", "nope"])

    assert result.exit_code == 1
    assert "Error" in result.output


def test_survey_abort(monkeypatch):
    """`survey abort` prints the returned message."""
    fake = MagicMock()
    fake.abort_run.return_value = {"run_id": "run-1", "message": "Run abort requested"}
    monkeypatch.setattr(survey_cli, "_survey_run_service", fake)

    result = runner.invoke(app, ["survey", "abort", "--run-id", "run-1"])

    assert result.exit_code == 0, result.output
    assert "Run abort requested" in result.output


def test_survey_abort_not_found(monkeypatch):
    """`survey abort` exits 1 when the run is unknown."""
    fake = MagicMock()
    fake.abort_run.return_value = None
    monkeypatch.setattr(survey_cli, "_survey_run_service", fake)

    result = runner.invoke(app, ["survey", "abort", "--run-id", "nope"])

    assert result.exit_code == 1
    assert "Error" in result.output


def test_survey_list(monkeypatch):
    """`survey list` prints a header and a row per run."""
    fake = MagicMock()
    fake.list_runs.return_value = [
        {
            "run_id": "run-1",
            "start_time": "2026-06-05T00:00:00+00:00",
            "camera_count": 2,
            "total_frame_count": 8,
            "status": "completed",
        }
    ]
    monkeypatch.setattr(survey_cli, "_survey_run_service", fake)

    result = runner.invoke(app, ["survey", "list", "--limit", "5"])

    assert result.exit_code == 0, result.output
    assert "run_id" in result.output
    assert "run-1" in result.output
    assert "completed" in result.output
    fake.list_runs.assert_called_once_with(5)


def test_survey_browse(monkeypatch):
    """`survey browse` prints grouping rows."""
    fake = MagicMock()
    fake.browse_groups.return_value = [
        {"phase": "camera_inventory", "camera": "cam-a", "zoom": 1.0, "frame_count": 3},
    ]
    monkeypatch.setattr(survey_cli, "_survey_run_service", fake)

    result = runner.invoke(
        app,
        ["survey", "browse", "--run-id", "run-1", "--phase", "1", "--camera", "cam-a"],
    )

    assert result.exit_code == 0, result.output
    assert "camera_inventory" in result.output
    assert "cam-a" in result.output
    fake.browse_groups.assert_called_once_with("run-1", phase=1, camera="cam-a", zoom=None)


def test_survey_browse_empty(monkeypatch):
    """`survey browse` prints a no-groupings line and exits 0 when empty."""
    fake = MagicMock()
    fake.browse_groups.return_value = []
    monkeypatch.setattr(survey_cli, "_survey_run_service", fake)

    result = runner.invoke(app, ["survey", "browse", "--run-id", "run-1"])

    assert result.exit_code == 0, result.output
    assert "No groupings" in result.output
