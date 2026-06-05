"""Integration tests for the multi-camera survey-run endpoints (issue #262)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from poc_homography.domain.vo import SurveyPlanConfig

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

_TARGET = "api.routers.camera_evaluation._survey_run_service"


# ---------------------------------------------------------------------------
# POST /camera-evaluation/survey/run/start/
# ---------------------------------------------------------------------------


class TestSurveyRunStart:
    """Tests for ``POST /camera-evaluation/survey/run/start/``."""

    def test_starts_run(self, client: TestClient) -> None:
        mock_service = MagicMock()
        mock_service.start_run.return_value = {
            "run_id": "run-1",
            "session_ids": {"cam-a": "sess-a"},
        }

        body = {"plan_config": SurveyPlanConfig().to_dict(), "camera_ids": ["cam-a"]}

        with patch(_TARGET, mock_service):
            resp = client.post("/camera-evaluation/survey/run/start/", json=body)

        assert resp.status_code == 200
        assert resp.json() == {
            "status": "success",
            "data": {"run_id": "run-1", "session_ids": {"cam-a": "sess-a"}},
        }
        # SurveyPlanConfig.from_dict was applied: a config object was passed.
        cfg_arg, ids_arg = mock_service.start_run.call_args.args
        assert isinstance(cfg_arg, SurveyPlanConfig)
        assert ids_arg == ["cam-a"]

    def test_invalid_plan_config_returns_400(self, client: TestClient) -> None:
        mock_service = MagicMock()
        # Bad schema_version makes SurveyPlanConfig.from_dict raise ValueError.
        body = {"plan_config": {"schema_version": "999"}, "camera_ids": ["cam-a"]}

        with patch(_TARGET, mock_service):
            resp = client.post("/camera-evaluation/survey/run/start/", json=body)

        assert resp.status_code == 400
        assert resp.json()["status"] == "error"
        mock_service.start_run.assert_not_called()

    def test_missing_field_returns_422(self, client: TestClient) -> None:
        with patch(_TARGET, MagicMock()):
            resp = client.post(
                "/camera-evaluation/survey/run/start/",
                json={"camera_ids": ["cam-a"]},
            )

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /camera-evaluation/survey/run/{run_id}/status/
# ---------------------------------------------------------------------------


class TestSurveyRunStatus:
    """Tests for ``GET /camera-evaluation/survey/run/{run_id}/status/``."""

    def test_returns_status(self, client: TestClient) -> None:
        mock_service = MagicMock()
        mock_service.get_status.return_value = {
            "run_id": "run-1",
            "cameras": {
                "cam-a": {
                    "session_id": "sess-a",
                    "phase": 5,
                    "frame_count": 42,
                    "status": "running",
                }
            },
        }

        with patch(_TARGET, mock_service):
            resp = client.get("/camera-evaluation/survey/run/run-1/status/")

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["cameras"]["cam-a"]["phase"] == 5
        assert data["cameras"]["cam-a"]["frame_count"] == 42

    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        mock_service = MagicMock()
        mock_service.get_status.return_value = None

        with patch(_TARGET, mock_service):
            resp = client.get("/camera-evaluation/survey/run/nope/status/")

        assert resp.status_code == 404
        assert resp.json()["status"] == "error"


# ---------------------------------------------------------------------------
# POST /camera-evaluation/survey/run/{run_id}/abort/
# ---------------------------------------------------------------------------


class TestSurveyRunAbort:
    """Tests for ``POST /camera-evaluation/survey/run/{run_id}/abort/``."""

    def test_aborts_run(self, client: TestClient) -> None:
        mock_service = MagicMock()
        mock_service.abort_run.return_value = {
            "run_id": "run-1",
            "message": "Run abort requested",
        }

        with patch(_TARGET, mock_service):
            resp = client.post("/camera-evaluation/survey/run/run-1/abort/")

        assert resp.status_code == 200
        assert resp.json()["data"]["run_id"] == "run-1"

    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        mock_service = MagicMock()
        mock_service.abort_run.return_value = None

        with patch(_TARGET, mock_service):
            resp = client.post("/camera-evaluation/survey/run/nope/abort/")

        assert resp.status_code == 404
        assert resp.json()["status"] == "error"


# ---------------------------------------------------------------------------
# GET /camera-evaluation/survey/runs/
# ---------------------------------------------------------------------------


class TestSurveyRuns:
    """Tests for ``GET /camera-evaluation/survey/runs/``."""

    def test_returns_runs(self, client: TestClient) -> None:
        mock_service = MagicMock()
        mock_service.list_runs.return_value = [
            {
                "run_id": "run-1",
                "start_time": "2026-06-05T00:00:00",
                "camera_count": 2,
                "total_frame_count": 100,
                "status": "completed",
            }
        ]

        with patch(_TARGET, mock_service):
            resp = client.get("/camera-evaluation/survey/runs/", params={"limit": 5})

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["limit"] == 5
        assert data["offset"] == 0
        assert data["runs"][0]["run_id"] == "run-1"
        mock_service.list_runs.assert_called_once_with(limit=5)


# ---------------------------------------------------------------------------
# GET /camera-evaluation/survey/runs/{run_id}/groups/
# ---------------------------------------------------------------------------


class TestSurveyRunGroups:
    """Tests for ``GET /camera-evaluation/survey/runs/{run_id}/groups/``."""

    def test_returns_groups_with_filters(self, client: TestClient) -> None:
        mock_service = MagicMock()
        mock_service.browse_groups.return_value = [
            {"phase": "main_survey", "camera": "cam-a", "zoom": 1.0, "frame_count": 7}
        ]

        with patch(_TARGET, mock_service):
            resp = client.get(
                "/camera-evaluation/survey/runs/run-1/groups/",
                params={"phase": 5, "camera": "cam-a", "zoom": 1.0},
            )

        assert resp.status_code == 200
        assert resp.json()["data"]["groups"][0]["frame_count"] == 7
        mock_service.browse_groups.assert_called_once_with(
            "run-1", phase=5, camera="cam-a", zoom=1.0
        )
