"""Tests for the concurrent multi-camera survey run Django views (issue #262).

The views delegate to ``webapp.camera_survey.views._survey_run_service``; each test
monkeypatches that singleton with a MagicMock returning canned dicts that match the
locked service contract, so the views are exercised without any real cameras.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add webapp to path for Django imports
PROJECT_ROOT = Path(__file__).parent.parent
WEBAPP_DIR = PROJECT_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

# Django test setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "homography_web.settings")

import django

django.setup()

from django.test import Client

from poc_homography.domain.vo import SurveyPlanConfig

# URL prefix under which camera_survey.urls is mounted (homography_web/urls.py).
PREFIX = "/camera-survey"


@pytest.fixture
def mock_service(monkeypatch):
    """Replace the survey run service singleton with a MagicMock.

    Django registers the app as ``camera_survey`` (webapp/ is on sys.path), so the
    URLconf resolves views via the ``camera_survey.views`` module object. We patch
    that exact module so the override reaches the running view; under this layout
    ``webapp.camera_survey.views`` is a *separate* module object and patching it has
    no effect on dispatched requests.
    """
    import camera_survey.views as views

    service = MagicMock()
    monkeypatch.setattr(views, "_survey_run_service", service)
    return service


@pytest.fixture
def client():
    return Client()


def test_start_run_success(client, mock_service) -> None:
    """POST survey/run/start/ with a valid body returns run_id + session_ids."""
    mock_service.start_run.return_value = {
        "run_id": "run-1",
        "session_ids": {"cam-a": "sess-a"},
    }
    body = {
        "plan_config": SurveyPlanConfig().to_dict(),
        "camera_ids": ["cam-a"],
    }
    resp = client.post(
        f"{PREFIX}/survey/run/start/",
        data=json.dumps(body),
        content_type="application/json",
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    assert payload["data"]["run_id"] == "run-1"
    assert payload["data"]["session_ids"] == {"cam-a": "sess-a"}
    # Service called with a SurveyPlanConfig and the camera_ids list.
    cfg_arg, cams_arg = mock_service.start_run.call_args.args
    assert isinstance(cfg_arg, SurveyPlanConfig)
    assert cams_arg == ["cam-a"]


def test_start_run_invalid_json(client, mock_service) -> None:
    """POST with non-JSON body returns 400."""
    resp = client.post(
        f"{PREFIX}/survey/run/start/",
        data="not json",
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_start_run_missing_plan_config(client, mock_service) -> None:
    """POST without plan_config returns 400."""
    resp = client.post(
        f"{PREFIX}/survey/run/start/",
        data=json.dumps({"camera_ids": ["cam-a"]}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    mock_service.start_run.assert_not_called()


def test_start_run_missing_camera_ids(client, mock_service) -> None:
    """POST without camera_ids returns 400."""
    resp = client.post(
        f"{PREFIX}/survey/run/start/",
        data=json.dumps({"plan_config": SurveyPlanConfig().to_dict()}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    mock_service.start_run.assert_not_called()


def test_start_run_camera_ids_not_strings(client, mock_service) -> None:
    """POST with non-string camera_ids returns 400."""
    resp = client.post(
        f"{PREFIX}/survey/run/start/",
        data=json.dumps({"plan_config": SurveyPlanConfig().to_dict(), "camera_ids": [1, 2]}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_start_run_bad_plan_config(client, mock_service) -> None:
    """POST with a plan_config that fails validation returns 400."""
    resp = client.post(
        f"{PREFIX}/survey/run/start/",
        data=json.dumps({"plan_config": {"schema_version": "999"}, "camera_ids": ["cam-a"]}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    mock_service.start_run.assert_not_called()


def test_status_success(client, mock_service) -> None:
    """GET survey/run/<id>/status/ returns per-camera phase + frame_count."""
    mock_service.get_status.return_value = {
        "run_id": "run-1",
        "cameras": {
            "cam-a": {
                "session_id": "sess-a",
                "phase": 3,
                "frame_count": 42,
                "status": "running",
            }
        },
    }
    resp = client.get(f"{PREFIX}/survey/run/run-1/status/")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["run_id"] == "run-1"
    cam = data["cameras"]["cam-a"]
    assert cam["phase"] == 3
    assert cam["frame_count"] == 42
    mock_service.get_status.assert_called_once_with("run-1")


def test_status_not_found(client, mock_service) -> None:
    """GET status for an unknown run returns 404."""
    mock_service.get_status.return_value = None
    resp = client.get(f"{PREFIX}/survey/run/nope/status/")
    assert resp.status_code == 404


def test_abort_success(client, mock_service) -> None:
    """POST survey/run/<id>/abort/ returns the abort message."""
    mock_service.abort_run.return_value = {"run_id": "run-1", "message": "Run abort requested"}
    resp = client.post(f"{PREFIX}/survey/run/run-1/abort/")
    assert resp.status_code == 200
    assert resp.json()["data"]["run_id"] == "run-1"
    mock_service.abort_run.assert_called_once_with("run-1")


def test_abort_not_found(client, mock_service) -> None:
    """POST abort for an unknown run returns 404."""
    mock_service.abort_run.return_value = None
    resp = client.post(f"{PREFIX}/survey/run/nope/abort/")
    assert resp.status_code == 404


def test_runs_list_success(client, mock_service) -> None:
    """GET survey/runs/ returns the runs list and echoes limit/offset."""
    mock_service.list_runs.return_value = [
        {
            "run_id": "run-1",
            "start_time": "2026-06-05T00:00:00",
            "camera_count": 1,
            "total_frame_count": 10,
            "status": "completed",
        }
    ]
    resp = client.get(f"{PREFIX}/survey/runs/")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["runs"][0]["run_id"] == "run-1"
    assert data["limit"] == 20
    assert data["offset"] == 0
    mock_service.list_runs.assert_called_once_with(limit=20)


def test_runs_list_bad_limit(client, mock_service) -> None:
    """GET survey/runs/ with a non-integer limit returns 400."""
    resp = client.get(f"{PREFIX}/survey/runs/?limit=abc")
    assert resp.status_code == 400


def test_groups_success_forwards_filters(client, mock_service) -> None:
    """GET survey/runs/<id>/groups/ forwards phase/camera/zoom to browse_groups."""
    mock_service.browse_groups.return_value = [
        {"phase": "p1", "camera": "cam-a", "zoom": 5.0, "frame_count": 7}
    ]
    resp = client.get(f"{PREFIX}/survey/runs/run-1/groups/?phase=3&camera=cam-a&zoom=5.0")
    assert resp.status_code == 200
    assert resp.json()["data"]["groups"][0]["frame_count"] == 7
    mock_service.browse_groups.assert_called_once_with("run-1", phase=3, camera="cam-a", zoom=5.0)


def test_groups_no_filters(client, mock_service) -> None:
    """GET groups/ without query params forwards None filters."""
    mock_service.browse_groups.return_value = []
    resp = client.get(f"{PREFIX}/survey/runs/run-1/groups/")
    assert resp.status_code == 200
    mock_service.browse_groups.assert_called_once_with("run-1", phase=None, camera=None, zoom=None)


def test_groups_bad_phase(client, mock_service) -> None:
    """GET groups/ with a non-integer phase returns 400."""
    resp = client.get(f"{PREFIX}/survey/runs/run-1/groups/?phase=abc")
    assert resp.status_code == 400


def test_groups_bad_zoom(client, mock_service) -> None:
    """GET groups/ with a non-numeric zoom returns 400."""
    resp = client.get(f"{PREFIX}/survey/runs/run-1/groups/?zoom=xyz")
    assert resp.status_code == 400
