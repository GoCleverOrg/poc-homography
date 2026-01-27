"""Tests for survey automation module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from poc_homography.calibration.lens_distortion.survey_automation import (
    CALIBRATION_SURVEY_PRESETS,
    CameraInfo,
    SurveyAutomation,
    SurveyAutomationError,
    SurveyAxis,
    SurveyProgress,
    SurveySession,
    SurveyStatus,
    TenantInfo,
)


class TestSurveyAutomation:
    """Tests for SurveyAutomation class."""

    def test_init_default_url(self) -> None:
        """Test default base URL is set correctly."""
        automation = SurveyAutomation()
        assert automation.base_url == "http://localhost:8000"

    def test_init_custom_url(self) -> None:
        """Test custom base URL is set correctly."""
        automation = SurveyAutomation("http://example.com:9000/")
        assert automation.base_url == "http://example.com:9000"  # Trailing slash stripped

    def test_api_url_building(self) -> None:
        """Test API URL is built correctly."""
        automation = SurveyAutomation("http://localhost:8000")
        url = automation._api_url("/api/tenants/")
        assert url == "http://localhost:8000/camera-survey/api/tenants/"

    @patch.object(requests.Session, "request")
    def test_get_tenants_success(self, mock_request: MagicMock) -> None:
        """Test successful tenant retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "tenants": [
                    {"id": "t1", "name": "Tenant 1", "description": "First tenant"},
                    {"id": "t2", "name": "Tenant 2"},
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        automation = SurveyAutomation()
        tenants = automation.get_tenants()

        assert len(tenants) == 2
        assert isinstance(tenants[0], TenantInfo)
        assert tenants[0].id == "t1"
        assert tenants[0].name == "Tenant 1"
        assert tenants[0].description == "First tenant"
        assert tenants[1].id == "t2"
        assert tenants[1].description == ""

    @patch.object(requests.Session, "request")
    def test_get_cameras_success(self, mock_request: MagicMock) -> None:
        """Test successful camera retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "cameras": [
                    {"id": "c1", "name": "Camera 1", "ip": "192.168.1.100", "model": "PTZ"},
                    {"id": "c2", "name": "Camera 2", "ip": "192.168.1.101"},
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        automation = SurveyAutomation()
        cameras = automation.get_cameras("t1")

        assert len(cameras) == 2
        assert isinstance(cameras[0], CameraInfo)
        assert cameras[0].id == "c1"
        assert cameras[0].name == "Camera 1"
        assert cameras[0].ip == "192.168.1.100"
        assert cameras[0].model == "PTZ"
        assert cameras[1].model is None

    @patch.object(requests.Session, "request")
    def test_start_survey_success(self, mock_request: MagicMock) -> None:
        """Test successful survey start."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"session_id": "abc123", "message": "Survey started"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        automation = SurveyAutomation()
        session_id = automation.start_survey(
            tenant_id="t1",
            camera_id="c1",
            axis="pan",
            start=0,
            end=360,
            step=15,
            session_tags=["test"],
        )

        assert session_id == "abc123"

        # Verify request payload
        call_args = mock_request.call_args
        assert call_args[0][0] == "POST"
        payload = call_args[1]["json"]
        assert payload["tenant_id"] == "t1"
        assert payload["camera_id"] == "c1"
        assert payload["axis"] == "pan"
        assert payload["start"] == 0
        assert payload["end"] == 360
        assert payload["step"] == 15
        assert payload["session_tags"] == ["test"]

    @patch.object(requests.Session, "request")
    def test_start_survey_with_enum_axis(self, mock_request: MagicMock) -> None:
        """Test survey start with SurveyAxis enum."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"session_id": "xyz789"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        automation = SurveyAutomation()
        session_id = automation.start_survey(
            tenant_id="t1",
            camera_id="c1",
            axis=SurveyAxis.TILT,
            start=-30,
            end=60,
            step=10,
        )

        assert session_id == "xyz789"
        call_args = mock_request.call_args
        payload = call_args[1]["json"]
        assert payload["axis"] == "tilt"

    @patch.object(requests.Session, "request")
    def test_get_survey_status_running(self, mock_request: MagicMock) -> None:
        """Test getting status of running survey."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "session_id": "abc123",
                "status": "running",
                "step_count": 5,
                "total_steps": 24,
                "current_ptz": {"pan": 75.0, "tilt": 30.0, "zoom": 1.0},
                "last_capture_path": "/path/to/image.jpg",
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        automation = SurveyAutomation()
        progress = automation.get_survey_status("abc123")

        assert isinstance(progress, SurveyProgress)
        assert progress.session_id == "abc123"
        assert progress.status == SurveyStatus.RUNNING
        assert progress.step_count == 5
        assert progress.total_steps == 24
        assert progress.current_ptz == {"pan": 75.0, "tilt": 30.0, "zoom": 1.0}
        assert progress.last_capture_path == "/path/to/image.jpg"

    @patch.object(requests.Session, "request")
    def test_get_session_success(self, mock_request: MagicMock) -> None:
        """Test getting session details."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "session_id": "abc123",
                "tenant_id": "t1",
                "camera_id": "c1",
                "axis": "pan",
                "start": 0,
                "end": 360,
                "step": 15,
                "status": "completed",
                "captures": [
                    {"ptz": {"pan": 0, "tilt": 30, "zoom": 1}, "path": "img1.jpg"},
                    {"ptz": {"pan": 15, "tilt": 30, "zoom": 1}, "path": "img2.jpg"},
                ],
                "session_path": "/path/to/session",
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        automation = SurveyAutomation()
        session = automation.get_session("abc123")

        assert isinstance(session, SurveySession)
        assert session.session_id == "abc123"
        assert session.tenant_id == "t1"
        assert session.camera_id == "c1"
        assert session.axis == "pan"
        assert session.status == SurveyStatus.COMPLETED
        assert len(session.captures) == 2
        assert session.session_path == "/path/to/session"

    @patch.object(requests.Session, "request")
    def test_api_error_handling(self, mock_request: MagicMock) -> None:
        """Test API error response handling."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "error",
            "message": "Camera not found",
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        automation = SurveyAutomation()
        with pytest.raises(SurveyAutomationError) as exc_info:
            automation.get_cameras("invalid_tenant")

        assert "Camera not found" in str(exc_info.value)

    @patch.object(requests.Session, "request")
    def test_request_exception_handling(self, mock_request: MagicMock) -> None:
        """Test request exception handling."""
        mock_request.side_effect = requests.RequestException("Connection failed")

        automation = SurveyAutomation()
        with pytest.raises(SurveyAutomationError) as exc_info:
            automation.get_tenants()

        assert "API request failed" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)

    @patch.object(requests.Session, "request")
    def test_list_sessions_success(self, mock_request: MagicMock) -> None:
        """Test listing sessions."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "sessions": [
                    {"session_id": "s1", "status": "completed"},
                    {"session_id": "s2", "status": "running"},
                ],
                "total": 10,
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        automation = SurveyAutomation()
        sessions, total = automation.list_sessions(limit=2, offset=0)

        assert len(sessions) == 2
        assert total == 10

    @patch.object(requests.Session, "request")
    @patch("time.sleep")
    def test_wait_for_completion_success(
        self, mock_sleep: MagicMock, mock_request: MagicMock
    ) -> None:
        """Test waiting for survey completion."""
        # First call returns running, second returns completed
        mock_response_running = MagicMock()
        mock_response_running.json.return_value = {
            "status": "success",
            "data": {
                "session_id": "abc123",
                "status": "running",
                "step_count": 12,
                "total_steps": 24,
            },
        }
        mock_response_running.raise_for_status = MagicMock()

        mock_response_completed = MagicMock()
        mock_response_completed.json.return_value = {
            "status": "success",
            "data": {
                "session_id": "abc123",
                "status": "completed",
                "step_count": 24,
                "total_steps": 24,
            },
        }
        mock_response_completed.raise_for_status = MagicMock()

        mock_request.side_effect = [mock_response_running, mock_response_completed]

        automation = SurveyAutomation()
        progress = automation.wait_for_completion("abc123", timeout_seconds=60)

        assert progress.status == SurveyStatus.COMPLETED
        assert progress.step_count == 24

    @patch.object(requests.Session, "request")
    @patch("time.sleep")
    def test_wait_for_completion_with_callback(
        self, mock_sleep: MagicMock, mock_request: MagicMock
    ) -> None:
        """Test progress callback is called."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "session_id": "abc123",
                "status": "completed",
                "step_count": 24,
                "total_steps": 24,
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        callback = MagicMock()
        automation = SurveyAutomation()
        automation.wait_for_completion("abc123", progress_callback=callback)

        assert callback.called

    @patch.object(requests.Session, "request")
    @patch("time.sleep")
    def test_wait_for_completion_failed(
        self, mock_sleep: MagicMock, mock_request: MagicMock
    ) -> None:
        """Test waiting for failed survey raises error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "session_id": "abc123",
                "status": "failed",
                "error": "Camera disconnected",
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        automation = SurveyAutomation()
        with pytest.raises(SurveyAutomationError) as exc_info:
            automation.wait_for_completion("abc123")

        assert "Survey failed" in str(exc_info.value)
        assert "Camera disconnected" in str(exc_info.value)

    @patch.object(requests.Session, "request")
    @patch("time.time")
    def test_wait_for_completion_timeout(
        self, mock_time: MagicMock, mock_request: MagicMock
    ) -> None:
        """Test timeout during wait."""
        # Simulate time passing beyond timeout
        mock_time.side_effect = [0, 0, 100]  # start, first poll, check after poll

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "session_id": "abc123",
                "status": "running",
                "step_count": 5,
                "total_steps": 24,
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response

        automation = SurveyAutomation()
        with pytest.raises(SurveyAutomationError) as exc_info:
            automation.wait_for_completion("abc123", timeout_seconds=10)

        assert "timed out" in str(exc_info.value)


class TestSurveyEnums:
    """Tests for survey enums."""

    def test_survey_axis_values(self) -> None:
        """Test SurveyAxis enum values."""
        assert SurveyAxis.PAN.value == "pan"
        assert SurveyAxis.TILT.value == "tilt"
        assert SurveyAxis.ZOOM.value == "zoom"

    def test_survey_status_values(self) -> None:
        """Test SurveyStatus enum values."""
        assert SurveyStatus.PENDING.value == "pending"
        assert SurveyStatus.RUNNING.value == "running"
        assert SurveyStatus.COMPLETED.value == "completed"
        assert SurveyStatus.FAILED.value == "failed"
        assert SurveyStatus.ABORTED.value == "aborted"


class TestCalibrationPresets:
    """Tests for calibration survey presets."""

    def test_presets_exist(self) -> None:
        """Test that presets are defined."""
        assert len(CALIBRATION_SURVEY_PRESETS) >= 3

    def test_presets_have_required_fields(self) -> None:
        """Test that all presets have required fields."""
        required_fields = {"name", "description", "axis", "start", "end", "step", "tags"}
        for preset in CALIBRATION_SURVEY_PRESETS:
            assert set(preset.keys()) >= required_fields

    def test_preset_axes_are_valid(self) -> None:
        """Test that preset axes are valid."""
        valid_axes = {"pan", "tilt", "zoom"}
        for preset in CALIBRATION_SURVEY_PRESETS:
            assert preset["axis"] in valid_axes

    def test_preset_step_is_positive(self) -> None:
        """Test that preset steps are positive."""
        for preset in CALIBRATION_SURVEY_PRESETS:
            assert preset["step"] > 0


class TestDataClasses:
    """Tests for data classes."""

    def test_tenant_info_creation(self) -> None:
        """Test TenantInfo dataclass."""
        tenant = TenantInfo(id="t1", name="Test Tenant", description="A test")
        assert tenant.id == "t1"
        assert tenant.name == "Test Tenant"
        assert tenant.description == "A test"

    def test_camera_info_creation(self) -> None:
        """Test CameraInfo dataclass."""
        camera = CameraInfo(id="c1", name="Test Camera", ip="192.168.1.1", model="PTZ")
        assert camera.id == "c1"
        assert camera.model == "PTZ"

    def test_camera_info_optional_model(self) -> None:
        """Test CameraInfo with optional model."""
        camera = CameraInfo(id="c1", name="Test", ip="192.168.1.1")
        assert camera.model is None

    def test_survey_progress_creation(self) -> None:
        """Test SurveyProgress dataclass."""
        progress = SurveyProgress(
            session_id="s1",
            status=SurveyStatus.RUNNING,
            step_count=5,
            total_steps=10,
            current_ptz={"pan": 45.0},
            last_capture_path="/path/img.jpg",
        )
        assert progress.step_count == 5
        assert progress.error_message is None

    def test_survey_session_creation(self) -> None:
        """Test SurveySession dataclass."""
        session = SurveySession(
            session_id="s1",
            tenant_id="t1",
            camera_id="c1",
            axis="pan",
            start=0,
            end=360,
            step=15,
            status=SurveyStatus.COMPLETED,
            captures=[{"path": "img.jpg"}],
        )
        assert session.session_path is None
        assert len(session.captures) == 1
