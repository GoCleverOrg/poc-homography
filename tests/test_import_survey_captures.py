"""Tests for import_survey_captures — survey→CapturedFrame bridge (issue #229)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from webapp.camera_survey.models import (
    CameraInfo,
    CaptureRecord,
    PTZPosition,
    SurveySession,
    SurveyStatus,
    TenantInfo,
)
from webapp.camera_survey.services import import_survey_captures

from poc_homography.infrastructure.repositories.repo_yaml_captured_frame import (
    RepoYamlCapturedFrame,
)
from poc_homography.types import Degrees, Unitless


def _make_session(
    *,
    status: SurveyStatus = SurveyStatus.COMPLETED,
    tenant: TenantInfo | None = TenantInfo(id="valte", name="Valte"),
    camera: CameraInfo | None = CameraInfo(id="valte_cam01", name="cam01", ip="1.2.3.4"),
    captures: list[CaptureRecord] | None = None,
) -> SurveySession:
    if captures is None:
        captures = [
            CaptureRecord(
                filename="valte_cam01_20240115_103045_30.0_15.0_1.0.jpg",
                timestamp="2024-01-15T10:30:45+00:00",
                ptz=PTZPosition(pan=30.0, tilt=15.0, zoom=1.0),
                step_index=0,
            ),
        ]
    return SurveySession(
        id="test-session-001",
        start_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        status=status,
        tenant=tenant,
        camera=camera,
        captures=captures,
    )


@pytest.fixture()
def frame_repo(tmp_path: Path) -> RepoYamlCapturedFrame:
    return RepoYamlCapturedFrame(tmp_path / "captured_frames")


@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    """Create a fake session directory with an image file."""
    d = tmp_path / "survey" / "20240115" / "test-session-001"
    d.mkdir(parents=True)
    (d / "valte_cam01_20240115_103045_30.0_15.0_1.0.jpg").write_bytes(b"\xff\xd8fake")
    return d


@pytest.fixture()
def _mock_deps(tmp_path: Path, frame_repo: RepoYamlCapturedFrame, session_dir: Path):
    """Patch external dependencies for import_survey_captures."""
    mock_map = MagicMock()
    mock_map.id = "valte"

    with (
        patch(
            "webapp.homography_web.frame_utils.get_map_from_tenant_id",
            return_value=mock_map,
        ),
        patch(
            "webapp.homography_web.frame_utils.get_frame_repo",
            return_value=frame_repo,
        ),
        patch(
            "webapp.homography_web.frame_utils.invalidate_cache",
        ) as mock_invalidate,
        patch(
            "webapp.camera_survey.services._get_session_path",
            return_value=session_dir,
        ),
    ):
        yield mock_invalidate


class TestSuccessfulImport:
    @pytest.mark.usefixtures("_mock_deps")
    def test_creates_captured_frame_yaml(
        self, frame_repo: RepoYamlCapturedFrame
    ) -> None:
        session = _make_session()
        count = import_survey_captures(session)

        assert count == 1
        frames = frame_repo.get_by_camera("valte", "cam01")
        assert len(frames) == 1
        frame = frames[0]
        assert frame.map_id == "valte"
        assert frame.camera_name == "cam01"
        assert frame.image_path == Path("valte_cam01_20240115_103045_30.0_15.0_1.0.jpg")

    @pytest.mark.usefixtures("_mock_deps")
    def test_copies_image_file(
        self, frame_repo: RepoYamlCapturedFrame
    ) -> None:
        session = _make_session()
        import_survey_captures(session)

        dest = frame_repo.image_dir_for("valte", "cam01")
        assert (dest / "valte_cam01_20240115_103045_30.0_15.0_1.0.jpg").exists()

    @pytest.mark.usefixtures("_mock_deps")
    def test_ptz_mapping(self, frame_repo: RepoYamlCapturedFrame) -> None:
        session = _make_session()
        import_survey_captures(session)

        frame = frame_repo.get_by_camera("valte", "cam01")[0]
        assert frame.ptz_state.pan_raw == Degrees(30.0)
        assert frame.ptz_state.tilt_deg == Degrees(15.0)
        assert frame.ptz_state.zoom == Unitless(1.0)

    @pytest.mark.usefixtures("_mock_deps")
    def test_invalidates_cache(self, _mock_deps) -> None:
        session = _make_session()
        import_survey_captures(session)
        _mock_deps.assert_called_once()


class TestSkipOnErrors:
    @pytest.mark.usefixtures("_mock_deps")
    def test_capture_with_errors_skipped(
        self, frame_repo: RepoYamlCapturedFrame
    ) -> None:
        captures = [
            CaptureRecord(
                filename="good.jpg",
                timestamp="2024-01-15T10:30:45+00:00",
                ptz=PTZPosition(pan=30.0, tilt=15.0, zoom=1.0),
                step_index=0,
            ),
            CaptureRecord(
                filename="bad.jpg",
                timestamp="2024-01-15T10:31:00+00:00",
                ptz=PTZPosition(pan=35.0, tilt=15.0, zoom=1.0),
                step_index=1,
                errors=["RTSP timeout"],
            ),
        ]
        session = _make_session(captures=captures)
        count = import_survey_captures(session)

        assert count == 1
        assert len(frame_repo.get_by_camera("valte", "cam01")) == 1


class TestIdempotentRerun:
    @pytest.mark.usefixtures("_mock_deps")
    def test_second_run_imports_zero(
        self, frame_repo: RepoYamlCapturedFrame
    ) -> None:
        session = _make_session()
        first = import_survey_captures(session)
        second = import_survey_captures(session)

        assert first == 1
        assert second == 0
        assert len(frame_repo.get_by_camera("valte", "cam01")) == 1


class TestSkipOnWrongStatus:
    """import_survey_captures is only called when status==COMPLETED, but the
    function itself doesn't check status — the caller (_execute_survey) does.
    This test documents that the guard lives at the call-site."""

    def test_aborted_session_not_called(self) -> None:
        """_execute_survey only calls import when COMPLETED."""
        # This is an integration-level concern; unit test verifies the guard
        # is in _execute_survey by checking the conditional.
        import inspect

        from webapp.camera_survey import services

        source = inspect.getsource(services.CameraSurveyService._execute_survey)
        assert "if session.status == SurveyStatus.COMPLETED" in source
        assert "import_survey_captures" in source


class TestSkipOnMissingTenant:
    def test_none_tenant_returns_zero(self) -> None:
        session = _make_session(tenant=None)
        count = import_survey_captures(session)
        assert count == 0


class TestSkipOnMissingCamera:
    def test_none_camera_returns_zero(self) -> None:
        session = _make_session(camera=None)
        count = import_survey_captures(session)
        assert count == 0


class TestSkipOnMissingMap:
    def test_no_map_returns_zero(self) -> None:
        with patch(
            "webapp.homography_web.frame_utils.get_map_from_tenant_id",
            return_value=None,
        ):
            session = _make_session()
            count = import_survey_captures(session)
            assert count == 0


class TestPtzNoneHandling:
    @pytest.mark.usefixtures("_mock_deps")
    def test_none_ptz_values_become_zero(
        self, frame_repo: RepoYamlCapturedFrame
    ) -> None:
        captures = [
            CaptureRecord(
                filename="nullptz.jpg",
                timestamp="2024-01-15T10:30:45+00:00",
                ptz=PTZPosition(pan=None, tilt=None, zoom=None),
                step_index=0,
            ),
        ]
        session = _make_session(captures=captures)
        import_survey_captures(session)

        frame = frame_repo.get_by_camera("valte", "cam01")[0]
        assert frame.ptz_state.pan_raw == Degrees(0.0)
        assert frame.ptz_state.tilt_deg == Degrees(0.0)
        assert frame.ptz_state.zoom == Unitless(0.0)
