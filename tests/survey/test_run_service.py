"""Unit tests for the C5 operator-surface service and in-memory orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING

from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig
from poc_homography.survey.orchestrator_memory import InMemorySurveyOrchestrator
from poc_homography.survey.run_service import SurveyRunService, build_survey_run_service

if TYPE_CHECKING:
    from poc_homography.domain.entities.survey.frame_record import FrameRecord


def _counting_ids() -> object:
    """Return a deterministic id factory yielding id-0, id-1, ..."""
    counter = {"n": -1}

    def _next() -> str:
        counter["n"] += 1
        return f"id-{counter['n']}"

    return _next


def _fixed_clock(moment: datetime):
    return lambda: moment


def _orchestrator(**kwargs) -> InMemorySurveyOrchestrator:
    return InMemorySurveyOrchestrator(
        id_factory=_counting_ids(),
        clock=_fixed_clock(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        frames_per_phase=2,
        **kwargs,
    )


class TestInMemoryOrchestrator:
    def test_start_assigns_run_and_sessions(self) -> None:
        orch = _orchestrator()
        handle = orch.start(SurveyPlanConfig(), ["cam-a", "cam-b"])
        assert handle.run_id == "id-0"
        assert set(handle.session_ids) == {"cam-a", "cam-b"}
        assert handle.session_ids["cam-a"] != handle.session_ids["cam-b"]

    def test_status_unknown_run_is_none(self) -> None:
        assert _orchestrator().status("missing") is None

    def test_iter_progress_completes_all_phases(self) -> None:
        orch = _orchestrator()
        handle = orch.start(SurveyPlanConfig(enabled_phases=frozenset({1, 2, 3})), ["cam-a"])
        events = list(orch.iter_progress(handle.run_id))
        # 3 phases x 1 camera
        assert len(events) == 3
        statuses = orch.status(handle.run_id)
        assert statuses is not None
        assert statuses["cam-a"].frame_count == 6  # 3 phases * 2 frames
        assert statuses["cam-a"].status == "completed"

    def test_abort_stops_progress(self) -> None:
        orch = _orchestrator()
        handle = orch.start(SurveyPlanConfig(), ["cam-a"])
        assert orch.abort(handle.run_id) is True
        assert list(orch.iter_progress(handle.run_id)) == []
        statuses = orch.status(handle.run_id)
        assert statuses is not None
        assert statuses["cam-a"].status == "aborted"

    def test_abort_unknown_run_is_false(self) -> None:
        assert _orchestrator().abort("missing") is False

    def test_list_runs_newest_first_and_limited(self) -> None:
        orch = _orchestrator()
        orch.start(SurveyPlanConfig(), ["cam-a"])
        orch.start(SurveyPlanConfig(), ["cam-b"])
        summaries = orch.list_runs(limit=1)
        assert len(summaries) == 1
        assert summaries[0].camera_count == 1


class TestSurveyRunService:
    def test_start_run_shape(self) -> None:
        service = SurveyRunService(_orchestrator())
        result = service.start_run(SurveyPlanConfig(), ["cam-a"])
        assert result["run_id"] == "id-0"
        assert list(result["session_ids"]) == ["cam-a"]  # type: ignore[arg-type]

    def test_get_status_unknown_is_none(self) -> None:
        assert SurveyRunService(_orchestrator()).get_status("missing") is None

    def test_get_status_reports_per_camera(self) -> None:
        service = SurveyRunService(_orchestrator())
        run = service.start_run(SurveyPlanConfig(enabled_phases=frozenset({1})), ["cam-a"])
        list(service.iter_progress(str(run["run_id"])))
        status = service.get_status(str(run["run_id"]))
        assert status is not None
        cams = status["cameras"]
        assert isinstance(cams, dict)
        assert cams["cam-a"]["frame_count"] == 2

    def test_abort_run_unknown_is_none(self) -> None:
        assert SurveyRunService(_orchestrator()).abort_run("missing") is None

    def test_abort_run_message(self) -> None:
        service = SurveyRunService(_orchestrator())
        run = service.start_run(SurveyPlanConfig(), ["cam-a"])
        result = service.abort_run(str(run["run_id"]))
        assert result == {"run_id": "id-0", "message": "Run abort requested"}

    def test_list_runs_dicts(self) -> None:
        service = SurveyRunService(_orchestrator())
        service.start_run(SurveyPlanConfig(), ["cam-a"])
        runs = service.list_runs(limit=20)
        assert runs[0]["run_id"] == "id-0"
        assert runs[0]["camera_count"] == 1


class _FakeRunRepo:
    """Minimal SurveyRunRepository fake returning duck-typed frame records."""

    def __init__(self, frames: list[object]) -> None:
        self._frames = frames

    def save_plan_config(self, run_id: str, config: SurveyPlanConfig) -> bool:
        return True

    def load_plan_config(self, run_id: str) -> SurveyPlanConfig:
        return SurveyPlanConfig()

    def get_frames_by_run(self, run_id: str) -> list[FrameRecord]:
        return self._frames  # type: ignore[return-value]


def _frame(phase_value: str, camera_id: str, zoom: float) -> object:
    return SimpleNamespace(
        capture=SimpleNamespace(phase=SimpleNamespace(value=phase_value)),
        camera=SimpleNamespace(camera_id=camera_id),
        reported=SimpleNamespace(reported_zoom=zoom),
    )


class TestBrowseGroups:
    def test_no_repo_returns_empty(self) -> None:
        assert SurveyRunService(_orchestrator()).browse_groups("run-1") == []

    def test_groups_and_counts(self) -> None:
        frames = [
            _frame("main_survey", "cam-a", 1.0),
            _frame("main_survey", "cam-a", 1.0),
            _frame("main_survey", "cam-a", 5.0),
            _frame("cross_zoom", "cam-b", 12.0),
        ]
        service = SurveyRunService(_orchestrator(), run_repo=_FakeRunRepo(frames))
        groups = service.browse_groups("run-1")
        assert {"phase": "main_survey", "camera": "cam-a", "zoom": 1.0, "frame_count": 2} in groups
        assert len(groups) == 3

    def test_filter_by_camera(self) -> None:
        frames = [_frame("main_survey", "cam-a", 1.0), _frame("cross_zoom", "cam-b", 12.0)]
        service = SurveyRunService(_orchestrator(), run_repo=_FakeRunRepo(frames))
        groups = service.browse_groups("run-1", camera="cam-b")
        assert len(groups) == 1
        assert groups[0]["camera"] == "cam-b"

    def test_filter_by_phase_number(self) -> None:
        frames = [_frame("camera_inventory", "cam-a", 1.0), _frame("main_survey", "cam-a", 1.0)]
        service = SurveyRunService(_orchestrator(), run_repo=_FakeRunRepo(frames))
        groups = service.browse_groups("run-1", phase=5)  # 5 -> main_survey
        assert len(groups) == 1
        assert groups[0]["phase"] == "main_survey"


def test_build_survey_run_service_default_wiring() -> None:
    service = build_survey_run_service()
    result = service.start_run(SurveyPlanConfig(), ["cam-a"])
    assert "run_id" in result
