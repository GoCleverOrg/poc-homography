"""PlannedSurveyRun resume / checkpoint tests for the survey planner."""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus
from poc_homography.survey.planner import (
    PhasePlan,
    PlannedSurveyRun,
    Pose,
)
from poc_homography.types import Degrees, Unitless


def _pose(i: int) -> Pose:
    return Pose(pan=Degrees(float(i)), tilt=Degrees(0.0), zoom=Unitless(1.0))


def _build_run() -> PlannedSurveyRun:
    phase_a = PhasePlan(phase=SurveyPhase.MAIN_SURVEY, poses=tuple(_pose(i) for i in range(4)))
    phase_b = PhasePlan(phase=SurveyPhase.CROSS_ZOOM, poses=tuple(_pose(10 + i) for i in range(3)))
    return PlannedSurveyRun(
        run_id="run-1",
        camera_id="cam-1",
        phases=(phase_a, phase_b),
        started_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )


class TestResume:
    def test_full_iteration_when_fresh(self):
        run = _build_run()
        items = list(run.remaining_poses())
        assert len(items) == 7

    def test_crash_and_resume_no_reexecution(self):
        run = _build_run()
        all_items = list(run.remaining_poses())
        total = len(all_items)

        completed: list[tuple[int, int]] = []
        crash_after = 3
        cursor_run = run
        for n, (pi, qi, _pose) in enumerate(run.remaining_poses()):
            if n >= crash_after:
                break
            completed.append((pi, qi))
            cursor_run = cursor_run.advance(pi, qi)

        # Reload from checkpoint.
        reloaded = PlannedSurveyRun.from_dict(cursor_run.to_dict())
        remaining = [(pi, qi) for pi, qi, _ in reloaded.remaining_poses()]

        # No re-execution: completed and remaining are disjoint.
        assert set(completed).isdisjoint(set(remaining))
        # Union equals the full pose set, no duplicates.
        union = completed + remaining
        assert len(union) == total
        assert len(set(union)) == total

    def test_status_transitions(self):
        run = _build_run()
        assert run.status is SurveyRunStatus.PENDING
        run = run.with_status(SurveyRunStatus.RUNNING)
        assert run.status is SurveyRunStatus.RUNNING
        run = run.with_status(SurveyRunStatus.PAUSED)
        assert run.status is SurveyRunStatus.PAUSED
        run = run.with_status(SurveyRunStatus.RUNNING)
        assert run.status is SurveyRunStatus.RUNNING
        run = run.with_status(SurveyRunStatus.COMPLETED)
        assert run.status is SurveyRunStatus.COMPLETED

    def test_roundtrip_serialization(self):
        run = _build_run().advance(0, 1).with_status(SurveyRunStatus.PAUSED)
        reloaded = PlannedSurveyRun.from_dict(run.to_dict())
        assert reloaded.run_id == run.run_id
        assert reloaded.camera_id == run.camera_id
        assert reloaded.status is SurveyRunStatus.PAUSED
        assert reloaded.cursor == run.cursor
        assert reloaded.to_dict() == run.to_dict()

    def test_header_returns_c1_survey_run(self):
        run = _build_run()
        header = run.header()
        assert isinstance(header, SurveyRun)
        assert header.run_id == run.run_id
        assert header.phases == frozenset({SurveyPhase.MAIN_SURVEY, SurveyPhase.CROSS_ZOOM})
