"""Round-trip, schema-version, and enum-coverage tests for the run aggregate."""

from __future__ import annotations

import pytest
from tests.domain.survey.builders import make_survey_run

from poc_homography.domain.entities.survey import SURVEY_SCHEMA_VERSION
from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus


class TestSurveyRunRoundTrip:
    def test_round_trip_running(self) -> None:
        run = make_survey_run(status=SurveyRunStatus.RUNNING, finished=False)
        restored = SurveyRun.from_dict(run.to_dict())
        assert restored == run
        assert restored.to_dict() == run.to_dict()

    def test_round_trip_completed_with_finished_at(self) -> None:
        run = make_survey_run(status=SurveyRunStatus.COMPLETED, finished=True)
        restored = SurveyRun.from_dict(run.to_dict())
        assert restored.finished_at == run.finished_at
        assert restored.status is SurveyRunStatus.COMPLETED

    def test_finished_at_none_round_trips(self) -> None:
        run = make_survey_run(finished=False)
        assert run.to_dict()["finished_at"] is None
        restored = SurveyRun.from_dict(run.to_dict())
        assert restored.finished_at is None

    def test_phases_round_trip(self) -> None:
        run = make_survey_run()
        restored = SurveyRun.from_dict(run.to_dict())
        assert restored.phases == run.phases

    def test_id_is_run_id(self) -> None:
        run = make_survey_run(run_id="run-abc")
        assert run.id == "run-abc"

    def test_schema_version_present(self) -> None:
        run = make_survey_run()
        assert run.schema_version == SURVEY_SCHEMA_VERSION
        assert run.to_dict()["schema_version"] == SURVEY_SCHEMA_VERSION


class TestSurveyRunSchemaVersion:
    def test_unknown_version_raises(self) -> None:
        run_dict = make_survey_run().to_dict()
        run_dict["schema_version"] = "0.9"
        with pytest.raises(ValueError, match="schema_version"):
            SurveyRun.from_dict(run_dict)


class TestSurveyPhaseEnum:
    def test_nine_phases(self) -> None:
        assert len(list(SurveyPhase)) == 9

    def test_values_are_plain_strings(self) -> None:
        for phase in SurveyPhase:
            assert isinstance(phase.value, str)
            assert phase.value == phase.value.lower()

    def test_values_stable(self) -> None:
        assert SurveyPhase.STATIC_JITTER.value == "static_jitter"
        assert SurveyPhase.CAMERA_INVENTORY.value == "camera_inventory"
        assert SurveyPhase("validation") is SurveyPhase.VALIDATION


class TestSurveyRunStatusEnum:
    def test_values(self) -> None:
        assert {s.value for s in SurveyRunStatus} == {
            "pending",
            "running",
            "paused",
            "completed",
            "failed",
            "aborted",
        }
