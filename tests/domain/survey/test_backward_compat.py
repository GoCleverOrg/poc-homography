"""Backward-compat: an existing SurveySession manifest still loads unchanged."""

from __future__ import annotations

from pathlib import Path

from webapp.camera_survey.models import SurveySession, SurveyStatus

from poc_homography.infrastructure.repositories.repo_yaml_survey_session import (
    RepoYamlSurveySession,
)

_FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "survey"
_SESSION_ID = "11111111-1111-4111-8111-111111111111"


class TestSurveySessionBackwardCompat:
    def test_existing_manifest_loads(self) -> None:
        repo = RepoYamlSurveySession(_FIXTURE_DIR, SurveySession.from_dict)
        session = repo.get(_SESSION_ID)
        assert session is not None
        assert session.id == _SESSION_ID
        assert session.status is SurveyStatus.COMPLETED

    def test_captures_preserved(self) -> None:
        repo = RepoYamlSurveySession(_FIXTURE_DIR, SurveySession.from_dict)
        session = repo.get(_SESSION_ID)
        assert session is not None
        assert len(session.captures) == 2
        assert session.captures[0].filename == "frame_0000.jpg"
