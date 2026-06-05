"""Round-trip and schema-version tests for FrameRecord and its sub-VOs."""

from __future__ import annotations

import pytest
from tests.domain.survey.builders import make_frame_record

from poc_homography.domain.entities.survey import SURVEY_SCHEMA_VERSION
from poc_homography.domain.entities.survey.frame_record import FrameRecord
from poc_homography.domain.enums.survey_phase import SurveyPhase


class TestFrameRecordRoundTrip:
    def test_round_trip_all_fields(self) -> None:
        record = make_frame_record(burst_id="burst-9", frame_index=4)
        restored = FrameRecord.from_dict(record.to_dict())
        assert restored == record
        assert restored.to_dict() == record.to_dict()

    def test_round_trip_with_optional_none(self) -> None:
        record = make_frame_record(burst_id=None)
        record_dict = record.to_dict()
        # Force the nullable fields to None to exercise the optional path.
        record_dict["capture"]["burst_id"] = None
        record_dict["commanded"]["commanded_focus"] = None
        record_dict["reported"]["reported_azimuth"] = None
        record_dict["reported"]["reported_elevation"] = None
        record_dict["reported"]["reported_focal_length_mm"] = None
        record_dict["reported"]["reported_focus"] = None
        restored = FrameRecord.from_dict(record_dict)
        assert restored.capture.burst_id is None
        assert restored.commanded.commanded_focus is None
        assert restored.reported.reported_azimuth is None
        assert restored.reported.reported_elevation is None
        assert restored.reported.reported_focal_length_mm is None
        assert restored.reported.reported_focus is None
        assert restored.to_dict() == record_dict

    def test_id_is_capture_id(self) -> None:
        record = make_frame_record(capture_id="cap-xyz")
        assert record.id == "cap-xyz"

    def test_schema_version_present_and_default(self) -> None:
        record = make_frame_record()
        assert record.schema_version == SURVEY_SCHEMA_VERSION
        assert record.to_dict()["schema_version"] == SURVEY_SCHEMA_VERSION

    def test_phase_serialised_as_string(self) -> None:
        record = make_frame_record(phase=SurveyPhase.DENSE_NADIR)
        assert record.to_dict()["capture"]["phase"] == "dense_nadir"


class TestFrameRecordSchemaVersion:
    def test_unknown_version_raises(self) -> None:
        record_dict = make_frame_record().to_dict()
        record_dict["schema_version"] = "999.0"
        with pytest.raises(ValueError, match="schema_version"):
            FrameRecord.from_dict(record_dict)
