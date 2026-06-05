"""Round-trip and schema tests for :class:`CameraInventoryRecord`."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from tests.survey.conftest import (
    make_capabilities,
    make_device_info,
    make_health,
    make_optics,
    make_preset,
    make_stream_profile,
)

from poc_homography.domain.entities.survey import SURVEY_SCHEMA_VERSION
from poc_homography.domain.entities.survey.inventory_record import (
    CameraInventoryRecord,
)
from poc_homography.domain.enums.survey_phase import SurveyPhase


def _make_record() -> CameraInventoryRecord:
    return CameraInventoryRecord(
        record_id="run-1_inventory",
        run_id="run-1",
        camera_id="icozee-camptz-04",
        captured_at=datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc),
        device_info=make_device_info(),
        capabilities=make_capabilities(),
        optics=make_optics(),
        health=make_health(),
        stream_profiles=(make_stream_profile(),),
        presets=(make_preset(),),
    )


class TestCameraInventoryRecord:
    def test_round_trip(self) -> None:
        record = _make_record()
        restored = CameraInventoryRecord.from_dict(record.to_dict())
        assert restored == record
        assert restored.to_dict() == record.to_dict()

    def test_phase_is_camera_inventory(self) -> None:
        record = _make_record()
        assert record.phase is SurveyPhase.CAMERA_INVENTORY
        assert record.to_dict()["phase"] == "camera_inventory"

    def test_id_is_record_id(self) -> None:
        assert _make_record().id == "run-1_inventory"

    def test_contains_full_self_report(self) -> None:
        data = _make_record().to_dict()
        # Identity + firmware, PTZ caps, image settings, stream config,
        # zoom/focus caps, and image-pipeline state are all present.
        assert data["device_info"]["firmware_version"] == "V5.7.3"
        assert data["capabilities"]["zoom_max"] == 25.0
        assert data["capabilities"]["focus_max"] == 1000
        assert data["optics"]["focus"]["style"] == "SEMIAUTOMATIC"
        assert data["stream_profiles"][0]["codec"] == "H.264"
        assert data["health"]["odometry"]["zoom_total_steps"] == 10
        assert data["presets"][0]["name"] == "home"

    def test_schema_version_default(self) -> None:
        assert _make_record().schema_version == SURVEY_SCHEMA_VERSION

    def test_unknown_version_raises(self) -> None:
        data = _make_record().to_dict()
        data["schema_version"] = "999.0"
        with pytest.raises(ValueError, match="schema_version"):
            CameraInventoryRecord.from_dict(data)
