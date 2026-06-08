"""VO round-trip tests for the horizon value objects."""

from __future__ import annotations

import pytest

from poc_homography.horizon import CalibrationResult, FramePlacement, HorizonEstimate
from poc_homography.horizon.models import HORIZON_SCHEMA_VERSION


class TestHorizonEstimateRoundTrip:
    def test_in_frame_round_trip(self):
        est = HorizonEstimate(
            placement=FramePlacement.IN_FRAME,
            image_height=1440,
            row=681.0,
            ground_fraction=0.527,
            method="geometric",
            confidence=0.9,
        )
        assert HorizonEstimate.from_dict(est.to_dict()) == est

    def test_above_frame_round_trip_with_none_row(self):
        est = HorizonEstimate(
            placement=FramePlacement.ABOVE_FRAME,
            image_height=1440,
            row=None,
            ground_fraction=1.0,
        )
        restored = HorizonEstimate.from_dict(est.to_dict())
        assert restored == est
        assert restored.row is None

    def test_to_dict_is_json_serialisable(self):
        import json

        est = HorizonEstimate(placement=FramePlacement.BELOW_FRAME, image_height=1440)
        assert json.loads(json.dumps(est.to_dict()))["placement"] == "below_frame"

    def test_rejects_unknown_schema_version(self):
        payload = HorizonEstimate(
            placement=FramePlacement.IN_FRAME, image_height=1440, row=1.0
        ).to_dict()
        payload["schema_version"] = "999"
        with pytest.raises(ValueError):
            HorizonEstimate.from_dict(payload)


class TestCalibrationResultRoundTrip:
    def test_round_trip(self):
        result = CalibrationResult(
            tilt_offset_deg=-30.8,
            vfov_deg=36.3,
            zoom=1.0,
            rms_fraction_residual=0.02,
            n_samples=4,
        )
        assert CalibrationResult.from_dict(result.to_dict()) == result

    def test_default_schema_version(self):
        result = CalibrationResult(
            tilt_offset_deg=-31.0,
            vfov_deg=37.0,
            zoom=1.0,
            rms_fraction_residual=0.0,
            n_samples=2,
        )
        assert result.schema_version == HORIZON_SCHEMA_VERSION

    def test_rejects_unknown_schema_version(self):
        payload = CalibrationResult(
            tilt_offset_deg=-31.0,
            vfov_deg=37.0,
            zoom=1.0,
            rms_fraction_residual=0.0,
            n_samples=2,
        ).to_dict()
        payload["schema_version"] = "2"
        with pytest.raises(ValueError):
            CalibrationResult.from_dict(payload)
