"""Calibration tests — recover the mount offset and VFOV from samples."""

from __future__ import annotations

import pytest

from poc_homography.horizon import calibrate_tilt_offset
from poc_homography.types import Pixels, Unitless

from .conftest import (
    BASE_FOCAL_LENGTH_MM,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MEASURED_TABLE,
    SENSOR_WIDTH_MM,
    ZOOM,
)

# (reported_tilt, detected_horizon_row) from the in-frame measured rows.
SAMPLES = [
    (tilt, fraction * IMAGE_HEIGHT) for tilt, fraction in MEASURED_TABLE if fraction is not None
]


def _calibrate(samples):
    return calibrate_tilt_offset(
        samples,
        Unitless(ZOOM),
        Pixels(IMAGE_WIDTH),
        Pixels(IMAGE_HEIGHT),
        sensor_width_mm=SENSOR_WIDTH_MM,
        base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
    )


class TestCalibration:
    def test_recovers_offset_near_minus_31(self):
        result = _calibrate(SAMPLES)
        assert result.tilt_offset_deg == pytest.approx(-31.0, abs=3.0)

    def test_verifies_vfov_near_37(self):
        result = _calibrate(SAMPLES)
        assert result.vfov_deg == pytest.approx(37.0, abs=4.0)

    def test_reports_sample_count_and_low_residual(self):
        result = _calibrate(SAMPLES)
        assert result.n_samples == len(SAMPLES)
        assert result.rms_fraction_residual < 0.05

    def test_requires_two_samples(self):
        with pytest.raises(ValueError):
            _calibrate(SAMPLES[:1])
