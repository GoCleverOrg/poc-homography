"""Geometric horizon predictor tests against the live calibration table."""

from __future__ import annotations

import pytest

from poc_homography.horizon import (
    FramePlacement,
    all_ground_tilt_threshold,
    predict_horizon,
    vertical_fov_degrees,
)
from poc_homography.types import Degrees, Pixels, Unitless

from .conftest import (
    BASE_FOCAL_LENGTH_MM,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MEASURED_TABLE,
    SENSOR_WIDTH_MM,
    ZOOM,
)


def _predict(tilt: float):
    return predict_horizon(
        Degrees(tilt),
        Unitless(ZOOM),
        Pixels(IMAGE_WIDTH),
        Pixels(IMAGE_HEIGHT),
        sensor_width_mm=SENSOR_WIDTH_MM,
        base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
    )


class TestGeometricPrediction:
    @pytest.mark.parametrize("tilt,fraction", [t for t in MEASURED_TABLE if t[1] is not None])
    def test_reproduces_measured_rows(self, tilt: float, fraction: float):
        est = _predict(tilt)
        assert est.placement is FramePlacement.IN_FRAME
        assert est.row is not None
        predicted_fraction = est.row / IMAGE_HEIGHT
        # Live table is hand-measured; geometry tracks it within 5% of height.
        assert predicted_fraction == pytest.approx(fraction, abs=0.05)

    def test_ground_fraction_complements_row(self):
        est = _predict(-30.0)
        assert est.row is not None
        assert est.ground_fraction == pytest.approx((IMAGE_HEIGHT - est.row) / IMAGE_HEIGHT)

    @pytest.mark.parametrize("tilt", [-10.0, 0.0, 10.0])
    def test_shallow_tilt_is_all_ground(self, tilt: float):
        est = _predict(tilt)
        assert est.placement is FramePlacement.ABOVE_FRAME
        assert est.row is None
        assert est.ground_fraction == 1.0

    def test_steep_up_is_all_sky(self):
        est = _predict(-55.0)
        assert est.placement is FramePlacement.BELOW_FRAME
        assert est.row is None
        assert est.ground_fraction == 0.0

    def test_method_is_geometric(self):
        assert _predict(-30.0).method == "geometric"


class TestVerticalFov:
    def test_vfov_near_37_at_zoom_one(self):
        vfov = vertical_fov_degrees(
            Unitless(ZOOM),
            Pixels(IMAGE_WIDTH),
            Pixels(IMAGE_HEIGHT),
            sensor_width_mm=SENSOR_WIDTH_MM,
            base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
        )
        assert float(vfov) == pytest.approx(37.0, abs=4.0)

    def test_vfov_shrinks_with_zoom(self):
        wide = vertical_fov_degrees(
            Unitless(1.0),
            Pixels(IMAGE_WIDTH),
            Pixels(IMAGE_HEIGHT),
            sensor_width_mm=SENSOR_WIDTH_MM,
            base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
        )
        tele = vertical_fov_degrees(
            Unitless(5.0),
            Pixels(IMAGE_WIDTH),
            Pixels(IMAGE_HEIGHT),
            sensor_width_mm=SENSOR_WIDTH_MM,
            base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
        )
        assert float(tele) < float(wide)


class TestAllGroundThreshold:
    def test_threshold_near_minus_13_at_zoom_one(self):
        threshold = all_ground_tilt_threshold(
            Unitless(ZOOM),
            Pixels(IMAGE_WIDTH),
            Pixels(IMAGE_HEIGHT),
            sensor_width_mm=SENSOR_WIDTH_MM,
            base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
        )
        # Live table: horizon leaves the frame (100% ground) at tilt >= -13.
        assert float(threshold) == pytest.approx(-13.0, abs=2.0)

    def test_threshold_is_boundary_of_all_ground(self):
        threshold = float(
            all_ground_tilt_threshold(
                Unitless(ZOOM),
                Pixels(IMAGE_WIDTH),
                Pixels(IMAGE_HEIGHT),
                sensor_width_mm=SENSOR_WIDTH_MM,
                base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
            )
        )
        # Just above the threshold → all ground; just below → horizon in frame.
        assert _predict(threshold + 1.0).placement is FramePlacement.ABOVE_FRAME
        assert _predict(threshold - 2.0).placement is FramePlacement.IN_FRAME

    def test_threshold_rises_as_zoom_narrows_vfov(self):
        wide = float(
            all_ground_tilt_threshold(
                Unitless(1.0),
                Pixels(IMAGE_WIDTH),
                Pixels(IMAGE_HEIGHT),
                sensor_width_mm=SENSOR_WIDTH_MM,
                base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
            )
        )
        tele = float(
            all_ground_tilt_threshold(
                Unitless(5.0),
                Pixels(IMAGE_WIDTH),
                Pixels(IMAGE_HEIGHT),
                sensor_width_mm=SENSOR_WIDTH_MM,
                base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
            )
        )
        # threshold = offset + VFOV/2. A narrower VFOV (tele) shrinks the
        # +VFOV/2 term, so the threshold drops toward the offset (-31).
        assert tele < wide
