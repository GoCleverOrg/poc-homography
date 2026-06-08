"""CV horizon-refinement tests on the committed calibration frames."""

from __future__ import annotations

import cv2
import pytest

from poc_homography.horizon import (
    FramePlacement,
    estimate_sky_fraction_from_jpeg_size,
    predict_horizon,
    refine_horizon_cv,
)
from poc_homography.types import Degrees, Pixels, Unitless

from .conftest import (
    ALL_GROUND_FIXTURES,
    BASE_FOCAL_LENGTH_MM,
    FIXTURE_TILTS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IN_FRAME_FIXTURES,
    SENSOR_WIDTH_MM,
    ZOOM,
)


def _load(fixture_dir, name):
    image = cv2.imread(str(fixture_dir / name))
    assert image is not None, f"failed to read fixture {name}"
    return image


def _geometric_fraction(tilt: float) -> float:
    est = predict_horizon(
        Degrees(tilt),
        Unitless(ZOOM),
        Pixels(IMAGE_WIDTH),
        Pixels(IMAGE_HEIGHT),
        sensor_width_mm=SENSOR_WIDTH_MM,
        base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
    )
    assert est.row is not None
    return est.row / IMAGE_HEIGHT


class TestCvRefinement:
    @pytest.mark.parametrize("name", IN_FRAME_FIXTURES)
    def test_cv_agrees_with_geometry(self, fixture_dir, name):
        est = refine_horizon_cv(_load(fixture_dir, name))
        assert est.placement is FramePlacement.IN_FRAME
        assert est.row is not None
        cv_fraction = est.row / IMAGE_HEIGHT
        geometric_fraction = _geometric_fraction(FIXTURE_TILTS[name])
        # CV edge-collapse lands within 10% of height of the geometric row.
        assert cv_fraction == pytest.approx(geometric_fraction, abs=0.10)

    @pytest.mark.parametrize("name", ALL_GROUND_FIXTURES)
    def test_all_ground_frames_not_reported_in_frame(self, fixture_dir, name):
        est = refine_horizon_cv(_load(fixture_dir, name))
        # No low-texture sky band ⇒ the detector must not invent a horizon.
        assert est.placement is FramePlacement.ABOVE_FRAME
        assert est.row is None

    def test_mostly_sky_frame_classified_as_sky(self, fixture_dir):
        est = refine_horizon_cv(_load(fixture_dir, "up_-48.jpg"))
        # 95% sky → near-zero texture everywhere → below-frame (all sky).
        assert est.placement is FramePlacement.BELOW_FRAME
        assert est.row is None

    def test_method_is_cv(self, fixture_dir):
        assert refine_horizon_cv(_load(fixture_dir, "up_-30.jpg")).method == "cv"


class TestJpegSizeProxy:
    def test_more_sky_yields_higher_estimate(self, fixture_dir):
        # up_-20 (least sky) → up_-48 (most sky): file size collapses, sky rises.
        order = ["up_-20.jpg", "up_-30.jpg", "up_-40.jpg", "up_-48.jpg"]
        estimates = [
            estimate_sky_fraction_from_jpeg_size(
                (fixture_dir / name).stat().st_size, IMAGE_WIDTH, IMAGE_HEIGHT
            )
            for name in order
        ]
        assert estimates == sorted(estimates), estimates
        assert estimates[0] < estimates[-1]

    def test_fraction_clamped_to_unit_interval(self):
        tiny = estimate_sky_fraction_from_jpeg_size(1_000, IMAGE_WIDTH, IMAGE_HEIGHT)
        huge = estimate_sky_fraction_from_jpeg_size(5_000_000, IMAGE_WIDTH, IMAGE_HEIGHT)
        assert tiny == 1.0
        assert huge == 0.0

    def test_invalid_anchors_raise(self):
        with pytest.raises(ValueError):
            estimate_sky_fraction_from_jpeg_size(
                100,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                full_ground_bytes_per_pixel=0.01,
                full_sky_bytes_per_pixel=0.05,
            )
