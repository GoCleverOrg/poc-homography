"""End-to-end orchestration tests for estimate_horizon."""

from __future__ import annotations

import cv2

from poc_homography.horizon import (
    FramePlacement,
    NullHorizonValidator,
    ValidationOutcome,
    estimate_horizon,
)
from poc_homography.types import Degrees, Pixels, Unitless

from .conftest import (
    BASE_FOCAL_LENGTH_MM,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SENSOR_WIDTH_MM,
    ZOOM,
)


def _estimate(tilt: float, **kwargs):
    return estimate_horizon(
        Degrees(tilt),
        Unitless(ZOOM),
        Pixels(IMAGE_WIDTH),
        Pixels(IMAGE_HEIGHT),
        sensor_width_mm=SENSOR_WIDTH_MM,
        base_focal_length_mm=BASE_FOCAL_LENGTH_MM,
        **kwargs,
    )


class TestEstimateHorizon:
    def test_geometry_only_without_image(self):
        est = _estimate(-30.0)
        assert est.placement is FramePlacement.IN_FRAME
        assert est.method == "geometric"

    def test_cv_refines_when_image_supplied(self, fixture_dir):
        image = cv2.imread(str(fixture_dir / "up_-30.jpg"))
        est = _estimate(-30.0, image=image)
        assert est.placement is FramePlacement.IN_FRAME
        assert est.method == "geometric+cv"

    def test_null_validator_leaves_estimate_intact(self, fixture_dir):
        image = cv2.imread(str(fixture_dir / "up_-30.jpg"))
        baseline = _estimate(-30.0, image=image)
        validated = _estimate(-30.0, image=image, validator=NullHorizonValidator())
        # Abstaining validator does not alter row or method.
        assert validated.row == baseline.row
        assert validated.method == baseline.method

    def test_contradicting_validator_lowers_confidence(self, fixture_dir):
        class _Disagree:
            def validate(self, image, estimate):
                return ValidationOutcome(agrees=False, confidence=0.9)

        image = cv2.imread(str(fixture_dir / "up_-30.jpg"))
        baseline = _estimate(-30.0, image=image)
        validated = _estimate(-30.0, image=image, validator=_Disagree())
        assert validated.confidence < baseline.confidence
        assert validated.method.endswith("+validated")
