"""FOV/angular-step tests for the survey planner."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.survey.planner import (
    angular_step_degrees,
    horizontal_fov_degrees,
    vertical_fov_degrees,
)

SPEC = CameraSpec.HIKVISION_DS_2DF8425IX


class TestAngularStep:
    def test_step_is_fov_times_one_minus_overlap(self):
        assert angular_step_degrees(60.0, 0.3) == pytest.approx(60.0 * 0.7)

    def test_zero_overlap_is_full_fov(self):
        assert angular_step_degrees(48.0, 0.0) == pytest.approx(48.0)

    def test_overlap_one_raises(self):
        with pytest.raises(ValueError):
            angular_step_degrees(60.0, 1.0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError):
            angular_step_degrees(60.0, -0.1)


class TestHorizontalFov:
    def test_horizontal_fov_at_zoom_one_matches_datasheet(self):
        # Datasheet says 59.8 deg at wide.
        fov = horizontal_fov_degrees(SPEC, 1.0)
        assert 59.0 <= fov <= 60.5

    def test_step_shrinks_as_zoom_increases(self):
        wide = horizontal_fov_degrees(SPEC, 1.0)
        tele = horizontal_fov_degrees(SPEC, 10.0)
        assert tele < wide
        step_wide = angular_step_degrees(wide, 0.2)
        step_tele = angular_step_degrees(tele, 0.2)
        assert step_tele < step_wide

    def test_vertical_smaller_than_horizontal(self):
        # 16:9 sensor -> vertical FOV is smaller.
        assert vertical_fov_degrees(SPEC, 1.0) < horizontal_fov_degrees(SPEC, 1.0)
