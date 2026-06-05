"""Nadir region tests for the survey planner."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.survey.planner import nadir_region

SPEC = CameraSpec.HIKVISION_DS_2DF8425IX
NADIR_PAN = 45.0
NADIR_TILT = 80.0
RADIUS = 15.0
ZOOM = 4.0
OVERLAP = 0.25


class TestNadirRegion:
    def test_non_empty(self):
        poses = nadir_region(SPEC, NADIR_PAN, NADIR_TILT, RADIUS, ZOOM, OVERLAP)
        assert poses

    def test_every_pose_within_radius(self):
        poses = nadir_region(SPEC, NADIR_PAN, NADIR_TILT, RADIUS, ZOOM, OVERLAP)
        for p in poses:
            dist = math.sqrt((p.pan - NADIR_PAN) ** 2 + (p.tilt - NADIR_TILT) ** 2)
            assert dist <= RADIUS + 1e-6

    def test_all_at_given_zoom(self):
        poses = nadir_region(SPEC, NADIR_PAN, NADIR_TILT, RADIUS, ZOOM, OVERLAP)
        assert all(p.zoom == ZOOM for p in poses)

    def test_disc_not_square(self):
        # A corner pose at (+R, +R) lies sqrt(2)*R away and must be excluded.
        poses = nadir_region(SPEC, NADIR_PAN, NADIR_TILT, RADIUS, ZOOM, OVERLAP)
        for p in poses:
            assert not (
                abs(p.pan - (NADIR_PAN + RADIUS)) < 1e-6
                and abs(p.tilt - (NADIR_TILT + RADIUS)) < 1e-6
            )
