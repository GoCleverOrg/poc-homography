"""FOV grid tests for the survey planner."""

import os
import sys
from itertools import pairwise

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.survey.planner import (
    angular_step_degrees,
    fov_grid,
    horizontal_fov_degrees,
    vertical_fov_degrees,
)

SPEC = CameraSpec.HIKVISION_DS_2DF8425IX
PAN_RANGE = (0.0, 90.0)
TILT_RANGE = (-20.0, 20.0)
ZOOM_LEVELS = [2.0, 5.0]
OVERLAP = 0.2


class TestFovGrid:
    def test_non_empty(self):
        poses = fov_grid(SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP)
        assert poses

    def test_every_pose_within_range_at_every_zoom(self):
        poses = fov_grid(SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP)
        for p in poses:
            assert PAN_RANGE[0] - 1e-6 <= p.pan <= PAN_RANGE[1] + 1e-6
            assert TILT_RANGE[0] - 1e-6 <= p.tilt <= TILT_RANGE[1] + 1e-6

    def test_all_zoom_levels_present(self):
        poses = fov_grid(SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP)
        zooms = {p.zoom for p in poses}
        assert zooms == set(ZOOM_LEVELS)

    def test_gaps_within_one_step(self):
        poses = fov_grid(SPEC, PAN_RANGE, TILT_RANGE, ZOOM_LEVELS, OVERLAP)
        for zoom in ZOOM_LEVELS:
            pan_step = angular_step_degrees(horizontal_fov_degrees(SPEC, zoom), OVERLAP)
            tilt_step = angular_step_degrees(vertical_fov_degrees(SPEC, zoom), OVERLAP)
            zoom_poses = [p for p in poses if p.zoom == zoom]
            pans = sorted({p.pan for p in zoom_poses})
            tilts = sorted({p.tilt for p in zoom_poses})
            for a, b in pairwise(pans):
                assert (b - a) <= pan_step + 1e-6
            for a, b in pairwise(tilts):
                assert (b - a) <= tilt_step + 1e-6

    def test_endpoints_covered(self):
        poses = fov_grid(SPEC, PAN_RANGE, TILT_RANGE, [3.0], OVERLAP)
        pans = {p.pan for p in poses}
        tilts = {p.tilt for p in poses}
        assert min(pans) == PAN_RANGE[0]
        assert max(pans) == PAN_RANGE[1]
        assert min(tilts) == TILT_RANGE[0]
        assert max(tilts) == TILT_RANGE[1]
