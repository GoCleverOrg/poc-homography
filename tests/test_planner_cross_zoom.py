"""Cross-zoom grouping tests for the survey planner."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.survey.planner import cross_zoom

ANCHORS = [(10.0, 5.0), (200.0, -10.0), (350.0, 0.0)]
ZOOMS = [1.0, 5.0, 15.0]


class TestCrossZoom:
    def test_count(self):
        poses = cross_zoom(ANCHORS, ZOOMS)
        assert len(poses) == len(ANCHORS) * len(ZOOMS)

    def test_zoom_levels_contiguous_per_anchor(self):
        poses = cross_zoom(ANCHORS, ZOOMS)
        n = len(ZOOMS)
        for i in range(len(ANCHORS)):
            chunk = poses[i * n : (i + 1) * n]
            assert [p.zoom for p in chunk] == ZOOMS
            assert all(p.region_id == f"region_{i}" for p in chunk)

    def test_region_grouping_order(self):
        poses = cross_zoom(ANCHORS, ZOOMS)
        # region ids appear in contiguous order, no interleaving.
        region_sequence = [p.region_id for p in poses]
        expected = []
        for i in range(len(ANCHORS)):
            expected.extend([f"region_{i}"] * len(ZOOMS))
        assert region_sequence == expected

    def test_anchor_coordinates_preserved(self):
        poses = cross_zoom(ANCHORS, ZOOMS)
        n = len(ZOOMS)
        for i, (pan, tilt) in enumerate(ANCHORS):
            chunk = poses[i * n : (i + 1) * n]
            assert all(p.pan == pan and p.tilt == tilt for p in chunk)
