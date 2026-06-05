"""Directional sweep tests for the survey planner."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.survey.planner import ApproachDirection, SweepAxis, directional_sweep


class TestDirectionalSweep:
    def test_returns_exactly_two_phase_plans(self):
        result = directional_sweep(
            SweepAxis.PAN, 0.0, 90.0, 30.0, 10.0, 1.0, phase=SurveyPhase.PTZ_CHARACTERIZATION
        )
        assert len(result) == 2

    def test_ascending_strictly_increasing(self):
        asc, _ = directional_sweep(
            SweepAxis.PAN, 0.0, 90.0, 30.0, 10.0, 1.0, phase=SurveyPhase.PTZ_CHARACTERIZATION
        )
        pans = [p.pan for p in asc.poses]
        assert all(pans[i] < pans[i + 1] for i in range(len(pans) - 1))

    def test_descending_strictly_decreasing(self):
        _, desc = directional_sweep(
            SweepAxis.PAN, 0.0, 90.0, 30.0, 10.0, 1.0, phase=SurveyPhase.PTZ_CHARACTERIZATION
        )
        pans = [p.pan for p in desc.poses]
        assert all(pans[i] > pans[i + 1] for i in range(len(pans) - 1))

    def test_approach_direction_tags(self):
        asc, desc = directional_sweep(
            SweepAxis.TILT, 0.0, 30.0, 10.0, 5.0, 2.0, phase=SurveyPhase.PTZ_CHARACTERIZATION
        )
        assert all(p.approach_direction is ApproachDirection.ASCENDING for p in asc.poses)
        assert all(p.approach_direction is ApproachDirection.DESCENDING for p in desc.poses)
        assert asc.approach_directions == (ApproachDirection.ASCENDING,)
        assert desc.approach_directions == (ApproachDirection.DESCENDING,)

    def test_endpoints_inclusive(self):
        asc, _ = directional_sweep(
            SweepAxis.PAN, 0.0, 100.0, 30.0, 10.0, 1.0, phase=SurveyPhase.PTZ_CHARACTERIZATION
        )
        assert asc.poses[0].pan == 0.0
        assert asc.poses[-1].pan == 100.0

    def test_zoom_ascending_is_wide_to_tele(self):
        asc, _ = directional_sweep(
            SweepAxis.ZOOM, 1.0, 25.0, 6.0, 0.0, 0.0, phase=SurveyPhase.ZOOM_CHARACTERIZATION
        )
        zooms = [p.zoom for p in asc.poses]
        assert zooms[0] < zooms[-1]
        assert zooms[0] == 1.0

    def test_fixed_values_placed_on_non_swept_axes(self):
        # axis=PAN -> fixed_a=tilt, fixed_b=zoom
        asc, _ = directional_sweep(
            SweepAxis.PAN, 0.0, 30.0, 30.0, 15.0, 3.0, phase=SurveyPhase.PTZ_CHARACTERIZATION
        )
        assert all(p.tilt == 15.0 for p in asc.poses)
        assert all(p.zoom == 3.0 for p in asc.poses)
