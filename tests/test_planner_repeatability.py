"""Repeatability sequence tests for the survey planner."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.survey.planner import Pose, repeatability_sequences
from poc_homography.types import Degrees, Unitless

TARGET_PAN = 120.0
TARGET_TILT = 30.0
TARGET_ZOOM = 8.0
DELTAS = [(-5.0, 0.0), (5.0, 0.0), (0.0, -5.0), (0.0, 5.0)]


class TestRepeatabilitySequences:
    def test_n_subsequences_for_n_deltas(self):
        seqs = repeatability_sequences(TARGET_PAN, TARGET_TILT, TARGET_ZOOM, DELTAS)
        assert len(seqs) == len(DELTAS)

    def test_each_subsequence_ends_at_target(self):
        target = Pose(
            pan=Degrees(TARGET_PAN), tilt=Degrees(TARGET_TILT), zoom=Unitless(TARGET_ZOOM)
        )
        seqs = repeatability_sequences(TARGET_PAN, TARGET_TILT, TARGET_ZOOM, DELTAS)
        for seq in seqs:
            assert seq[-1] == target

    def test_approach_start_is_offset(self):
        seqs = repeatability_sequences(TARGET_PAN, TARGET_TILT, TARGET_ZOOM, DELTAS)
        for (pan_off, tilt_off), seq in zip(DELTAS, seqs):
            start = seq[0]
            assert start.pan == TARGET_PAN + pan_off
            assert start.tilt == TARGET_TILT + tilt_off
            assert start.zoom == TARGET_ZOOM

    def test_each_subsequence_has_two_poses(self):
        seqs = repeatability_sequences(TARGET_PAN, TARGET_TILT, TARGET_ZOOM, DELTAS)
        assert all(len(seq) == 2 for seq in seqs)
