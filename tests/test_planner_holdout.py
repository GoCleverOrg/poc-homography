"""Holdout partition tests for the survey planner."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poc_homography.survey.planner import Pose, partition_holdout
from poc_homography.types import Degrees, Unitless


def _make_poses(n: int) -> list[Pose]:
    return [Pose(pan=Degrees(float(i)), tilt=Degrees(0.0), zoom=Unitless(1.0)) for i in range(n)]


class TestPartitionHoldout:
    def test_holdout_size_is_floor(self):
        poses = _make_poses(10)
        train, holdout = partition_holdout(poses, 0.3, seed=42)
        assert len(holdout) == math.floor(10 * 0.3)
        assert len(train) == 10 - len(holdout)

    def test_disjoint_and_total(self):
        poses = _make_poses(20)
        train, holdout = partition_holdout(poses, 0.25, seed=7)
        train_pans = {p.pan for p in train}
        holdout_pans = {p.pan for p in holdout}
        assert train_pans.isdisjoint(holdout_pans)
        assert len(train) + len(holdout) == 20

    def test_holdout_tagged(self):
        poses = _make_poses(10)
        train, holdout = partition_holdout(poses, 0.4, seed=1)
        assert all(p.is_holdout for p in holdout)
        assert all(not p.is_holdout for p in train)

    def test_deterministic_same_seed(self):
        poses = _make_poses(50)
        t1, h1 = partition_holdout(poses, 0.2, seed=99)
        t2, h2 = partition_holdout(poses, 0.2, seed=99)
        assert [p.pan for p in h1] == [p.pan for p in h2]
        assert [p.pan for p in t1] == [p.pan for p in t2]

    def test_different_seed_differs(self):
        poses = _make_poses(50)
        _, h1 = partition_holdout(poses, 0.2, seed=1)
        _, h2 = partition_holdout(poses, 0.2, seed=2)
        assert {p.pan for p in h1} != {p.pan for p in h2}

    def test_invalid_fraction_raises(self):
        poses = _make_poses(5)
        with pytest.raises(ValueError):
            partition_holdout(poses, 1.5, seed=0)
        with pytest.raises(ValueError):
            partition_holdout(poses, -0.1, seed=0)
