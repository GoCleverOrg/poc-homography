"""Multi-phase survey planner (FOV-based pose generation).

Public surface for the planner: pose value objects, phase plans, the six
pose-pattern generators, the run aggregate with resume support, the per-camera
lock registry, and the FOV helpers.
"""

from __future__ import annotations

from poc_homography.survey.planner.fov import (
    angular_step_degrees,
    horizontal_fov_degrees,
    vertical_fov_degrees,
)
from poc_homography.survey.planner.generators import (
    SweepAxis,
    cross_zoom,
    directional_sweep,
    fov_grid,
    nadir_region,
    partition_holdout,
    repeatability_sequences,
)
from poc_homography.survey.planner.locks import CameraLockRegistry
from poc_homography.survey.planner.phase_plan import PhasePlan
from poc_homography.survey.planner.poses import ApproachDirection, Pose
from poc_homography.survey.planner.run import PlannedSurveyRun, ResumeCursor

__all__ = [
    "Pose",
    "ApproachDirection",
    "PhasePlan",
    "SweepAxis",
    "directional_sweep",
    "fov_grid",
    "nadir_region",
    "cross_zoom",
    "repeatability_sequences",
    "partition_holdout",
    "PlannedSurveyRun",
    "ResumeCursor",
    "CameraLockRegistry",
    "horizontal_fov_degrees",
    "vertical_fov_degrees",
    "angular_step_degrees",
]
