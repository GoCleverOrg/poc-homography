"""Concrete survey phases (C4): wire the C3 planner onto the C2 capture engine.

Each phase is a thin orchestration layer. It retrieves poses or sweep
parameters from the planner (:mod:`poc_homography.survey.planner`), drives the
capture engine (:class:`poc_homography.infrastructure.survey.capture_engine.SurveyCaptureEngine`)
to execute captures, and ensures every emitted C1 record carries the correct
:class:`~poc_homography.domain.enums.survey_phase.SurveyPhase` tag plus the
planner-derived survey context (region id / approach direction / sequence
index). Pose generation, movement mechanics, and persistence all live in C3,
C2, and C1 respectively; C4 only binds them.

The :func:`~poc_homography.survey.phases.runner.execute_survey` runner sequences
all nine phases for one camera and routes every record to a
:class:`~poc_homography.survey.phases.common.PhaseSink`.
"""

from __future__ import annotations

from poc_homography.survey.phases.common import (
    PhaseResult,
    PhaseSink,
    pose_to_commanded,
)
from poc_homography.survey.phases.executors import (
    run_cross_zoom,
    run_dense_nadir,
    run_inventory,
    run_jitter,
    run_main_survey,
    run_ptz_characterization,
    run_repeatability,
    run_validation,
    run_zoom_characterization,
)
from poc_homography.survey.phases.runner import SurveyPlan, execute_survey

__all__ = [
    "PhaseResult",
    "PhaseSink",
    "pose_to_commanded",
    "run_inventory",
    "run_ptz_characterization",
    "run_zoom_characterization",
    "run_dense_nadir",
    "run_main_survey",
    "run_cross_zoom",
    "run_repeatability",
    "run_jitter",
    "run_validation",
    "SurveyPlan",
    "execute_survey",
]
