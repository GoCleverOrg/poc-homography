"""SQLAlchemy 2.0 ORM models for all domain entities and infrastructure tables.

Import all models here so that ``Base.metadata`` sees every table when
``create_all()`` or Alembic ``autogenerate`` runs.
"""

from __future__ import annotations

from poc_homography.infrastructure.models.annotation import AnnotationModel
from poc_homography.infrastructure.models.calibration_line_trace_set import (
    CalibrationLineTraceSetModel,
)
from poc_homography.infrastructure.models.camera_calibration import CameraCalibrationModel
from poc_homography.infrastructure.models.camera_config import CameraConfigModel
from poc_homography.infrastructure.models.captured_frame import CapturedFrameModel
from poc_homography.infrastructure.models.diagnostic_session import DiagnosticSessionModel
from poc_homography.infrastructure.models.ground_control_point import GroundControlPointModel
from poc_homography.infrastructure.models.lens_calibration_table import LensCalibrationTableModel
from poc_homography.infrastructure.models.line import LineModel
from poc_homography.infrastructure.models.line_annotation import LineAnnotationModel
from poc_homography.infrastructure.models.map import MapModel
from poc_homography.infrastructure.models.stress_test_session import StressTestSessionModel
from poc_homography.infrastructure.models.survey_run import SurveyRunModel
from poc_homography.infrastructure.models.survey_session import SurveySessionModel
from poc_homography.infrastructure.models.tenant import TenantModel
from poc_homography.infrastructure.models.user import UserModel

__all__ = [
    "AnnotationModel",
    "CalibrationLineTraceSetModel",
    "CameraCalibrationModel",
    "CameraConfigModel",
    "CapturedFrameModel",
    "DiagnosticSessionModel",
    "GroundControlPointModel",
    "LensCalibrationTableModel",
    "LineAnnotationModel",
    "LineModel",
    "MapModel",
    "StressTestSessionModel",
    "SurveyRunModel",
    "SurveySessionModel",
    "TenantModel",
    "UserModel",
]
