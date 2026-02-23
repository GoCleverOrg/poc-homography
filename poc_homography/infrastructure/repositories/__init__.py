"""Repository implementations for the infrastructure layer."""

from poc_homography.infrastructure.repositories.base import RepoYaml
from poc_homography.infrastructure.repositories.repo_yaml_annotation import (
    RepoYamlAnnotation,
)
from poc_homography.infrastructure.repositories.repo_yaml_calibration_line_trace_set import (
    RepoYamlCalibrationLineTraceSet,
)
from poc_homography.infrastructure.repositories.repo_yaml_camera_calibration import (
    RepoYamlCameraCalibration,
)
from poc_homography.infrastructure.repositories.repo_yaml_camera_config import (
    RepoYamlCameraConfig,
)
from poc_homography.infrastructure.repositories.repo_yaml_captured_frame import (
    RepoYamlCapturedFrame,
)

# Session repositories (date-partitioned storage, webapp models)
from poc_homography.infrastructure.repositories.repo_yaml_diagnostic_session import (
    RepoYamlDiagnosticSession,
)
from poc_homography.infrastructure.repositories.repo_yaml_ground_control_point import (
    RepoYamlGroundControlPoint,
)
from poc_homography.infrastructure.repositories.repo_yaml_lens_calibration_table import (
    RepoYamlLensCalibrationTable,
)
from poc_homography.infrastructure.repositories.repo_yaml_line import RepoYamlLine
from poc_homography.infrastructure.repositories.repo_yaml_line_annotation import (
    RepoYamlLineAnnotation,
)
from poc_homography.infrastructure.repositories.repo_yaml_map import RepoYamlMap
from poc_homography.infrastructure.repositories.repo_yaml_stress_test_session import (
    RepoYamlStressTestSession,
)
from poc_homography.infrastructure.repositories.repo_yaml_survey_session import (
    RepoYamlSurveySession,
)
from poc_homography.infrastructure.repositories.repo_yaml_tenant import RepoYamlTenant

__all__ = [
    # Base class
    "RepoYaml",
    # Concrete repositories
    "RepoYamlAnnotation",
    "RepoYamlCalibrationLineTraceSet",
    "RepoYamlCameraCalibration",
    "RepoYamlCameraConfig",
    "RepoYamlCapturedFrame",
    "RepoYamlDiagnosticSession",
    "RepoYamlGroundControlPoint",
    "RepoYamlLensCalibrationTable",
    "RepoYamlLine",
    "RepoYamlLineAnnotation",
    "RepoYamlMap",
    "RepoYamlTenant",
    "RepoYamlStressTestSession",
    "RepoYamlSurveySession",
]
