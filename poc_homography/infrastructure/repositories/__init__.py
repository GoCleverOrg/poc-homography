"""Repository implementations for the infrastructure layer."""

from poc_homography.infrastructure.repositories.base import RepoPostgres, RepoYaml

# Concrete repositories (Postgres) — domain entity repos
from poc_homography.infrastructure.repositories.repo_postgres_annotation import (
    RepoPostgresAnnotation,
)
from poc_homography.infrastructure.repositories.repo_postgres_calibration_line_trace_set import (
    RepoPostgresCalibrationLineTraceSet,
)
from poc_homography.infrastructure.repositories.repo_postgres_camera_calibration import (
    RepoPostgresCameraCalibration,
)
from poc_homography.infrastructure.repositories.repo_postgres_camera_config import (
    RepoPostgresCameraConfig,
)
from poc_homography.infrastructure.repositories.repo_postgres_captured_frame import (
    RepoPostgresCapturedFrame,
)

# Concrete repositories (Postgres) — session repos (webapp layer, untyped)
from poc_homography.infrastructure.repositories.repo_postgres_diagnostic_session import (
    RepoPostgresDiagnosticSession,
)
from poc_homography.infrastructure.repositories.repo_postgres_ground_control_point import (
    RepoPostgresGroundControlPoint,
)
from poc_homography.infrastructure.repositories.repo_postgres_lens_calibration_table import (
    RepoPostgresLensCalibrationTable,
)
from poc_homography.infrastructure.repositories.repo_postgres_line import (
    RepoPostgresLine,
)
from poc_homography.infrastructure.repositories.repo_postgres_line_annotation import (
    RepoPostgresLineAnnotation,
)
from poc_homography.infrastructure.repositories.repo_postgres_map import (
    RepoPostgresMap,
)
from poc_homography.infrastructure.repositories.repo_postgres_stress_test_session import (
    RepoPostgresStressTestSession,
)
from poc_homography.infrastructure.repositories.repo_postgres_survey_run import (
    RepoPostgresSurveyRun,
)
from poc_homography.infrastructure.repositories.repo_postgres_survey_session import (
    RepoPostgresSurveySession,
)
from poc_homography.infrastructure.repositories.repo_postgres_tenant import (
    RepoPostgresTenant,
)

# Concrete repositories (YAML)
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
from poc_homography.infrastructure.repositories.repo_yaml_survey_run import (
    RepoYamlSurveyRun,
)
from poc_homography.infrastructure.repositories.repo_yaml_survey_session import (
    RepoYamlSurveySession,
)
from poc_homography.infrastructure.repositories.repo_yaml_tenant import RepoYamlTenant

__all__ = [
    # Base classes
    "RepoPostgres",
    "RepoYaml",
    # Concrete repositories (Postgres) — domain entity repos
    "RepoPostgresAnnotation",
    "RepoPostgresCalibrationLineTraceSet",
    "RepoPostgresCameraCalibration",
    "RepoPostgresCameraConfig",
    "RepoPostgresCapturedFrame",
    "RepoPostgresGroundControlPoint",
    "RepoPostgresLensCalibrationTable",
    "RepoPostgresLine",
    "RepoPostgresLineAnnotation",
    "RepoPostgresMap",
    "RepoPostgresTenant",
    # Concrete repositories (Postgres) — session repos
    "RepoPostgresDiagnosticSession",
    "RepoPostgresStressTestSession",
    "RepoPostgresSurveyRun",
    "RepoPostgresSurveySession",
    # Concrete repositories (YAML)
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
    "RepoYamlSurveyRun",
    "RepoYamlSurveySession",
]
