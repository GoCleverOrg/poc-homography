"""YAML-based CalibrationLineTraceSet repository."""

from pathlib import Path

from poc_homography.domain.entities.calibration_line_trace_set import (
    CalibrationLineTraceSet,
)
from poc_homography.infrastructure.repositories.base import RepoYaml


class RepoYamlCalibrationLineTraceSet(RepoYaml[CalibrationLineTraceSet]):
    """Repository for CalibrationLineTraceSet entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, CalibrationLineTraceSet)
