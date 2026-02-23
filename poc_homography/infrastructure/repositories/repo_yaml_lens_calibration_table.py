"""YAML-based LensCalibrationTable repository."""

from pathlib import Path

from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.infrastructure.repositories.base import RepoYaml


class RepoYamlLensCalibrationTable(RepoYaml[LensCalibrationTable]):
    """Repository for LensCalibrationTable entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, LensCalibrationTable)
