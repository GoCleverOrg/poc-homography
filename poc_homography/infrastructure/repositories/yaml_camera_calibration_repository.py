"""YAML-based CameraCalibration repository."""

from pathlib import Path

from poc_homography.domain.entities.camera_calibration import CameraCalibration
from poc_homography.infrastructure.repositories.base import YamlRepositoryBase


class YamlCameraCalibrationRepository(YamlRepositoryBase[CameraCalibration]):
    """Repository for CameraCalibration entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, CameraCalibration)
