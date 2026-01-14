"""YAML-based CameraCalibration repository implementation."""

from pathlib import Path

from poc_homography.domain.entities.camera_calibration import CameraCalibration
from poc_homography.infrastructure.repositories.base import YamlRepositoryBase


class YamlCameraCalibrationRepository(YamlRepositoryBase[CameraCalibration]):
    """Repository that loads CameraCalibration entities from YAML files.

    Expected YAML format:
        camera_id: valte/Valte
        position:
          x: 1234.5
          y: 5678.9
        height: 4.71
        base_orientation:
          yaw: 51.7
          pitch: -0.25
          roll: 0.0
        distortion:
          k1: -0.341052
          k2: 0.787571
          p1: 0.0
          p2: 0.0

    Files are stored with names like "valte__Valte.yaml" where
    the camera_id "valte/Valte" has "/" replaced with "__".
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the repository.

        Args:
            data_dir: Directory containing camera calibration YAML files.
        """
        super().__init__(data_dir, CameraCalibration)
