"""YAML-based CameraCalibration repository implementation."""

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

    def _get_entity_id(self, entity: CameraCalibration) -> str:
        """Extract camera ID from calibration entity."""
        return entity.camera_id

    def _entity_to_dict(self, entity: CameraCalibration) -> dict:
        """Convert CameraCalibration to YAML-serializable dictionary."""
        return entity.to_dict()

    def _dict_to_entity(self, data: dict) -> CameraCalibration | None:
        """Reconstruct CameraCalibration from YAML dictionary."""
        return CameraCalibration.from_dict(data)
