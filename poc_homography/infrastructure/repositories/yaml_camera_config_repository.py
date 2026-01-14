"""YAML-based CameraConfig repository implementation."""

from poc_homography.domain.entities.camera_config import CameraConfig
from poc_homography.infrastructure.repositories.base import YamlRepositoryBase


class YamlCameraConfigRepository(YamlRepositoryBase[CameraConfig]):
    """Repository that loads CameraConfig entities from YAML files.

    Expected YAML format:
        map_id: valte
        name: Valte
        spec: HIKVISION_DS_2DF8425IX
        credential:
          username: admin
          password: password123
        ip_address: 10.207.99.178  # optional

    Files are stored with names like "valte__Valte.yaml" where
    the camera_id "valte/Valte" has "/" replaced with "__".
    """

    def _get_entity_id(self, entity: CameraConfig) -> str:
        """Extract camera ID from config entity."""
        return entity.id

    def _entity_to_dict(self, entity: CameraConfig) -> dict:
        """Convert CameraConfig to YAML-serializable dictionary."""
        return entity.to_dict()

    def _dict_to_entity(self, data: dict) -> CameraConfig | None:
        """Reconstruct CameraConfig from YAML dictionary."""
        return CameraConfig.from_dict(data)

    def get_by_map(self, map_id: str) -> dict[str, CameraConfig]:
        """Retrieve all camera configurations for a specific map.

        Args:
            map_id: ID of the map to get camera configs for.

        Returns:
            Dictionary mapping camera_id to CameraConfig for all cameras on the map.
        """
        configs = self.get_by_prefix(f"{map_id}/")
        # Filter to ensure map_id matches (extra safety check)
        return {k: v for k, v in configs.items() if v.map_id == map_id}
