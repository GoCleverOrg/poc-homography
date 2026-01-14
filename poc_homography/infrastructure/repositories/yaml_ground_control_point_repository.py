"""YAML-based GroundControlPoint repository implementation."""

from poc_homography.domain.entities.ground_control_point import GroundControlPoint
from poc_homography.infrastructure.repositories.base import YamlRepositoryBase


class YamlGroundControlPointRepository(YamlRepositoryBase[GroundControlPoint]):
    """Repository that loads GroundControlPoint entities from YAML files.

    Expected YAML format:
        name: Z1
        map_point:
          map_id: valte
          pixel_point:
            x: 1234.5
            y: 5678.9

    Files are stored with names like "valte__Z1.yaml" where
    the GCP ID "valte/Z1" has "/" replaced with "__".
    """

    def _get_entity_id(self, entity: GroundControlPoint) -> str:
        """Extract GCP ID from entity."""
        return entity.id

    def _entity_to_dict(self, entity: GroundControlPoint) -> dict:
        """Convert GroundControlPoint to YAML-serializable dictionary."""
        return entity.to_dict()

    def _dict_to_entity(self, data: dict) -> GroundControlPoint | None:
        """Reconstruct GroundControlPoint from YAML dictionary."""
        return GroundControlPoint.from_dict(data)

    def get_by_map(self, map_id: str) -> dict[str, GroundControlPoint]:
        """Retrieve all GCPs for a specific map.

        Args:
            map_id: Identifier for the map.

        Returns:
            Dictionary mapping GCP ID to GroundControlPoint entity.
            Empty dict if no GCPs exist for the map.
        """
        return self.get_by_prefix(f"{map_id}/")
