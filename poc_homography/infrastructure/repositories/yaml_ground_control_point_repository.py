"""YAML-based GroundControlPoint repository implementation."""

from pathlib import Path

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

    def __init__(self, data_dir: Path) -> None:
        """Initialize the repository.

        Args:
            data_dir: Directory containing GCP YAML files.
        """
        super().__init__(data_dir, GroundControlPoint)

    def get_by_map(self, map_id: str) -> dict[str, GroundControlPoint]:
        """Retrieve all GCPs for a specific map.

        Args:
            map_id: Identifier for the map.

        Returns:
            Dictionary mapping GCP ID to GroundControlPoint entity.
            Empty dict if no GCPs exist for the map.
        """
        return self.get_by_prefix(f"{map_id}/")
