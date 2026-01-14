"""YAML-based Map repository implementation."""

from pathlib import Path

from poc_homography.domain.entities.map import Map
from poc_homography.infrastructure.repositories.base import YamlRepositoryBase


class YamlMapRepository(YamlRepositoryBase[Map]):
    """Repository that loads Map entities from YAML files.

    Expected YAML format:
        id: valte
        photo:
          path: valte.png
          width: 12000
          height: 8000
        geotiff:
          geotransform:
            origin_easting: 737575.05
            pixel_width: 0.15
            row_rotation: 0.0
            origin_northing: 4391595.45
            col_rotation: 0.0
            pixel_height: -0.15
          crs: "EPSG:25830"

    Files are stored as {map_id}.yaml in the data directory.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the repository.

        Args:
            data_dir: Directory containing map YAML files.
        """
        super().__init__(data_dir, Map, create_dir=False)
