"""YAML-based Map repository implementation."""

from pathlib import Path

from PIL import Image

from poc_homography.domain.entities.map import Map
from poc_homography.infrastructure.repositories.base import YamlRepositoryBase
from poc_homography.types import Pixels


class YamlMapRepository(YamlRepositoryBase[Map]):
    """Repository that loads Map entities from YAML files.

    Expected YAML format:
        id: valte
        photo:
          path: Cartografia_valencia.tif
        geotiff:
          geotransform: [737575.05, 0.15, 0.0, 4391595.45, 0.0, -0.15]
          crs: "EPSG:25830"

    The photo path can be absolute or relative to a base directory.
    """

    def __init__(self, data_dir: Path, base_photo_dir: Path | None = None) -> None:
        """Initialize the repository.

        Args:
            data_dir: Directory containing map YAML files.
            base_photo_dir: Base directory for resolving relative photo paths.
                           If None, uses data_dir's parent.
        """
        super().__init__(data_dir, create_dir=False)
        self._base_photo_dir = (
            Path(base_photo_dir) if base_photo_dir else self._data_dir.parent.parent
        )

    def _id_to_path(self, entity_id: str) -> Path:
        """Convert map ID to filesystem path.

        Maps use simple IDs without separators, so no replacement needed.
        """
        return self._data_dir / f"{entity_id}.yaml"

    def _get_entity_id(self, entity: Map) -> str:
        """Extract map ID from entity."""
        return entity.id

    def _entity_to_dict(self, entity: Map) -> dict:
        """Convert Map to YAML-serializable dictionary."""
        return entity.to_dict(photo_base_path=self._base_photo_dir)

    def _dict_to_entity(self, data: dict) -> Map | None:
        """Reconstruct Map from YAML dictionary.

        Loads image dimensions from the actual file.
        """
        # Resolve photo path to load dimensions
        photo_path_str = data["photo"]["path"]
        photo_path = Path(photo_path_str)
        if not photo_path.is_absolute():
            photo_path = self._base_photo_dir / photo_path_str

        # Load dimensions from image file
        with Image.open(photo_path) as img:
            width = Pixels(img.width)
            height = Pixels(img.height)

        return Map.from_dict(
            data,
            photo_width=width,
            photo_height=height,
            photo_base_path=self._base_photo_dir,
        )
