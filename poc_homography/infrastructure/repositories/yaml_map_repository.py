"""YAML-based Map repository implementation."""

from pathlib import Path

import yaml
from PIL import Image

from poc_homography.domain.entities.map import Map
from poc_homography.domain.vo.geotiff import GeoTiff
from poc_homography.domain.vo.photo import Photo
from poc_homography.types import Pixels


class YamlMapRepository:
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
        self._data_dir = Path(data_dir)
        self._base_photo_dir = (
            Path(base_photo_dir) if base_photo_dir else self._data_dir.parent.parent
        )
        self._cache: dict[str, Map] = {}

    def get(self, map_id: str) -> Map | None:
        """Retrieve a map by its ID.

        Args:
            map_id: Unique identifier for the map.

        Returns:
            The Map entity if found, None otherwise.
        """
        if map_id in self._cache:
            return self._cache[map_id]

        yaml_path = self._data_dir / f"{map_id}.yaml"
        if not yaml_path.exists():
            return None

        map_entity = self._load_map(yaml_path)
        if map_entity:
            self._cache[map_id] = map_entity
        return map_entity

    def get_all(self) -> list[Map]:
        """Retrieve all available maps.

        Returns:
            List of all Map entities.
        """
        maps = []
        for yaml_path in self._data_dir.glob("*.yaml"):
            map_id = yaml_path.stem
            map_entity = self.get(map_id)
            if map_entity:
                maps.append(map_entity)
        return maps

    def exists(self, map_id: str) -> bool:
        """Check if a map exists.

        Args:
            map_id: Unique identifier for the map.

        Returns:
            True if the map exists, False otherwise.
        """
        if map_id in self._cache:
            return True
        yaml_path = self._data_dir / f"{map_id}.yaml"
        return yaml_path.exists()

    def save(self, map_entity: Map) -> None:
        """Save a map (create or update).

        Args:
            map_entity: The Map entity to save.
        """
        yaml_path = self._data_dir / f"{map_entity.id}.yaml"

        # Convert photo path to relative if it's under base_photo_dir
        photo_path = map_entity.photo.path
        try:
            relative_path = photo_path.relative_to(self._base_photo_dir)
            photo_path_str = str(relative_path)
        except ValueError:
            # Path is not relative to base_photo_dir, use absolute
            photo_path_str = str(photo_path)

        data = {
            "id": map_entity.id,
            "photo": {
                "path": photo_path_str,
            },
            "geotiff": {
                "geotransform": list(map_entity.geotiff.geotransform),
                "crs": map_entity.geotiff.crs,
            },
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        # Update cache
        self._cache[map_entity.id] = map_entity

    def delete(self, map_id: str) -> bool:
        """Delete a map by its ID.

        Args:
            map_id: Unique identifier for the map.

        Returns:
            True if the map was deleted, False if it didn't exist.
        """
        yaml_path = self._data_dir / f"{map_id}.yaml"

        if not yaml_path.exists():
            return False

        yaml_path.unlink()

        # Remove from cache
        if map_id in self._cache:
            del self._cache[map_id]

        return True

    def _load_map(self, yaml_path: Path) -> Map | None:
        """Load a Map entity from a YAML file.

        Args:
            yaml_path: Path to the YAML file.

        Returns:
            The Map entity, or None if loading fails.
        """
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        map_id = data["id"]

        # Resolve photo path
        photo_path_str = data["photo"]["path"]
        photo_path = Path(photo_path_str)
        if not photo_path.is_absolute():
            photo_path = self._base_photo_dir / photo_path_str

        # Load photo dimensions
        with Image.open(photo_path) as img:
            width = Pixels(img.width)
            height = Pixels(img.height)

        photo = Photo(path=photo_path, width=width, height=height)

        # Parse geotiff
        gt_data = data["geotiff"]
        geotransform = tuple(gt_data["geotransform"])
        crs = gt_data["crs"]
        geotiff = GeoTiff(geotransform=geotransform, crs=crs)  # type: ignore[arg-type]

        return Map(id=map_id, photo=photo, geotiff=geotiff)
