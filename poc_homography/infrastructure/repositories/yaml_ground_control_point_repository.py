"""YAML-based GroundControlPoint repository implementation."""

from pathlib import Path

import yaml

from poc_homography.domain.entities.ground_control_point import GroundControlPoint
from poc_homography.domain.vo.map_point import MapPoint
from poc_homography.domain.vo.pixel_point import PixelPoint


class YamlGroundControlPointRepository:
    """Repository that loads GroundControlPoint entities from YAML files.

    Expected YAML format:
        map_id: valte
        points:
          - id: Z1
            pixel_x: 1234.5
            pixel_y: 5678.9
          - id: Z2
            pixel_x: 2345.6
            pixel_y: 6789.0

    Each map has its own YAML file named {map_id}.yaml in the data directory.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the repository.

        Args:
            data_dir: Directory containing GCP YAML files (one per map).
        """
        self._data_dir = Path(data_dir)
        self._cache: dict[str, dict[str, GroundControlPoint]] = {}

    def get_by_map(self, map_id: str) -> dict[str, GroundControlPoint]:
        """Retrieve all GCPs for a specific map.

        Args:
            map_id: Identifier for the map.

        Returns:
            Dictionary mapping GCP ID to GroundControlPoint entity.
            Empty dict if no GCPs exist for the map.
        """
        return self._get_gcps_for_map(map_id)

    def save(self, gcp: GroundControlPoint) -> None:
        """Save a GCP (create or update).

        Args:
            gcp: The GroundControlPoint to save.
        """
        map_id = gcp.map_id
        gcps = self._get_gcps_for_map(map_id)

        # Update or add (dict handles uniqueness)
        gcps[gcp.id] = gcp

        self._save_gcps_for_map(map_id, gcps)

    def delete(self, gcp_id: str) -> bool:
        """Delete a GCP by its ID.

        Args:
            gcp_id: Unique identifier for the GCP (format: "map_id/name").

        Returns:
            True if the GCP was deleted, False if it didn't exist.
        """
        if "/" not in gcp_id:
            return False

        map_id, _ = gcp_id.split("/", 1)
        gcps = self._get_gcps_for_map(map_id)

        if gcp_id in gcps:
            del gcps[gcp_id]
            self._save_gcps_for_map(map_id, gcps)
            return True

        return False

    def exists(self, gcp_id: str) -> bool:
        """Check if a GCP exists.

        Args:
            gcp_id: Unique identifier for the GCP (format: "map_id/name").

        Returns:
            True if the GCP exists, False otherwise.
        """
        if "/" not in gcp_id:
            return False

        map_id, _ = gcp_id.split("/", 1)
        gcps = self._get_gcps_for_map(map_id)
        return gcp_id in gcps

    def _get_gcps_for_map(self, map_id: str) -> dict[str, GroundControlPoint]:
        """Load GCPs for a map, using cache if available."""
        if map_id in self._cache:
            return self._cache[map_id]

        yaml_path = self._data_dir / f"{map_id}.yaml"
        if not yaml_path.exists():
            self._cache[map_id] = {}
            return {}

        gcps = self._load_gcps(yaml_path)
        self._cache[map_id] = gcps
        return gcps

    def _load_gcps(self, yaml_path: Path) -> dict[str, GroundControlPoint]:
        """Load GCPs from a YAML file."""
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return {}

        map_id = data.get("map_id", yaml_path.stem)
        points_data = data.get("points", [])

        gcps: dict[str, GroundControlPoint] = {}
        for point_data in points_data:
            name = str(point_data["id"])
            pixel_x = float(point_data["pixel_x"])
            pixel_y = float(point_data["pixel_y"])

            pixel_point = PixelPoint(_x=pixel_x, _y=pixel_y)
            map_point = MapPoint(map_id=map_id, pixel_point=pixel_point)
            gcp = GroundControlPoint(name=name, map_point=map_point)
            gcps[gcp.id] = gcp

        return gcps

    def _save_gcps_for_map(self, map_id: str, gcps: dict[str, GroundControlPoint]) -> None:
        """Save GCPs to a YAML file."""
        yaml_path = self._data_dir / f"{map_id}.yaml"

        points_data = []
        for gcp in gcps.values():
            points_data.append(
                {
                    "id": gcp.name,
                    "pixel_x": float(gcp.map_point.pixel_point._x),
                    "pixel_y": float(gcp.map_point.pixel_point._y),
                }
            )

        data = {
            "map_id": map_id,
            "points": points_data,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        # Update cache
        self._cache[map_id] = gcps
