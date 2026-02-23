"""Registry for managing collections of GCPs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from poc_homography.map_points.map_point import MapPoint


class FileSystem(Protocol):
    """Protocol for file system operations."""

    def read_text(self, path: str | Path) -> str:
        """Read text from a file."""
        ...

    def write_text(self, path: str | Path, content: str) -> None:
        """Write text to a file."""
        ...


class DefaultFileSystem:
    """Default file system implementation."""

    def read_text(self, path: str | Path) -> str:
        """Read text from a file."""
        return Path(path).read_text(encoding="utf-8")

    def write_text(self, path: str | Path, content: str) -> None:
        """Write text to a file."""
        Path(path).write_text(content, encoding="utf-8")


def _get_fs(fs: FileSystem | None) -> FileSystem:
    """Return the provided filesystem or the default."""
    return fs if fs is not None else DefaultFileSystem()


@dataclass(frozen=True)
class GCPRegistry:
    """Immutable registry for managing GCPs.

    This class stores a collection of GCPs, allowing efficient lookup by ID
    and providing serialization to/from YAML format.

    Attributes:
        map_id: Identifier for the map these points belong to.
        points: Mapping from point ID to MapPoint objects.
    """

    map_id: str
    points: dict[str, MapPoint] = field(default_factory=dict, hash=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary for serialization.

        Returns:
            Dictionary with map_id and points array.
            Each point dict includes an "id" key from the registry's dictionary key.
        """
        return {
            "map_id": self.map_id,
            "points": [
                {"id": point_id, **point.to_dict()} for point_id, point in self.points.items()
            ],
        }

    def __iter__(self):
        """Iterate over MapPoint values in the registry."""
        return iter(self.points.values())

    def __len__(self) -> int:
        """Return number of points in the registry."""
        return len(self.points)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GCPRegistry:
        """Create registry from dictionary.

        Args:
            data: Dictionary with map_id and points array.
                  Each point dict must have an "id" key which becomes the dictionary key.

        Returns:
            New GCPRegistry instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If data format is invalid.
        """
        map_id = str(data["map_id"])
        points_data = data.get("points", [])

        points: dict[str, MapPoint] = {}
        for point_data in points_data:
            # Extract id from the point data (external key)
            point_id = str(point_data["id"])
            # Create MapPoint without id (it's not a field anymore)
            point = MapPoint.from_dict(point_data)
            # Use the extracted id as the dictionary key
            points[point_id] = point

        return cls(map_id=map_id, points=points)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> GCPRegistry:
        """Create registry from YAML string.

        Args:
            yaml_str: YAML string representation.

        Returns:
            New GCPRegistry instance.

        Raises:
            yaml.YAMLError: If YAML is invalid.
            KeyError: If required keys are missing.
            ValueError: If data format is invalid or content is empty.
        """
        data = yaml.safe_load(yaml_str)
        if data is None:
            raise ValueError("YAML content is empty or contains only whitespace")
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        """Convert registry to YAML string.

        Returns:
            YAML string representation.
        """
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def save(self, path: str | Path, fs: FileSystem | None = None) -> None:
        """Save registry to YAML file.

        Args:
            path: Path to output file (.yaml or .yml).
            fs: File system implementation (default: DefaultFileSystem).
        """
        _get_fs(fs).write_text(Path(path), self.to_yaml())

    @classmethod
    def load(cls, path: str | Path, fs: FileSystem | None = None) -> GCPRegistry:
        """Load registry from YAML file.

        Args:
            path: Path to input file (.yaml or .yml).
            fs: File system implementation (default: DefaultFileSystem).

        Returns:
            New GCPRegistry instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            yaml.YAMLError: If YAML is invalid.
            KeyError: If required keys are missing.
            ValueError: If data format is invalid.
        """
        content = _get_fs(fs).read_text(Path(path))
        return cls.from_yaml(content)


# ---------------------------------------------------------------------------
# Repository adapter functions (bridge legacy GCPRegistry <-> DDD repos)
# ---------------------------------------------------------------------------


def from_gcp_repo(data_dir: Path, map_id: str) -> GCPRegistry:
    """Load a GCPRegistry from the DDD ``RepoYamlGroundControlPoint`` repository.

    Args:
        data_dir: Directory containing per-GCP YAML files.
        map_id: Map identifier to filter GCPs by.

    Returns:
        GCPRegistry populated with the legacy MapPoint representation.
    """
    from poc_homography.infrastructure.repositories import RepoYamlGroundControlPoint

    repo = RepoYamlGroundControlPoint(data_dir)
    all_gcps = repo.get_all()

    points: dict[str, MapPoint] = {}
    for gcp in all_gcps:
        if gcp.map_id != map_id:
            continue
        mp = gcp.map_point
        points[gcp.name] = MapPoint(
            pixel_x=float(mp.pixel_point.x),
            pixel_y=float(mp.pixel_point.y),
        )

    return GCPRegistry(map_id=map_id, points=points)


def save_to_gcp_repo(registry: GCPRegistry, data_dir: Path) -> None:
    """Save a GCPRegistry to the DDD ``RepoYamlGroundControlPoint`` repository.

    Each point in the registry is converted to a ``GroundControlPoint`` entity
    and persisted as an individual YAML file.

    Args:
        registry: The legacy registry to persist.
        data_dir: Directory for per-GCP YAML files.
    """
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint
    from poc_homography.domain.vo.map_point import MapPoint as DomainMapPoint
    from poc_homography.domain.vo.pixel_point import PixelPoint
    from poc_homography.infrastructure.repositories import RepoYamlGroundControlPoint

    repo = RepoYamlGroundControlPoint(data_dir)
    for name, point in registry.points.items():
        gcp = GroundControlPoint(
            name=name,
            map_point=DomainMapPoint(
                map_id=registry.map_id,
                pixel_point=PixelPoint.create(point.pixel_x, point.pixel_y),
            ),
        )
        repo.save(gcp)


def list_map_ids(data_dir: Path) -> list[str]:
    """Return sorted unique map IDs found in the GCP repository.

    Args:
        data_dir: Directory containing per-GCP YAML files.

    Returns:
        Sorted list of unique map_id strings.
    """
    from poc_homography.infrastructure.repositories import RepoYamlGroundControlPoint

    repo = RepoYamlGroundControlPoint(data_dir)
    all_gcps = repo.get_all()
    return sorted({gcp.map_id for gcp in all_gcps})
