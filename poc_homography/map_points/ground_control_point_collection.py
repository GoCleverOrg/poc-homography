"""Registry for managing collections of map points."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

import yaml

from poc_homography.domain.vo.map_point import MapPoint
from poc_homography.domain.vo.pixel_point import PixelPoint


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
class GroundControlPointCollection:
    """Immutable registry for managing map points.

    This class stores a collection of map points, allowing efficient lookup by ID
    and providing serialization to/from YAML format.

    Attributes:
        map_id: Identifier for the map these points belong to.
        points: Mapping from point ID to MapPoint objects.
    """

    map_id: str
    points: dict[str, MapPoint] = field(default_factory=dict, hash=False)

    def __iter__(self) -> Iterator[tuple[str, MapPoint]]:
        """Iterate over GCPs as (gcp_id, map_point) tuples."""
        return iter(self.points.items())

    def __len__(self) -> int:
        """Return number of GCPs in registry."""
        return len(self.points)

    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary for serialization.

        Returns:
            Dictionary with map_id and points array.
            Each point dict includes an "id" key from the registry's dictionary key
            and pixel_x/pixel_y coordinates extracted from the MapPoint's pixel_point.
        """
        return {
            "map_id": self.map_id,
            "points": [
                {
                    "id": point_id,
                    "pixel_x": float(point.pixel_point.x),
                    "pixel_y": float(point.pixel_point.y),
                }
                for point_id, point in self.points.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundControlPointCollection:
        """Create registry from dictionary.

        Args:
            data: Dictionary with map_id and points array.
                  Each point dict must have an "id" key which becomes the dictionary key,
                  plus pixel_x and pixel_y for coordinates.

        Returns:
            New GroundControlPointCollection instance.

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
            # Create MapPoint with map_id and pixel coordinates
            point = MapPoint(
                map_id=map_id,
                pixel_point=PixelPoint(
                    _x=float(point_data["pixel_x"]),
                    _y=float(point_data["pixel_y"]),
                ),
            )
            # Use the extracted id as the dictionary key
            points[point_id] = point

        return cls(map_id=map_id, points=points)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> GroundControlPointCollection:
        """Create registry from YAML string.

        Args:
            yaml_str: YAML string representation.

        Returns:
            New GroundControlPointCollection instance.

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

        Raises:
            ValueError: If file extension is not .yaml or .yml.
        """
        path = Path(path)
        if path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError(f"Unsupported file extension: {path.suffix}. Use .yaml or .yml")
        _get_fs(fs).write_text(path, self.to_yaml())

    @classmethod
    def load(cls, path: str | Path, fs: FileSystem | None = None) -> GroundControlPointCollection:
        """Load registry from YAML file.

        Args:
            path: Path to input file (.yaml or .yml).
            fs: File system implementation (default: DefaultFileSystem).

        Returns:
            New GroundControlPointCollection instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            yaml.YAMLError: If YAML is invalid.
            KeyError: If required keys are missing.
            ValueError: If data format is invalid or file extension is not supported.
        """
        path = Path(path)
        if path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError(f"Unsupported file extension: {path.suffix}. Use .yaml or .yml")
        content = _get_fs(fs).read_text(path)
        return cls.from_yaml(content)
