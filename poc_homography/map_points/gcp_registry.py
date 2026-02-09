"""Registry for managing collections of GCPs."""

from __future__ import annotations

import json
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
    and providing serialization to/from JSON and YAML formats.

    Attributes:
        map_id: Identifier for the map these points belong to.
        points: Mapping from point ID to MapPoint objects.
    """

    map_id: str
    points: dict[str, MapPoint] = field(default_factory=dict, hash=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary for JSON serialization.

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

    def to_json(self, indent: int = 2) -> str:
        """Convert registry to JSON string.

        Args:
            indent: Number of spaces for JSON indentation (default: 2).

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

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
    def from_json(cls, json_str: str) -> GCPRegistry:
        """Create registry from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            New GCPRegistry instance.

        Raises:
            json.JSONDecodeError: If JSON is invalid.
            KeyError: If required keys are missing.
            ValueError: If data format is invalid.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

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
        """Save registry to file (JSON or YAML based on extension).

        Args:
            path: Path to output file (.json or .yaml/.yml).
            fs: File system implementation (default: DefaultFileSystem).
        """
        path = Path(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            _get_fs(fs).write_text(path, self.to_yaml())
        else:
            _get_fs(fs).write_text(path, self.to_json())

    @classmethod
    def load(cls, path: str | Path, fs: FileSystem | None = None) -> GCPRegistry:
        """Load registry from file (JSON or YAML based on extension).

        Args:
            path: Path to input file (.json or .yaml/.yml).
            fs: File system implementation (default: DefaultFileSystem).

        Returns:
            New GCPRegistry instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If JSON is invalid.
            yaml.YAMLError: If YAML is invalid.
            KeyError: If required keys are missing.
            ValueError: If data format is invalid.
        """
        path = Path(path)
        content = _get_fs(fs).read_text(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            return cls.from_yaml(content)
        return cls.from_json(content)
