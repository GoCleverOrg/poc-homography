"""Registry for managing collections of map points."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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


@dataclass(frozen=True)
class MapPointRegistry:
    """Immutable registry for managing map points.

    This class stores a collection of map points, allowing efficient lookup by ID
    and providing serialization to/from JSON format.

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
        """
        return {
            "map_id": self.map_id,
            "points": [point.to_dict() for point in self.points.values()],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert registry to JSON string.

        Args:
            indent: Number of spaces for JSON indentation (default: 2).

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path, fs: FileSystem | None = None) -> None:
        """Save registry to JSON file.

        Args:
            path: Path to output JSON file.
            fs: File system implementation (default: DefaultFileSystem).
        """
        if fs is None:
            fs = DefaultFileSystem()
        content = self.to_json()
        fs.write_text(path, content)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapPointRegistry:
        """Create registry from dictionary.

        Args:
            data: Dictionary with map_id and points array.

        Returns:
            New MapPointRegistry instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If data format is invalid.
        """
        map_id = str(data["map_id"])
        points_data = data.get("points", [])

        points: dict[str, MapPoint] = {}
        for point_data in points_data:
            point = MapPoint.from_dict(point_data)
            points[point.id] = point

        return cls(map_id=map_id, points=points)

    @classmethod
    def from_json(cls, json_str: str) -> MapPointRegistry:
        """Create registry from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            New MapPointRegistry instance.

        Raises:
            json.JSONDecodeError: If JSON is invalid.
            KeyError: If required keys are missing.
            ValueError: If data format is invalid.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path, fs: FileSystem | None = None) -> MapPointRegistry:
        """Load registry from JSON file.

        Args:
            path: Path to input JSON file.
            fs: File system implementation (default: DefaultFileSystem).

        Returns:
            New MapPointRegistry instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If JSON is invalid.
            KeyError: If required keys are missing.
            ValueError: If data format is invalid.
        """
        if fs is None:
            fs = DefaultFileSystem()
        content = fs.read_text(path)
        return cls.from_json(content)
