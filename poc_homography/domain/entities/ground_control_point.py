from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.domain.vo.map_point import MapPoint


@dataclass
class GroundControlPoint:
    """Ground Control Point (GCP) with map point ID and pixel coordinates."""

    name: str
    map_point: MapPoint

    @property
    def map_id(self) -> str:
        return self.map_point.map_id

    @property
    def id(self) -> str:
        return f"{self.map_id}/{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "map_point": self.map_point.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundControlPoint:
        """Create GroundControlPoint from dictionary."""
        return cls(
            name=data["name"],
            map_point=MapPoint.from_dict(data["map_point"]),
        )
