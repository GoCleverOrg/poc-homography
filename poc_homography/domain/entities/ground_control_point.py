from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.domain.vo.map_point import MapPoint


@dataclass(frozen=True, eq=False)
class GroundControlPoint:
    """Ground Control Point (GCP) - a reference point on a georeferenced map.

    Coordinates are stored as pixels on the map image.
    """

    name: str
    map_point: MapPoint

    @property
    def id(self) -> str:
        """Composite identity: ``{map_id}/{name}``."""
        return f"{self.map_point.map_id}/{self.name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def map_id(self) -> str:
        return self.map_point.map_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
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
