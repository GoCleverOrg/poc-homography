"""Line entity — a line on a map image defined by two pixel endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo import PixelPoint


@dataclass
class Line:
    """A line on a map image, defined by two pixel endpoints.

    Lines are independent entities with their own coordinates, not tied to GCPs.
    Follows the same pattern as GroundControlPoint.

    Attributes:
        name: Line identifier (e.g. "L1").
        map_id: Map this line belongs to.
        start: Start endpoint in map pixels.
        end: End endpoint in map pixels.
        id: Auto-generated composite ID ``{map_id}/{name}``.
    """

    name: str
    map_id: str
    start: PixelPoint
    end: PixelPoint
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"{self.map_id}/{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "map_id": self.map_id,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Line:
        """Create Line from dictionary."""
        from poc_homography.domain.vo import PixelPoint

        return cls(
            id=data.get("id", ""),
            name=data["name"],
            map_id=data["map_id"],
            start=PixelPoint.from_dict(data["start"]),
            end=PixelPoint.from_dict(data["end"]),
        )
