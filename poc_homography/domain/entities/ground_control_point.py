from dataclasses import dataclass

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
