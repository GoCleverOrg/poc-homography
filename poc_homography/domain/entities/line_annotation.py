"""LineAnnotation entity — a camera observation of a map line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo import PixelPoint, PTZState


@dataclass(frozen=True, eq=False)
class LineAnnotation:
    """A camera observation of a map line.

    Analogous to ``Annotation`` (GCP observation) but for lines.
    Lines are n-point polylines: ``points`` stores every traced pixel
    coordinate.  ``start_pixel`` and ``end_pixel`` are the first and
    last elements for quick access.

    Attributes:
        line_id: Which map line (e.g. "L4").
        frame_id: Which camera image.
        camera_pose: PTZ state during observation.
        start_pixel: First point of the polyline.
        end_pixel: Last point of the polyline.
        points: All traced pixel coordinates, ``[(x, y), ...]``.
            ``None`` for legacy 2-point-only annotations.
    """

    line_id: str
    frame_id: str
    camera_pose: PTZState
    start_pixel: PixelPoint
    end_pixel: PixelPoint
    points: tuple[PixelPoint, ...] | None = None

    @property
    def id(self) -> str:
        """Composite ID for Entity protocol: ``frame_id/line_id``."""
        return f"{self.frame_id}/{self.line_id}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {
            "line_id": self.line_id,
            "frame_id": self.frame_id,
            "camera_pose": self.camera_pose.to_dict(),
            "start_pixel": self.start_pixel.to_dict(),
            "end_pixel": self.end_pixel.to_dict(),
        }
        if self.points is not None:
            d["points"] = [[float(p.x), float(p.y)] for p in self.points]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineAnnotation:
        """Create LineAnnotation from dictionary."""
        from poc_homography.domain.vo import PixelPoint, PTZState

        raw_points = data.get("points")
        points: tuple[PixelPoint, ...] | None = None
        if raw_points and len(raw_points) >= 2:
            points = tuple(PixelPoint.create(float(p[0]), float(p[1])) for p in raw_points)
        elif raw_points and len(raw_points) == 1:
            import logging

            logging.getLogger(__name__).warning(
                "LineAnnotation has single-point polyline (line_id=%s, frame_id=%s) — dropped",
                data.get("line_id"),
                data.get("frame_id"),
            )

        return cls(
            line_id=data["line_id"],
            frame_id=data["frame_id"],
            camera_pose=PTZState.from_dict(data["camera_pose"]),
            start_pixel=PixelPoint.from_dict(data["start_pixel"]),
            end_pixel=PixelPoint.from_dict(data["end_pixel"]),
            points=points,
        )
