"""LineAnnotation entity — a camera observation of a map line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo import PixelPoint, PTZState


@dataclass(frozen=True)
class LineAnnotation:
    """A camera observation of a map line.

    Analogous to ``Annotation`` (GCP observation) but for lines.

    Attributes:
        line_id: Which map line (e.g. "L4").
        frame_id: Which camera image.
        camera_pose: PTZ state during observation.
        start_pixel: Observed start in camera image.
        end_pixel: Observed end in camera image.
    """

    line_id: str
    frame_id: str
    camera_pose: PTZState
    start_pixel: PixelPoint
    end_pixel: PixelPoint

    @property
    def id(self) -> str:
        """Composite ID for Entity protocol: ``frame_id/line_id``."""
        return f"{self.frame_id}/{self.line_id}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "line_id": self.line_id,
            "frame_id": self.frame_id,
            "camera_pose": self.camera_pose.to_dict(),
            "start_pixel": self.start_pixel.to_dict(),
            "end_pixel": self.end_pixel.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineAnnotation:
        """Create LineAnnotation from dictionary."""
        from poc_homography.domain.vo import PixelPoint, PTZState

        return cls(
            line_id=data["line_id"],
            frame_id=data["frame_id"],
            camera_pose=PTZState.from_dict(data["camera_pose"]),
            start_pixel=PixelPoint.from_dict(data["start_pixel"]),
            end_pixel=PixelPoint.from_dict(data["end_pixel"]),
        )
