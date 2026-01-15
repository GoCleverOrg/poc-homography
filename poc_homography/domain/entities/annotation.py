"""Annotation entity linking GCPs to pixel observations in captured frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo import PixelPoint, PTZState


@dataclass(frozen=True)
class Annotation:
    """An annotation links a Ground Control Point (GCP) to its observed pixel location in a camera image.

    Annotations are stored per-frame and reference GCPs from the GCP repository.
    The PTZ state is captured at the time of annotation (may differ from frame's PTZ
    if manual adjustments were made).

    Attributes:
        gcp_id: ID of the GCP in the GCP registry.
        frame_id: ID of the CapturedFrame this annotation belongs to.
        camera_pose: PTZ state when this observation was captured.
        pixel: Pixel coordinates in camera image where GCP appears.
    """

    gcp_id: str
    frame_id: str
    camera_pose: PTZState
    pixel: PixelPoint

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gcp_id": self.gcp_id,
            "frame_id": self.frame_id,
            "camera_pose": self.camera_pose.to_dict(),
            "pixel": self.pixel.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        """Create Annotation from dictionary."""
        from poc_homography.domain.vo import PixelPoint, PTZState

        return cls(
            gcp_id=data["gcp_id"],
            frame_id=data["frame_id"],
            camera_pose=PTZState.from_dict(data["camera_pose"]),
            pixel=PixelPoint.from_dict(data["pixel"]),
        )
