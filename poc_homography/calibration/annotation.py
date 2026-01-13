"""Annotation and capture context for calibration data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.pixel_point import PixelPoint


@dataclass(frozen=True)
class Annotation:
    """An annotation links a Ground Control Point (GCP) to its observed pixel location in a camera image.

    Attributes:
        gcp_id: ID of the GCP in the GCP registry.
        pose_id: ID of the camera pose when this observation was captured.
        pixel: Pixel coordinates in camera image.
    """

    gcp_id: str
    pose_id: str
    pixel: PixelPoint

    def to_dict(self) -> dict[str, Any]:
        """Convert Annotation to a dictionary for JSON serialization.

        Returns:
            Dictionary with gcp_id, pose_id, and pixel (x, y) keys.
        """
        return {
            "gcp_id": self.gcp_id,
            "pose_id": self.pose_id,
            "pixel": {
                "x": self.pixel.x,
                "y": self.pixel.y,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        """Create Annotation from a dictionary.

        Args:
            data: Dictionary with gcp_id, pose_id, and pixel keys.
                  Note: pose_id defaults to empty string if not present
                  (for backward compatibility with older data files).

        Returns:
            New Annotation instance.

        Raises:
            KeyError: If required keys are missing from data.
            ValueError: If data types are invalid.
        """
        pixel_data = data["pixel"]
        return cls(
            gcp_id=str(data["gcp_id"]),
            pose_id=str(data.get("pose_id", "")),
            pixel=PixelPoint(
                x=float(pixel_data["x"]),
                y=float(pixel_data["y"]),
            ),
        )


@dataclass(frozen=True)
class CameraPose:
    """Camera state when a calibration frame was captured.

    Attributes:
        camera: Camera name (e.g., "Valte", "Setram").
        pan_raw: Raw pan position from PTZ API.
        tilt_deg: Tilt angle in degrees.
        zoom: Zoom level (1.0 = no zoom).
    """

    camera: str
    pan_raw: float
    tilt_deg: float
    zoom: float

    def to_dict(self) -> dict[str, Any]:
        """Convert CameraPose to a dictionary for JSON serialization.

        Returns:
            Dictionary with camera, pan_raw, tilt_deg, and zoom keys.
        """
        return {
            "camera": self.camera,
            "pan_raw": self.pan_raw,
            "tilt_deg": self.tilt_deg,
            "zoom": self.zoom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraPose:
        """Create CameraPose from a dictionary.

        Args:
            data: Dictionary with camera, pan_raw, tilt_deg, and zoom keys.

        Returns:
            New CameraPose instance.

        Raises:
            KeyError: If required keys are missing from data.
            ValueError: If data types are invalid.
        """
        return cls(
            camera=str(data["camera"]),
            pan_raw=float(data["pan_raw"]),
            tilt_deg=float(data["tilt_deg"]),
            zoom=float(data["zoom"]),
        )
