"""Annotation and capture context for calibration data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_homography.domain.vo import PixelPoint, PTZState


@dataclass(frozen=True)
class Annotation:
    """An annotation links a Ground Control Point (GCP) to its observed pixel location in a camera image.

    Attributes:
        gcp_id: ID of the GCP in the GCP registry.
        pose_id: ID of the camera pose when this observation was captured.
        pixel: Pixel coordinates in camera image.
    """

    gcp_id: str
    camera_pose: PTZState
    pixel: PixelPoint
