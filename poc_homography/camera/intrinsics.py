"""
Camera intrinsics computation from PTZ status.

This module computes a camera's intrinsic matrix from a zoom level and sensor
parameters. Querying the live PTZ position is delegated to
:class:`poc_homography.infrastructure.clients.hikvision.isapi_client.HikvisionISAPIClient`;
this module no longer talks to camera hardware directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from poc_homography.types import Degrees, Millimeters, Pixels, PixelsFloat, Unitless

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PTZStatus:
    """Camera PTZ (Pan-Tilt-Zoom) status."""

    pan: Degrees
    """Pan angle in degrees (azimuth)"""

    tilt: Degrees
    """Tilt angle in degrees (elevation, positive = down for Hikvision)"""

    zoom: Unitless
    """Zoom factor (1.0 = no zoom)"""


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    focal_length_mm: Millimeters
    """Focal length in millimeters (zoom-adjusted)"""

    focal_length_px: PixelsFloat
    """Focal length in pixels"""

    cx: PixelsFloat
    """Principal point X coordinate (pixels)"""

    cy: PixelsFloat
    """Principal point Y coordinate (pixels)"""

    sensor_width_mm: Millimeters
    """Sensor width in millimeters"""

    base_focal_length_mm: Millimeters
    """Base focal length in millimeters (at 1x zoom)"""

    image_width: Pixels
    """Image width in pixels"""

    image_height: Pixels
    """Image height in pixels"""

    K: NDArray[np.float64]
    """Intrinsic matrix (3x3 numpy array)"""


def compute_intrinsics(
    zoom: Unitless,
    image_width: Pixels,
    image_height: Pixels,
    sensor_width_mm: Millimeters,
    base_focal_length_mm: Millimeters,
) -> CameraIntrinsics:
    """
    Compute camera intrinsics from zoom and sensor parameters.

    Args:
        zoom: Zoom factor (1.0 = no zoom)
        image_width: Image width in pixels
        image_height: Image height in pixels
        sensor_width_mm: Sensor width in millimeters
        base_focal_length_mm: Base focal length in millimeters (at 1x zoom)

    Returns:
        Camera intrinsic parameters including intrinsic matrix
    """
    # Compute focal length
    f_mm = Millimeters(base_focal_length_mm * zoom)
    f_px = PixelsFloat(f_mm * (image_width / sensor_width_mm))

    # Principal point (image center)
    cx = PixelsFloat(image_width / 2.0)
    cy = PixelsFloat(image_height / 2.0)

    # Intrinsic matrix
    K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])

    return CameraIntrinsics(
        focal_length_mm=f_mm,
        focal_length_px=f_px,
        cx=cx,
        cy=cy,
        sensor_width_mm=sensor_width_mm,
        base_focal_length_mm=base_focal_length_mm,
        image_width=image_width,
        image_height=image_height,
        K=K,
    )
