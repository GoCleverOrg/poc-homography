"""Field-of-view helpers for FOV-based pose generation.

Pure functions that derive angular field of view from a :class:`CameraSpec`
by reusing :func:`poc_homography.camera.intrinsics.compute_intrinsics`. No
focal-length or pixel arithmetic is reimplemented here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from poc_homography.camera.intrinsics import compute_intrinsics
from poc_homography.types import Degrees, Unitless

if TYPE_CHECKING:
    from poc_homography.domain.enums.camera_spec import CameraSpec


def horizontal_fov_degrees(spec: CameraSpec, zoom: float) -> Degrees:
    """Compute the horizontal field of view in degrees at a zoom level.

    Args:
        spec: Camera specification supplying sensor/optics dimensions.
        zoom: Zoom factor (1.0 = wide).

    Returns:
        Horizontal field of view in degrees.
    """
    intrinsics = compute_intrinsics(
        Unitless(zoom),
        spec.image_width,
        spec.image_height,
        spec.sensor_width,
        spec.base_focal_length,
    )
    f_px = float(intrinsics.focal_length_px)
    return Degrees(math.degrees(2.0 * math.atan(spec.image_width / (2.0 * f_px))))


def vertical_fov_degrees(spec: CameraSpec, zoom: float) -> Degrees:
    """Compute the vertical field of view in degrees at a zoom level.

    Args:
        spec: Camera specification supplying sensor/optics dimensions.
        zoom: Zoom factor (1.0 = wide).

    Returns:
        Vertical field of view in degrees.
    """
    intrinsics = compute_intrinsics(
        Unitless(zoom),
        spec.image_width,
        spec.image_height,
        spec.sensor_width,
        spec.base_focal_length,
    )
    f_px = float(intrinsics.focal_length_px)
    return Degrees(math.degrees(2.0 * math.atan(spec.image_height / (2.0 * f_px))))


def angular_step_degrees(fov_deg: Degrees, overlap_fraction: float) -> Degrees:
    """Compute the angular step between adjacent poses for a target overlap.

    Args:
        fov_deg: Field of view in degrees along the swept axis.
        overlap_fraction: Desired fractional overlap between adjacent frames,
            in ``[0.0, 1.0)``.

    Returns:
        Angular step in degrees (``fov_deg * (1 - overlap_fraction)``).

    Raises:
        ValueError: If ``overlap_fraction`` is outside ``[0.0, 1.0)``.
    """
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError(f"overlap_fraction must be in [0.0, 1.0); got {overlap_fraction!r}")
    return Degrees(fov_deg * (1.0 - overlap_fraction))
