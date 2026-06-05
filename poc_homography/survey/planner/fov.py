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


def _focal_length_px(spec: CameraSpec, zoom: float) -> float:
    """Return the focal length in pixels at a *clamped* zoom level.

    Routes through :meth:`CameraSpec.focal_length_at_zoom` so the zoom is
    clamped to ``[1.0, max_zoom]`` (matching the optics model used elsewhere),
    then reuses :func:`compute_intrinsics` for the millimetre→pixel conversion.
    """
    # ``focal_length_at_zoom`` applies the [1.0, max_zoom] clamp; recover the
    # clamped zoom factor and feed it through the shared intrinsics routine.
    clamped_focal = spec.focal_length_at_zoom(zoom)
    clamped_zoom = clamped_focal / spec.base_focal_length
    intrinsics = compute_intrinsics(
        Unitless(clamped_zoom),
        spec.image_width,
        spec.image_height,
        spec.sensor_width,
        spec.base_focal_length,
    )
    return float(intrinsics.focal_length_px)


def horizontal_fov_degrees(spec: CameraSpec, zoom: float) -> Degrees:
    """Compute the horizontal field of view in degrees at a zoom level.

    Args:
        spec: Camera specification supplying sensor/optics dimensions.
        zoom: Zoom factor (clamped to ``[1.0, max_zoom]``; 1.0 = wide).

    Returns:
        Horizontal field of view in degrees.
    """
    f_px = _focal_length_px(spec, zoom)
    return Degrees(math.degrees(2.0 * math.atan(spec.image_width / (2.0 * f_px))))


def vertical_fov_degrees(spec: CameraSpec, zoom: float) -> Degrees:
    """Compute the vertical field of view in degrees at a zoom level.

    Args:
        spec: Camera specification supplying sensor/optics dimensions.
        zoom: Zoom factor (clamped to ``[1.0, max_zoom]``; 1.0 = wide).

    Returns:
        Vertical field of view in degrees.
    """
    f_px = _focal_length_px(spec, zoom)
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
