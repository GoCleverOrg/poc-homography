"""Geometric horizon predictor — the no-image workhorse.

The horizon line (the vanishing line of the ground plane) projects to an
analytic image row determined entirely by the optical-axis elevation and the
camera intrinsics::

    y_h = c_y + f_y * tan(elevation_true)

where ``elevation_true`` is the optical-axis elevation above horizontal
(positive = pointing up) and ``c_y``, ``f_y`` come from the intrinsic matrix
``K``. This is algebraically identical to the ``y_h = c_y - f_y*tan(e)``
form in the issue once ``e`` is measured downward from horizontal.

The reported PTZ tilt is related to the true elevation by a fixed mount
offset::

    elevation_true = tilt_offset - reported_tilt

For ``icozee-camptz-04`` the offset is ≈ ``-31°`` (reported tilt ``-31°``
points the optical axis at the horizon). The tilt convention is positive =
down, matching :class:`poc_homography.camera.intrinsics.PTZStatus`.

Intrinsics are always delegated to
:func:`poc_homography.camera.intrinsics.compute_intrinsics`; no focal-length or
pixel arithmetic is reimplemented here.
"""

from __future__ import annotations

import math

from poc_homography.camera.intrinsics import compute_intrinsics
from poc_homography.horizon.models import FramePlacement, HorizonEstimate
from poc_homography.types import Degrees, Millimeters, Pixels, Unitless

# Default optics for the Hikvision DS-2DF8425IX used on icozee-camptz-04.
DEFAULT_SENSOR_WIDTH_MM = Millimeters(6.78)
DEFAULT_BASE_FOCAL_LENGTH_MM = Millimeters(5.9)
DEFAULT_TILT_OFFSET_DEG = Degrees(-31.0)
"""Reported tilt at which the optical axis is horizontal (true elevation 0)."""


def vertical_fov_degrees(
    zoom: Unitless,
    image_width: Pixels,
    image_height: Pixels,
    sensor_width_mm: Millimeters = DEFAULT_SENSOR_WIDTH_MM,
    base_focal_length_mm: Millimeters = DEFAULT_BASE_FOCAL_LENGTH_MM,
) -> Degrees:
    """Vertical field of view in degrees at a zoom level.

    Reuses :func:`compute_intrinsics` for the focal length so the VFOV shrinks
    correctly as zoom increases.
    """
    intr = compute_intrinsics(
        zoom, image_width, image_height, sensor_width_mm, base_focal_length_mm
    )
    return Degrees(math.degrees(2.0 * math.atan(image_height / (2.0 * intr.focal_length_px))))


def predict_horizon(
    reported_tilt_deg: Degrees,
    zoom: Unitless,
    image_width: Pixels,
    image_height: Pixels,
    sensor_width_mm: Millimeters = DEFAULT_SENSOR_WIDTH_MM,
    base_focal_length_mm: Millimeters = DEFAULT_BASE_FOCAL_LENGTH_MM,
    tilt_offset_deg: Degrees = DEFAULT_TILT_OFFSET_DEG,
) -> HorizonEstimate:
    """Predict the horizon placement from PTZ tilt + intrinsics (no image).

    Args:
        reported_tilt_deg: PTZ-reported tilt (positive = down).
        zoom: Zoom factor (1.0 = wide).
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        sensor_width_mm: Sensor width (defaults to the DS-2DF8425IX).
        base_focal_length_mm: Base focal length at 1× zoom.
        tilt_offset_deg: Reported tilt mapping to true elevation 0.

    Returns:
        A :class:`HorizonEstimate` with ``method="geometric"``. ``placement`` is
        ``ABOVE_FRAME`` (all ground) when the horizon sits above the top edge,
        ``BELOW_FRAME`` (all sky) when below the bottom edge, else ``IN_FRAME``
        with a concrete ``row``.
    """
    intr = compute_intrinsics(
        zoom, image_width, image_height, sensor_width_mm, base_focal_length_mm
    )
    elevation_deg = float(tilt_offset_deg) - float(reported_tilt_deg)
    row = intr.cy + intr.focal_length_px * math.tan(math.radians(elevation_deg))

    height = int(image_height)
    if row < 0.0:
        # Horizon above the top edge → entire frame is below the horizon.
        return HorizonEstimate(
            placement=FramePlacement.ABOVE_FRAME,
            image_height=height,
            row=None,
            ground_fraction=1.0,
            method="geometric",
        )
    if row >= height:
        # Horizon below the bottom edge → entire frame is above the horizon.
        return HorizonEstimate(
            placement=FramePlacement.BELOW_FRAME,
            image_height=height,
            row=None,
            ground_fraction=0.0,
            method="geometric",
        )
    return HorizonEstimate(
        placement=FramePlacement.IN_FRAME,
        image_height=height,
        row=float(row),
        ground_fraction=(height - float(row)) / height,
        method="geometric",
    )


def all_ground_tilt_threshold(
    zoom: Unitless,
    image_width: Pixels,
    image_height: Pixels,
    sensor_width_mm: Millimeters = DEFAULT_SENSOR_WIDTH_MM,
    base_focal_length_mm: Millimeters = DEFAULT_BASE_FOCAL_LENGTH_MM,
    tilt_offset_deg: Degrees = DEFAULT_TILT_OFFSET_DEG,
) -> Degrees:
    """Maximum reported tilt for which the whole frame is below the horizon.

    For any ``reported_tilt >= threshold`` the horizon sits at or above the top
    edge, so the frame is 100% ground and capture there wastes no pixels on sky.

    The threshold is ``tilt_offset + VFOV(zoom)/2`` and rises toward the offset
    as zoom narrows the VFOV — it is expressed as a function of zoom, never a
    constant.
    """
    vfov = vertical_fov_degrees(
        zoom, image_width, image_height, sensor_width_mm, base_focal_length_mm
    )
    return Degrees(float(tilt_offset_deg) + float(vfov) / 2.0)
