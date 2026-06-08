"""Phase-0 horizon calibration → per-azimuth tilt envelope.

The first step of the horizon-aware 2-step survey. At a coarse set of pans
(e.g. every 30-45°) and a wide zoom, the camera is aimed near the predicted
horizon, a frame is captured, and :func:`poc_homography.horizon.estimate_horizon`
(geometric prediction → optional CV refine → optional vision confirm) locates
the ground/sky boundary. From each per-azimuth detection the *maximum-up useful
tilt* is derived — the most-up reported tilt at which the detected boundary
still sits at the top edge of frame. The result is a
:class:`~poc_homography.domain.vo.tilt_envelope.TiltEnvelope` the C3 pose
generators consume to skip sky tiles (step 2).

Geometry (and intrinsics) are delegated entirely to the horizon and intrinsics
modules; no focal-length or pixel arithmetic is reimplemented here.

Convention: **positive tilt = down** (see
:mod:`poc_homography.horizon.geometry`).
"""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from poc_homography.camera.intrinsics import compute_intrinsics
from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.domain.vo.tilt_envelope import TiltEnvelope
from poc_homography.horizon.estimate import estimate_horizon
from poc_homography.horizon.geometry import (
    DEFAULT_TILT_OFFSET_DEG,
    all_ground_tilt_threshold,
    vertical_fov_degrees,
)
from poc_homography.horizon.models import FramePlacement
from poc_homography.types import Degrees, Unitless

if TYPE_CHECKING:
    from collections.abc import Sequence

    from poc_homography.domain.protocols.camera_device import CameraDevice
    from poc_homography.horizon.models import HorizonEstimate
    from poc_homography.horizon.validation import HorizonValidator


def calibrate_horizon_envelope(
    camera: CameraDevice,
    *,
    pans: Sequence[float],
    zoom: float = 1.0,
    spec: CameraSpec = CameraSpec.HIKVISION_DS_2DF8425IX,
    validator: HorizonValidator | None = None,
    tilt_offset_deg: float = float(DEFAULT_TILT_OFFSET_DEG),
    settle_timeout_s: float = 5.0,
) -> TiltEnvelope:
    """Derive a per-azimuth tilt envelope from a coarse Phase-0 pan sweep.

    For each pan the camera is aimed at the predicted horizon (reported tilt
    ``tilt_offset_deg`` → true elevation 0, so the horizon sits near frame
    centre), allowed to settle, and a snapshot is captured and analysed with
    :func:`estimate_horizon`. The per-azimuth max-up useful tilt is then derived
    from the detected boundary (see :func:`_bound_from_estimate`).

    Args:
        camera: The PTZ camera device to drive.
        pans: Coarse pan angles in degrees (e.g. every 30-45°). At least one.
        zoom: Wide zoom factor for the calibration frames (1.0 = wide).
        spec: Camera specification supplying resolution and optics.
        validator: Optional injected vision validator passed to
            :func:`estimate_horizon`.
        tilt_offset_deg: Reported tilt mapping to true elevation 0 (the aim
            point and the geometric model's offset).
        settle_timeout_s: Stabilisation timeout after each move, in seconds.

    Returns:
        A :class:`TiltEnvelope` whose ``bounds`` map each pan (reduced modulo
        360) to its max-up useful tilt, with the confirmed ``tilt_offset_deg``
        and the ``vfov_deg`` at ``zoom``.

    Raises:
        ValueError: If ``pans`` is empty.
    """
    if not pans:
        raise ValueError("calibration needs >= 1 pan angle")

    width = spec.image_width
    height = spec.image_height
    aim_tilt = tilt_offset_deg

    bounds: dict[float, float] = {}
    for pan in pans:
        reported = camera.move_absolute(pan=pan, tilt=aim_tilt, zoom=zoom)
        camera.wait_for_stabilization(timeout_s=settle_timeout_s)
        reported_tilt = float(reported.tilt_deg)

        image = _decode_snapshot(camera.capture_snapshot())
        estimate = estimate_horizon(
            Degrees(reported_tilt),
            Unitless(zoom),
            width,
            height,
            image=image,
            validator=validator,
            sensor_width_mm=spec.sensor_width,
            base_focal_length_mm=spec.base_focal_length,
            tilt_offset_deg=Degrees(tilt_offset_deg),
        )
        bounds[pan % 360.0] = _bound_from_estimate(
            estimate, reported_tilt, zoom, spec, tilt_offset_deg
        )

    vfov = float(
        vertical_fov_degrees(
            Unitless(zoom),
            width,
            height,
            spec.sensor_width,
            spec.base_focal_length,
        )
    )
    return TiltEnvelope(
        bounds=bounds,
        tilt_offset_deg=tilt_offset_deg,
        vfov_deg=vfov,
        zoom=zoom,
    )


def _bound_from_estimate(
    estimate: HorizonEstimate,
    reported_tilt: float,
    zoom: float,
    spec: CameraSpec,
    tilt_offset_deg: float,
) -> float:
    """Derive the max-up useful tilt from one per-azimuth horizon estimate.

    With the geometric horizon row at the capture tilt being
    ``H(t0) = cy + f*tan(tilt_offset - t0)``, a detected boundary at row ``r``
    sits ``d = r - H(t0)`` pixels off the flat horizon (terrain/structures push
    the boundary above it, so ``d < 0``). The useful bound is the reported tilt
    that brings that same boundary to the top edge (row 0)::

        tilt_bound = tilt_offset + degrees(atan((cy + d) / f))

    which reduces to :func:`all_ground_tilt_threshold` (``tilt_offset + VFOV/2``)
    when ``d == 0``. When the boundary is not in frame (all-ground or all-sky
    detections give no row), the flat geometric threshold is used as a safe
    fallback.
    """
    if estimate.placement is not FramePlacement.IN_FRAME or estimate.row is None:
        return float(
            all_ground_tilt_threshold(
                Unitless(zoom),
                spec.image_width,
                spec.image_height,
                spec.sensor_width,
                spec.base_focal_length,
                Degrees(tilt_offset_deg),
            )
        )

    intr = compute_intrinsics(
        Unitless(zoom),
        spec.image_width,
        spec.image_height,
        spec.sensor_width,
        spec.base_focal_length,
    )
    f_px = float(intr.focal_length_px)
    cy = float(intr.cy)
    horizon_row = cy + f_px * math.tan(math.radians(tilt_offset_deg - reported_tilt))
    delta = float(estimate.row) - horizon_row
    return tilt_offset_deg + math.degrees(math.atan((cy + delta) / f_px))


def _decode_snapshot(data: bytes) -> np.ndarray:
    """Decode JPEG snapshot bytes to an RGB ``numpy`` array for CV refinement."""
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))
