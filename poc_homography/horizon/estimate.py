"""Orchestration: geometric prediction → optional CV refine → optional validate.

A single entry point that runs the geometric workhorse first (always, no image
needed) and then optionally refines the row with the classical-CV detector and
cross-checks it with an injected vision validator. Geometry remains
authoritative: CV only overrides when it confidently agrees the horizon is in
frame, and a validator that *contradicts* the estimate only lowers confidence —
it never fabricates a row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.horizon.cv_refine import refine_horizon_cv
from poc_homography.horizon.geometry import (
    DEFAULT_BASE_FOCAL_LENGTH_MM,
    DEFAULT_SENSOR_WIDTH_MM,
    DEFAULT_TILT_OFFSET_DEG,
    predict_horizon,
)
from poc_homography.horizon.models import FramePlacement, HorizonEstimate

if TYPE_CHECKING:
    import numpy as np

    from poc_homography.horizon.validation import HorizonValidator
    from poc_homography.types import Degrees, Millimeters, Pixels, Unitless


def estimate_horizon(
    reported_tilt_deg: Degrees,
    zoom: Unitless,
    image_width: Pixels,
    image_height: Pixels,
    *,
    image: np.ndarray | None = None,
    validator: HorizonValidator | None = None,
    sensor_width_mm: Millimeters = DEFAULT_SENSOR_WIDTH_MM,
    base_focal_length_mm: Millimeters = DEFAULT_BASE_FOCAL_LENGTH_MM,
    tilt_offset_deg: Degrees = DEFAULT_TILT_OFFSET_DEG,
) -> HorizonEstimate:
    """Estimate the horizon, refining the geometric prediction when possible.

    Args:
        reported_tilt_deg: PTZ-reported tilt (positive = down).
        zoom: Zoom factor (1.0 = wide).
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        image: Optional BGR/grayscale frame enabling CV refinement + validation.
        validator: Optional injected vision validator (default: none).
        sensor_width_mm: Sensor width.
        base_focal_length_mm: Base focal length at 1× zoom.
        tilt_offset_deg: Reported tilt mapping to true elevation 0.

    Returns:
        A :class:`HorizonEstimate`. ``method`` is ``"geometric"``,
        ``"geometric+cv"``, or those suffixed with ``"+validated"`` when a
        validator weighed in.
    """
    estimate = predict_horizon(
        reported_tilt_deg,
        zoom,
        image_width,
        image_height,
        sensor_width_mm=sensor_width_mm,
        base_focal_length_mm=base_focal_length_mm,
        tilt_offset_deg=tilt_offset_deg,
    )

    if image is not None:
        cv_estimate = refine_horizon_cv(image)
        # CV only overrides when both paths agree the horizon is in frame.
        if (
            cv_estimate.placement is FramePlacement.IN_FRAME
            and estimate.placement is FramePlacement.IN_FRAME
            and cv_estimate.row is not None
        ):
            estimate = HorizonEstimate(
                placement=FramePlacement.IN_FRAME,
                image_height=estimate.image_height,
                row=cv_estimate.row,
                ground_fraction=cv_estimate.ground_fraction,
                method="geometric+cv",
                confidence=cv_estimate.confidence,
            )

    if validator is not None and image is not None:
        outcome = validator.validate(image, estimate)
        if outcome.agrees is not None:
            # Contradiction halves confidence; agreement leaves it intact.
            adjusted = estimate.confidence * (1.0 if outcome.agrees else 0.5)
            estimate = HorizonEstimate(
                placement=estimate.placement,
                image_height=estimate.image_height,
                row=estimate.row,
                ground_fraction=estimate.ground_fraction,
                method=f"{estimate.method}+validated",
                confidence=adjusted,
            )

    return estimate
