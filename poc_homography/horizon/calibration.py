"""One-time tilt→true-elevation calibration for the horizon predictor.

Given a handful of ``(reported_tilt, detected_horizon_row)`` samples this fits
the fixed mount offset (the reported tilt at which the optical axis is
horizontal) and, as a cross-check, the vertical field of view implied by the
samples. The fit inverts the same geometric model used by
:mod:`poc_homography.horizon.geometry`::

    row = c_y + f_y * tan(tilt_offset - reported_tilt)

with ``c_y = image_height / 2`` fixed and ``(tilt_offset, f_y)`` solved by
least squares. ``f_y`` then yields ``VFOV = 2*atan(image_height / (2*f_y))``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from poc_homography.camera.intrinsics import compute_intrinsics
from poc_homography.horizon.geometry import (
    DEFAULT_BASE_FOCAL_LENGTH_MM,
    DEFAULT_SENSOR_WIDTH_MM,
    DEFAULT_TILT_OFFSET_DEG,
)
from poc_homography.horizon.models import CalibrationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from poc_homography.types import Degrees, Millimeters, Pixels, Unitless


def calibrate_tilt_offset(
    samples: Sequence[tuple[float, float]],
    zoom: Unitless,
    image_width: Pixels,
    image_height: Pixels,
    sensor_width_mm: Millimeters = DEFAULT_SENSOR_WIDTH_MM,
    base_focal_length_mm: Millimeters = DEFAULT_BASE_FOCAL_LENGTH_MM,
    tilt_offset_guess_deg: Degrees = DEFAULT_TILT_OFFSET_DEG,
) -> CalibrationResult:
    """Fit the tilt→elevation mount offset (and verify VFOV) from samples.

    Args:
        samples: ``(reported_tilt_deg, detected_horizon_row_px)`` pairs. At
            least two non-degenerate tilts are required.
        zoom: Zoom factor the samples were captured at.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        sensor_width_mm: Sensor width (for the focal-length initial guess).
        base_focal_length_mm: Base focal length at 1× zoom.
        tilt_offset_guess_deg: Initial guess for the offset.

    Returns:
        A :class:`CalibrationResult` with the fitted ``tilt_offset_deg``,
        cross-checked ``vfov_deg``, and the RMS fraction residual.

    Raises:
        ValueError: If fewer than two samples are supplied.
    """
    if len(samples) < 2:
        raise ValueError(f"calibration needs >= 2 samples; got {len(samples)}")

    tilts = np.array([float(t) for t, _ in samples], dtype=np.float64)
    rows = np.array([float(r) for _, r in samples], dtype=np.float64)
    height = float(image_height)
    cy = height / 2.0

    intr = compute_intrinsics(
        zoom, image_width, image_height, sensor_width_mm, base_focal_length_mm
    )
    f_px_guess = float(intr.focal_length_px)

    def residuals(params: np.ndarray) -> np.ndarray:
        offset, f_px = params
        predicted = cy + f_px * np.tan(np.radians(offset - tilts))
        # Normalise to frame fractions so the RMS is resolution-independent.
        return (predicted - rows) / height

    solution = least_squares(
        residuals,
        x0=np.array([float(tilt_offset_guess_deg), f_px_guess]),
    )
    offset_fit, f_px_fit = (float(solution.x[0]), float(solution.x[1]))
    vfov_deg = math.degrees(2.0 * math.atan(height / (2.0 * abs(f_px_fit))))
    rms = float(np.sqrt(np.mean(residuals(solution.x) ** 2)))

    return CalibrationResult(
        tilt_offset_deg=offset_fit,
        vfov_deg=vfov_deg,
        zoom=float(zoom),
        rms_fraction_residual=rms,
        n_samples=len(samples),
    )
