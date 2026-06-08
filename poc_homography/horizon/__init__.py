"""Horizon detection for PTZ survey planning.

Locates the horizon line (the vanishing line of the ground plane) in a PTZ
frame so the survey can avoid wasting capture on sky. Three cooperating paths:

* **Geometric** (:mod:`~poc_homography.horizon.geometry`) — the workhorse;
  predicts the horizon row analytically from tilt + intrinsics, no image needed.
* **Classical CV** (:mod:`~poc_homography.horizon.cv_refine`) — refines/validates
  the row from an image via per-row edge-energy collapse, plus a coarse
  JPEG-size sky proxy.
* **Optional vision hook** (:mod:`~poc_homography.horizon.validation`) — a
  dependency-injected interface for an LLM/vision check (default no-op).

:mod:`~poc_homography.horizon.calibration` fits the one-time tilt→elevation
mount offset from a few samples.
"""

from poc_homography.horizon.calibration import calibrate_tilt_offset
from poc_homography.horizon.cv_refine import (
    estimate_sky_fraction_from_jpeg_size,
    refine_horizon_cv,
)
from poc_homography.horizon.estimate import estimate_horizon
from poc_homography.horizon.geometry import (
    DEFAULT_TILT_OFFSET_DEG,
    all_ground_tilt_threshold,
    predict_horizon,
    vertical_fov_degrees,
)
from poc_homography.horizon.models import (
    HORIZON_SCHEMA_VERSION,
    CalibrationResult,
    FramePlacement,
    HorizonEstimate,
)
from poc_homography.horizon.validation import (
    HorizonValidator,
    NullHorizonValidator,
    ValidationOutcome,
)

__all__ = [
    "DEFAULT_TILT_OFFSET_DEG",
    "HORIZON_SCHEMA_VERSION",
    "CalibrationResult",
    "FramePlacement",
    "HorizonEstimate",
    "HorizonValidator",
    "NullHorizonValidator",
    "ValidationOutcome",
    "all_ground_tilt_threshold",
    "calibrate_tilt_offset",
    "estimate_horizon",
    "estimate_sky_fraction_from_jpeg_size",
    "predict_horizon",
    "refine_horizon_cv",
    "vertical_fov_degrees",
]
