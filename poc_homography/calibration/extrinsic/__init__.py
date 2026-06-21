"""Per-PTZ-state extrinsic registration (image ↔ orthophoto homography).

This package composes the already-existing lens-distortion calibration table,
frame line detector, orthophoto line detector, and the line-based homography
solver into a single per-PTZ-state registration pipeline.

The pipeline does NOT implement homography/RANSAC/ICP/distortion itself — it
delegates all of that to :class:`MapPointHomography.compute_from_lines`. The
seeded leaf takes an explicit operator correspondence; the automatic leaf
(:func:`match_and_register`) discovers the correspondence itself and feeds the
discovered seed to the seeded core.
"""

from poc_homography.calibration.extrinsic.auto_match import (
    AutoMatchResult,
    InsufficientCorrespondenceError,
    match_and_register,
)
from poc_homography.calibration.extrinsic.ptz_registration import (
    PtzRegistrationResult,
    register_frame_lines,
    register_ptz_state,
)

__all__ = [
    "AutoMatchResult",
    "InsufficientCorrespondenceError",
    "PtzRegistrationResult",
    "match_and_register",
    "register_frame_lines",
    "register_ptz_state",
]
