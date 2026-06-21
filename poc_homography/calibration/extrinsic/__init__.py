"""Per-PTZ-state extrinsic registration (image ↔ orthophoto homography).

This package composes the already-existing lens-distortion calibration table,
frame line detector, orthophoto line detector, and the line-based homography
solver into a single per-PTZ-state registration pipeline.

The pipeline does NOT implement homography/RANSAC/ICP/distortion itself — it
delegates all of that to :class:`MapPointHomography.compute_from_lines`. The
frame-line ↔ ortho-line correspondence is *seeded* (explicit operator hints);
fully automatic matching is intentionally out of scope here.
"""

from poc_homography.calibration.extrinsic.ptz_registration import (
    PtzRegistrationResult,
    register_frame_lines,
    register_ptz_state,
)

__all__ = [
    "PtzRegistrationResult",
    "register_frame_lines",
    "register_ptz_state",
]
