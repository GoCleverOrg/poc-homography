"""Per-PTZ-state extrinsic registration (image ↔ orthophoto homography).

This package composes the already-existing lens-distortion calibration table,
frame line detector, orthophoto line detector, and the line-based homography
solver into a single per-PTZ-state registration pipeline.

The pipeline does NOT implement homography/RANSAC/ICP/distortion itself — it
delegates all of that to :class:`MapPointHomography.compute_from_lines`. The
seeded leaf takes an explicit operator correspondence; the automatic leaf
(:func:`match_and_register`) discovers the correspondence itself and feeds the
discovered seed to the seeded core. The validation leaf
(:func:`validate_with_holdout`, :func:`consolidate_ptz_frames`) adds hold-out
reprojection validation in ground meters and multi-frame consolidation on top.
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
from poc_homography.calibration.extrinsic.validation import (
    FrameCorrespondence,
    HoldoutValidationError,
    HoldoutValidationResult,
    MultiFrameConsolidationResult,
    ReprojectionStats,
    consolidate_ptz_frames,
    validate_with_holdout,
)

__all__ = [
    "AutoMatchResult",
    "FrameCorrespondence",
    "HoldoutValidationError",
    "HoldoutValidationResult",
    "InsufficientCorrespondenceError",
    "MultiFrameConsolidationResult",
    "PtzRegistrationResult",
    "ReprojectionStats",
    "consolidate_ptz_frames",
    "match_and_register",
    "register_frame_lines",
    "register_ptz_state",
    "validate_with_holdout",
]
