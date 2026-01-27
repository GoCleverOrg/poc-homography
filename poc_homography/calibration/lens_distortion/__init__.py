"""Lens distortion calibration using parking spot lines as calibration targets.

This package provides tools for determining camera lens distortion coefficients
by using parking lot line markings as substitutes for traditional checkerboard
calibration patterns.

Key insight: In an undistorted image, straight lines in the real world appear straight.
In a distorted image, straight lines appear curved. By detecting parking spot lines
and measuring their curvature, we can solve for distortion coefficients that would
straighten them.

Modules:
    models: Data structures for lines, calibration entries, and results
    line_detection: Automatic parking line detection using Hough transforms
    distortion_solver: Optimization-based distortion coefficient solver
    calibration_table: Zoom-dependent calibration storage with interpolation
    survey_automation: MCP Playwright and API automation for Camera Survey Tool
"""

from poc_homography.calibration.lens_distortion.calibration_table import (
    CameraCalibrationTable,
    ZoomCalibrationEntry,
)
from poc_homography.calibration.lens_distortion.distortion_solver import (
    DistortionSolver,
    SolverConfig,
    SolverResult,
)
from poc_homography.calibration.lens_distortion.line_detection import (
    CandidateLine,
    LineDetectionConfig,
    LineDetector,
)
from poc_homography.calibration.lens_distortion.masked_line_detection import (
    MaskedLineDetectionConfig,
    MaskedLineDetectionResult,
    MaskedLineDetector,
)
from poc_homography.calibration.lens_distortion.models import (
    CameraLine,
    GroundTruthLine,
    LineCorrespondence,
)
from poc_homography.calibration.lens_distortion.sam3_masking import (
    SAM3Config,
    SAM3Result,
    SAM3Segmenter,
    apply_mask,
    dilate_mask,
)
from poc_homography.calibration.lens_distortion.survey_automation import (
    CALIBRATION_SURVEY_PRESETS,
    CameraInfo,
    SurveyAutomation,
    SurveyAutomationError,
    SurveyAxis,
    SurveyProgress,
    SurveySession,
    SurveyStatus,
    TenantInfo,
)

__all__ = [
    # Models
    "CameraLine",
    "CandidateLine",
    "GroundTruthLine",
    "LineCorrespondence",
    # Line detection
    "LineDetectionConfig",
    "LineDetector",
    # Distortion solver
    "DistortionSolver",
    "SolverConfig",
    "SolverResult",
    # Calibration table
    "CameraCalibrationTable",
    "ZoomCalibrationEntry",
    # Survey automation
    "SurveyAutomation",
    "SurveyAutomationError",
    "SurveyAxis",
    "SurveyStatus",
    "SurveyProgress",
    "SurveySession",
    "TenantInfo",
    "CameraInfo",
    "CALIBRATION_SURVEY_PRESETS",
    # SAM3 masking
    "SAM3Config",
    "SAM3Result",
    "SAM3Segmenter",
    "apply_mask",
    "dilate_mask",
    # Masked line detection
    "MaskedLineDetectionConfig",
    "MaskedLineDetectionResult",
    "MaskedLineDetector",
]
