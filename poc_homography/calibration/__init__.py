"""Camera calibration utilities.

This package provides tools for camera calibration including:
- GCP-based calibration (annotation, comprehensive, interactive, projection modules)
- Lens distortion calibration using parking spot lines (lens_distortion subpackage)
"""

from poc_homography.calibration.annotation import Annotation, CaptureContext
from poc_homography.calibration.comprehensive import (
    GCP,
    TARGET_ERROR_THRESHOLD_PX,
    CalibrationParams,
    compute_projection_error,
    print_results,
    run_calibration,
    undistort_point_simple,
)
from poc_homography.calibration.interactive import (
    CalibrationResults,
    CalibrationSession,
    ReferencePoint,
    run_batch_calibration,
    run_interactive_session,
)
from poc_homography.calibration.projection import (
    ProjectionAnalysisResult,
    analyze_projection_error,
)

# Lens distortion calibration exports
from poc_homography.calibration.lens_distortion import (
    CameraCalibrationTable,
    CameraLine,
    CandidateLine,
    DistortionSolver,
    GroundTruthLine,
    LineCorrespondence,
    LineDetectionConfig,
    LineDetector,
    SolverConfig,
    SolverResult,
    ZoomCalibrationEntry,
)

__all__ = [
    # GCP-based calibration
    "analyze_projection_error",
    "Annotation",
    "CalibrationParams",
    "CalibrationResults",
    "CalibrationSession",
    "CaptureContext",
    "compute_projection_error",
    "GCP",
    "print_results",
    "ProjectionAnalysisResult",
    "ReferencePoint",
    "run_batch_calibration",
    "run_calibration",
    "run_interactive_session",
    "TARGET_ERROR_THRESHOLD_PX",
    "undistort_point_simple",
    # Lens distortion calibration
    "CameraCalibrationTable",
    "CameraLine",
    "CandidateLine",
    "DistortionSolver",
    "GroundTruthLine",
    "LineCorrespondence",
    "LineDetectionConfig",
    "LineDetector",
    "SolverConfig",
    "SolverResult",
    "ZoomCalibrationEntry",
]
