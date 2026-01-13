"""Camera calibration utilities."""

from poc_homography.calibration.comprehensive import (
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
from poc_homography.domain import Annotation, GroundControlPoint, PTZState

__all__ = [
    "analyze_projection_error",
    "Annotation",
    "CalibrationParams",
    "CalibrationResults",
    "CalibrationSession",
    "PTZState",
    "compute_projection_error",
    "GroundControlPoint",
    "print_results",
    "ProjectionAnalysisResult",
    "ReferencePoint",
    "run_batch_calibration",
    "run_calibration",
    "run_interactive_session",
    "TARGET_ERROR_THRESHOLD_PX",
    "undistort_point_simple",
]
