"""Lens distortion calibration using parking spot lines as calibration targets.

This package provides tools for determining camera lens distortion coefficients
by using parking lot line markings as substitutes for traditional checkerboard
calibration patterns.

Key insight: In an undistorted image, straight lines in the real world appear straight.
In a distorted image, straight lines appear curved. By detecting parking spot lines
and measuring their curvature, we can solve for distortion coefficients that would
straighten them.

Public API modules:
    models: Data structures for lines, calibration entries, and results
    line_detection: Automatic parking line detection using Hough transforms
    distortion_solver: Optimization-based distortion coefficient solver
    annotated_line_solver: Annotated-line distortion solver using N-point traces
    calibration_table: Zoom-dependent calibration storage with interpolation
    apply_calibration: Undistortion utilities and calibration file management
"""

from poc_homography.calibration.lens_distortion.annotated_line_solver import (
    AnnotatedLineSolver,
    AnnotatedLineSolverConfig,
    CameraLineAnnotation,
    build_camera_line_annotations,
    split_lines,
)
from poc_homography.calibration.lens_distortion.apply_calibration import (
    distort_points,
    line_straightness_error,
    measure_line_straightness,
    undistort_image,
    undistort_points,
)
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
from poc_homography.calibration.lens_distortion.models import (
    CameraLine,
    GroundTruthLine,
    LineCorrespondence,
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
    # Annotated-line solver
    "AnnotatedLineSolver",
    "AnnotatedLineSolverConfig",
    "CameraLineAnnotation",
    "build_camera_line_annotations",
    "split_lines",
    # Distortion & measurement utilities
    "distort_points",
    "line_straightness_error",
    "undistort_points",
    "undistort_image",
    "measure_line_straightness",
]
