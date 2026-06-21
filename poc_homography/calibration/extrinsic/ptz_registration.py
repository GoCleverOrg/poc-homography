"""Per-PTZ-state image↔orthophoto ground-plane homography registration.

For a single PTZ state this module composes the existing building blocks into a
homography that maps **frame (camera) pixels → orthophoto map-pixel** coords:

1. Fetch ``K(zoom)`` + distortion from a :class:`CameraCalibrationTable` at the
   commanded zoom.
2. Take the frame lines (detected via :meth:`LineDetector.detect`, or supplied
   directly) as the camera-side annotations.
3. Take the orthophoto lines' map-pixel endpoints as the ``line_registry``.
4. Feed both — together with a *seeded* frame-line ↔ ortho-line correspondence
   — to :meth:`MapPointHomography.compute_from_lines`, which undistorts the
   frame pixels internally and solves the homography via its ICP/RANSAC core.

This is the SEEDED leaf: the operator provides the correspondence explicitly as
a ``Mapping[int, str]`` from frame-line index to ortho line id. The fully
automatic frame↔ortho matcher is a separate future issue and is NOT built here.

Ortho line ids: :class:`OrthoLine` has no intrinsic id, so the orthophoto lines
are keyed positionally as ``f"ortho_{i}"`` for the *i*-th entry of the
``ortho_lines`` sequence passed in. The seed maps each frame-line index to one
of those ``ortho_{i}`` ids.

The commanded pan/tilt pose-prior path (:class:`IntrinsicExtrinsicHomography`
in ``poc_homography.homography.intrinsic_extrinsic``) is an *alternative* seed
source for the future automatic matcher; it is intentionally not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from poc_homography.calibration.lens_distortion.line_detection import LineDetector
from poc_homography.homography.map_points import MapPointHomography

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    import numpy.typing as npt

    from poc_homography.calibration.lens_distortion.calibration_table import (
        CameraCalibrationTable,
    )
    from poc_homography.calibration.lens_distortion.line_detection import CandidateLine
    from poc_homography.domain.vo.ortho_line import OrthoLine

# Minimum seeded correspondences: compute_from_lines needs >= 2 lines (4 points).
_MIN_SEEDED_LINES = 2


@dataclass(frozen=True)
class PtzRegistrationResult:
    """Outcome of registering one PTZ state's frame to the orthophoto.

    Wraps the metrics returned by
    :meth:`MapPointHomography.compute_from_lines` together with the PTZ
    provenance the registration was run for.

    Attributes:
        homography_matrix: 3x3 matrix mapping frame pixels → ortho map-pixels.
        inverse_matrix: 3x3 inverse mapping ortho map-pixels → frame pixels.
        num_lines: Number of seeded line correspondences used.
        num_inliers: Inlier point-to-line correspondences after RANSAC/ICP.
        inlier_ratio: Ratio of inliers to total correspondences in [0.0, 1.0].
        mean_perp_error: Mean perpendicular error in map pixels.
        max_perp_error: Maximum perpendicular error in map pixels.
        rmse: Root-mean-square of perpendicular errors in map pixels.
        run_id: Survey run this PTZ state came from (None if not applicable).
        camera_id: Camera this PTZ state came from (None if not applicable).
        commanded_zoom: Zoom the camera was commanded to for this frame.
        commanded_tilt: Tilt the camera was commanded to for this frame.
    """

    homography_matrix: npt.NDArray[np.float64]
    inverse_matrix: npt.NDArray[np.float64]
    num_lines: int
    num_inliers: int
    inlier_ratio: float
    mean_perp_error: float
    max_perp_error: float
    rmse: float
    run_id: str | None
    camera_id: str | None
    commanded_zoom: float
    commanded_tilt: float


def _intrinsic_kwargs(
    calibration_table: CameraCalibrationTable,
    commanded_zoom: float,
) -> dict[str, float]:
    """Build the K+distortion kwargs for :class:`MapPointHomography`.

    Distortion is wired through ONLY when intrinsics (fx, fy, cx, cy) are also
    available at this zoom, because the constructor requires all nine
    parameters together or none. When intrinsics are absent, an empty dict is
    returned so the solver runs without undistortion.

    Args:
        calibration_table: Per-zoom camera calibration table.
        commanded_zoom: Zoom to look the coefficients up at.

    Returns:
        Either ``{k1, k2, k3, p1, p2, fx, fy, cx, cy}`` (all nine) or ``{}``.
    """
    intrinsics = calibration_table.get_intrinsics(commanded_zoom)
    if intrinsics is None:
        return {}

    distortion = calibration_table.get_coefficients(commanded_zoom)
    return {
        "k1": float(distortion.k1),
        "k2": float(distortion.k2),
        "k3": float(distortion.k3),
        "p1": float(distortion.p1),
        "p2": float(distortion.p2),
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
    }


def _build_line_inputs(
    frame_lines: Sequence[CandidateLine],
    ortho_lines: Sequence[OrthoLine],
    seed: Mapping[int, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Translate the seed into solver line annotations + a line registry.

    For each ``frame_index -> ortho_id`` entry in the seed, emit one camera-side
    line annotation (distorted frame pixels) and register the matching ortho
    line's map-pixel endpoints under ``ortho_id``.

    Args:
        frame_lines: Detected/supplied camera frame lines.
        ortho_lines: Orthophoto lines, keyed positionally as ``ortho_{i}``.
        seed: Operator hint, frame-line index → ortho line id.

    Returns:
        ``(line_annotations, line_registry)`` for ``compute_from_lines``.

    Raises:
        ValueError: If the seed is too small, or references an out-of-range
            frame index or an unknown ortho id.
    """
    if len(seed) < _MIN_SEEDED_LINES:
        raise ValueError(
            f"Need at least {_MIN_SEEDED_LINES} seeded correspondences "
            f"(compute_from_lines needs >= 2 lines), got {len(seed)}"
        )

    ortho_by_id = {f"ortho_{i}": line for i, line in enumerate(ortho_lines)}

    line_annotations: list[dict[str, Any]] = []
    line_registry: dict[str, dict[str, float]] = {}

    for frame_index, ortho_id in seed.items():
        if not 0 <= frame_index < len(frame_lines):
            raise ValueError(
                f"Seed frame-line index {frame_index} out of range "
                f"for {len(frame_lines)} frame lines"
            )
        if ortho_id not in ortho_by_id:
            raise ValueError(
                f"Seed references unknown ortho id {ortho_id!r}; "
                f"valid ids are ortho_0..ortho_{len(ortho_lines) - 1}"
            )

        frame_line = frame_lines[frame_index]
        ortho_line = ortho_by_id[ortho_id]

        line_annotations.append(
            {
                "line_id": ortho_id,
                "start_pixel_x": float(frame_line.start[0]),
                "start_pixel_y": float(frame_line.start[1]),
                "end_pixel_x": float(frame_line.end[0]),
                "end_pixel_y": float(frame_line.end[1]),
            }
        )
        line_registry[ortho_id] = {
            "start_x": float(ortho_line.start_pixel[0]),
            "start_y": float(ortho_line.start_pixel[1]),
            "end_x": float(ortho_line.end_pixel[0]),
            "end_y": float(ortho_line.end_pixel[1]),
        }

    return line_annotations, line_registry


def register_frame_lines(
    *,
    frame_lines: Sequence[CandidateLine],
    ortho_lines: Sequence[OrthoLine],
    seed: Mapping[int, str],
    calibration_table: CameraCalibrationTable,
    commanded_zoom: float,
    commanded_tilt: float,
    map_id: str,
    run_id: str | None = None,
    camera_id: str | None = None,
    ransac_threshold: float = 50.0,
    min_inlier_ratio: float = 0.5,
) -> PtzRegistrationResult:
    """Register supplied frame lines to ortho lines via a seeded correspondence.

    Pure core of the pipeline: builds line annotations + line registry from the
    seed, fetches ``K(zoom)`` + distortion from ``calibration_table`` at
    ``commanded_zoom``, constructs a :class:`MapPointHomography` (with distortion
    only when intrinsics are present), runs ``compute_from_lines``, and wraps the
    result with PTZ provenance.

    Args:
        frame_lines: Camera frame lines in DISTORTED pixel space.
        ortho_lines: Orthophoto lines, keyed positionally as ``ortho_{i}``.
        seed: Operator hint, frame-line index → ortho line id (``ortho_{i}``).
        calibration_table: Per-zoom camera calibration table.
        commanded_zoom: Zoom the frame was captured at (intrinsics lookup key).
        commanded_tilt: Tilt the frame was captured at (provenance only).
        map_id: Orthophoto map identifier for the homography provider.
        run_id: Survey run for provenance (optional).
        camera_id: Camera id for provenance (optional).
        ransac_threshold: Max reprojection error (map pixels) for inliers.
        min_inlier_ratio: Minimum inlier ratio considered a valid fit.

    Returns:
        :class:`PtzRegistrationResult` with the frame→ortho homography + metrics.

    Raises:
        ValueError: If the seed is invalid or the fit quality is too poor.
        RuntimeError: If the underlying homography computation fails.
    """
    line_annotations, line_registry = _build_line_inputs(frame_lines, ortho_lines, seed)

    provider = MapPointHomography(map_id, **_intrinsic_kwargs(calibration_table, commanded_zoom))
    result = provider.compute_from_lines(
        line_annotations,
        line_registry,
        ransac_threshold=ransac_threshold,
        min_inlier_ratio=min_inlier_ratio,
    )

    return PtzRegistrationResult(
        homography_matrix=result.homography_matrix,
        inverse_matrix=result.inverse_matrix,
        num_lines=result.num_lines,
        num_inliers=result.num_inliers,
        inlier_ratio=result.inlier_ratio,
        mean_perp_error=result.mean_perp_error,
        max_perp_error=result.max_perp_error,
        rmse=result.rmse,
        run_id=run_id,
        camera_id=camera_id,
        commanded_zoom=commanded_zoom,
        commanded_tilt=commanded_tilt,
    )


def register_ptz_state(
    *,
    frame: npt.NDArray[np.uint8],
    ortho_lines: Sequence[OrthoLine],
    seed: Mapping[int, str],
    calibration_table: CameraCalibrationTable,
    commanded_zoom: float,
    commanded_tilt: float,
    map_id: str,
    run_id: str | None = None,
    camera_id: str | None = None,
    line_detector: LineDetector | None = None,
    ransac_threshold: float = 50.0,
    min_inlier_ratio: float = 0.5,
) -> PtzRegistrationResult:
    """Detect frame lines then register them to the orthophoto.

    Runs :meth:`LineDetector.detect` on ``frame`` and delegates to
    :func:`register_frame_lines`. The seed indexes into the DETECTED line list
    (lines sorted by confidence, highest first — see :class:`LineDetector`).

    Args:
        frame: BGR (or grayscale) camera image to detect lines in.
        ortho_lines: Orthophoto lines, keyed positionally as ``ortho_{i}``.
        seed: Operator hint, DETECTED-line index → ortho line id.
        calibration_table: Per-zoom camera calibration table.
        commanded_zoom: Zoom the frame was captured at (intrinsics lookup key).
        commanded_tilt: Tilt the frame was captured at (provenance only).
        map_id: Orthophoto map identifier for the homography provider.
        run_id: Survey run for provenance (optional).
        camera_id: Camera id for provenance (optional).
        line_detector: Detector to use; a default :class:`LineDetector` is
            constructed when None.
        ransac_threshold: Max reprojection error (map pixels) for inliers.
        min_inlier_ratio: Minimum inlier ratio considered a valid fit.

    Returns:
        :class:`PtzRegistrationResult` with the frame→ortho homography + metrics.

    Raises:
        ValueError: If the seed is invalid or the fit quality is too poor.
        RuntimeError: If the underlying homography computation fails.
    """
    detector = line_detector or LineDetector()
    frame_lines = detector.detect(frame)

    return register_frame_lines(
        frame_lines=frame_lines,
        ortho_lines=ortho_lines,
        seed=seed,
        calibration_table=calibration_table,
        commanded_zoom=commanded_zoom,
        commanded_tilt=commanded_tilt,
        map_id=map_id,
        run_id=run_id,
        camera_id=camera_id,
        ransac_threshold=ransac_threshold,
        min_inlier_ratio=min_inlier_ratio,
    )


__all__ = [
    "PtzRegistrationResult",
    "register_frame_lines",
    "register_ptz_state",
]
