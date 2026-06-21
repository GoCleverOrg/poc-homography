"""Hold-out reprojection validation + multi-frame consolidation for one PTZ state.

This leaf adds two things on top of the seeded/auto registration leaves
(:mod:`poc_homography.calibration.extrinsic.ptz_registration`) WITHOUT building
a new homography estimator or a general bundle adjuster:

1. **Hold-out reprojection validation** (:func:`validate_with_holdout`).
   The matched frame-line ↔ ortho-line correspondences are split into a *fit*
   set and a *hold-out* set. The homography is solved on the fit set only (via
   the existing :func:`register_frame_lines`) and its reprojection error is
   measured on the held-out correspondences it never saw. Error is reported in
   **ortho ground units (meters)** — projected and true endpoints are both run
   through :meth:`GeoTiff.pixel_to_geo` and compared in easting/northing — not
   just in map pixels. A hold-out RMS above the caller's threshold raises the
   typed :class:`HoldoutValidationError`.

2. **Multi-frame consolidation** (:func:`consolidate_ptz_frames`).
   Several frames captured at the *same* PTZ state are consolidated into ONE
   registration by *pooling* every frame's line correspondences into a single
   :meth:`MapPointHomography.compute_from_lines` solve. Pooling more endpoints
   averages out per-frame line-detection noise, so the consolidated fit reduces
   or matches the typical single-frame reprojection error. This deliberately
   REUSES the existing ICP/RANSAC line solver rather than implementing an
   N-pose bundle adjuster (a separate future sub-epic).

Both features report error in meters through the same
:class:`ReprojectionStats` (RMS / 90th-percentile / max), computed over line
*endpoints*. Endpoint correspondence is assumed — it holds for the synthetic
known-H datasets and for frames whose lines were matched to ortho lines by the
seeded/auto leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from poc_homography.calibration.extrinsic.ptz_registration import (
    PtzRegistrationResult,
    build_line_inputs,
    intrinsic_kwargs,
    register_frame_lines,
)
from poc_homography.homography.map_points import MapPointHomography
from poc_homography.types import PixelsFloat

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import numpy.typing as npt

    from poc_homography.calibration.lens_distortion.calibration_table import (
        CameraCalibrationTable,
    )
    from poc_homography.calibration.lens_distortion.line_detection import CandidateLine
    from poc_homography.domain.vo.geotiff import GeoTiff
    from poc_homography.domain.vo.ortho_line import OrthoLine

# Minimum lines the underlying solver needs (compute_from_lines: >= 2 lines).
_MIN_FIT_LINES = 2
# Hold-out validation needs at least one held-out line on top of the fit set.
_MIN_HOLDOUT_SEED = _MIN_FIT_LINES + 1
# Percentile reported alongside RMS/max for the error distribution's tail.
_TAIL_PERCENTILE = 90.0


@dataclass(frozen=True)
class ReprojectionStats:
    """Reprojection-error summary in ortho ground units (meters).

    Attributes:
        rms_m: Root-mean-square endpoint error in meters.
        p90_m: 90th-percentile endpoint error in meters.
        max_m: Maximum endpoint error in meters.
        n_points: Number of endpoint correspondences the stats were computed over.
    """

    rms_m: float
    p90_m: float
    max_m: float
    n_points: int


@dataclass(frozen=True)
class HoldoutValidationResult:
    """Outcome of a fit/hold-out reprojection validation.

    Attributes:
        registration: Registration solved on the FIT correspondences only.
        fit_stats: Reprojection error (meters) over the fit correspondences.
        holdout_stats: Reprojection error (meters) over the held-out
            correspondences the fit never saw.
        rms_threshold_m: Hold-out RMS threshold (meters) the fit was checked
            against; ``holdout_stats.rms_m`` is guaranteed ``<=`` this value.
    """

    registration: PtzRegistrationResult
    fit_stats: ReprojectionStats
    holdout_stats: ReprojectionStats
    rms_threshold_m: float


@dataclass(frozen=True)
class MultiFrameConsolidationResult:
    """Consolidated registration over several frames at one PTZ state.

    Mirrors :class:`PtzRegistrationResult` but adds the frame count and a
    meters-unit reprojection summary over every pooled endpoint.

    Attributes:
        homography_matrix: 3x3 matrix mapping frame pixels → ortho map-pixels.
        inverse_matrix: 3x3 inverse mapping ortho map-pixels → frame pixels.
        num_frames: Number of frames pooled into the consolidated solve.
        num_lines: Total pooled line correspondences across all frames.
        num_inliers: Inlier point-to-line correspondences after RANSAC/ICP.
        inlier_ratio: Ratio of inliers to total correspondences in [0.0, 1.0].
        mean_perp_error: Mean perpendicular error in map pixels (solver units).
        max_perp_error: Maximum perpendicular error in map pixels.
        rmse: Root-mean-square of perpendicular errors in map pixels.
        reprojection_stats: Endpoint reprojection error in meters.
        run_id: Survey run this PTZ state came from (None if not applicable).
        camera_id: Camera this PTZ state came from (None if not applicable).
        commanded_zoom: Zoom the camera was commanded to for these frames.
        commanded_tilt: Tilt the camera was commanded to for these frames.
    """

    homography_matrix: npt.NDArray[np.float64]
    inverse_matrix: npt.NDArray[np.float64]
    num_frames: int
    num_lines: int
    num_inliers: int
    inlier_ratio: float
    mean_perp_error: float
    max_perp_error: float
    rmse: float
    reprojection_stats: ReprojectionStats
    run_id: str | None
    camera_id: str | None
    commanded_zoom: float
    commanded_tilt: float


@dataclass(frozen=True)
class FrameCorrespondence:
    """One frame's contribution to a multi-frame consolidation.

    Attributes:
        frame_lines: Camera frame lines in DISTORTED pixel space.
        seed: Frame-line index → ortho line id (``ortho_{i}``) correspondence.
    """

    frame_lines: Sequence[CandidateLine]
    seed: Mapping[int, str]


class HoldoutValidationError(Exception):
    """Raised when hold-out reprojection RMS exceeds the configured threshold.

    Attributes:
        holdout_rms_m: Measured hold-out RMS error in meters.
        threshold_m: Threshold the measured RMS exceeded.
        holdout_stats: Full hold-out reprojection statistics.
    """

    def __init__(
        self,
        holdout_rms_m: float,
        threshold_m: float,
        holdout_stats: ReprojectionStats,
    ) -> None:
        """Build the error from the failing hold-out statistics.

        Args:
            holdout_rms_m: Measured hold-out RMS error in meters.
            threshold_m: Threshold the measured RMS exceeded.
            holdout_stats: Full hold-out reprojection statistics.
        """
        super().__init__(
            f"Hold-out reprojection RMS {holdout_rms_m:.4f} m exceeds threshold {threshold_m:.4f} m"
        )
        self.holdout_rms_m = holdout_rms_m
        self.threshold_m = threshold_m
        self.holdout_stats = holdout_stats


def _ortho_index(ortho_id: str) -> int:
    """Parse the positional index out of an ``ortho_{i}`` id.

    Args:
        ortho_id: Ortho line id of the form ``ortho_{i}``.

    Returns:
        The integer index ``i``.

    Raises:
        ValueError: If ``ortho_id`` is not of the form ``ortho_{i}``.
    """
    prefix = "ortho_"
    if not ortho_id.startswith(prefix):
        raise ValueError(f"Malformed ortho id {ortho_id!r}; expected 'ortho_<int>'")
    try:
        return int(ortho_id[len(prefix) :])
    except ValueError as exc:
        raise ValueError(f"Malformed ortho id {ortho_id!r}; expected 'ortho_<int>'") from exc


def _apply_homography(
    homography: npt.NDArray[np.float64], point: tuple[float, float]
) -> tuple[float, float]:
    """Apply a 3x3 homography to a single (x, y) point in float64.

    Uses an explicit float64 matrix-vector product (not
    ``cv2.perspectiveTransform``, which casts to float32) to keep meter-scale
    reprojection errors free of avoidable single-precision noise.

    Args:
        homography: 3x3 homography matrix.
        point: (x, y) source point.

    Returns:
        (x', y') transformed point in the destination frame.
    """
    vec = homography @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))


def _correspondence_pairs(
    frame_lines: Sequence[CandidateLine],
    ortho_lines: Sequence[OrthoLine],
    seed: Mapping[int, str],
) -> list[tuple[CandidateLine, OrthoLine]]:
    """Resolve a seed into (frame_line, ortho_line) pairs.

    Args:
        frame_lines: Camera frame lines.
        ortho_lines: Orthophoto lines, keyed positionally as ``ortho_{i}``.
        seed: Frame-line index → ortho line id correspondence.

    Returns:
        One ``(frame_line, ortho_line)`` tuple per seed entry.

    Raises:
        ValueError: If a seed entry references an out-of-range frame index or an
            ortho index outside ``ortho_lines``.
    """
    pairs: list[tuple[CandidateLine, OrthoLine]] = []
    for frame_index, ortho_id in seed.items():
        if not 0 <= frame_index < len(frame_lines):
            raise ValueError(
                f"Seed frame-line index {frame_index} out of range "
                f"for {len(frame_lines)} frame lines"
            )
        ortho_idx = _ortho_index(ortho_id)
        if not 0 <= ortho_idx < len(ortho_lines):
            raise ValueError(
                f"Seed references ortho index {ortho_idx} out of range "
                f"for {len(ortho_lines)} ortho lines"
            )
        pairs.append((frame_lines[frame_index], ortho_lines[ortho_idx]))
    return pairs


def _reprojection_errors_m(
    homography: npt.NDArray[np.float64],
    pairs: Sequence[tuple[CandidateLine, OrthoLine]],
    geotiff: GeoTiff,
) -> list[float]:
    """Per-endpoint reprojection errors in meters for matched line pairs.

    Each frame-line endpoint is projected to ortho map-pixels by ``homography``,
    then both the projected point and the true ortho endpoint are converted to
    ground coordinates via :meth:`GeoTiff.pixel_to_geo`; the Euclidean distance
    between them (in CRS meters) is the error.

    Args:
        homography: Frame-pixel → ortho map-pixel homography.
        pairs: Matched ``(frame_line, ortho_line)`` correspondences.
        geotiff: Georeference used to convert map pixels to ground meters.

    Returns:
        One error (meters) per matched endpoint (two per line pair).
    """
    errors: list[float] = []
    for frame_line, ortho_line in pairs:
        for frame_pt, ortho_pt in (
            (frame_line.start, ortho_line.start_pixel),
            (frame_line.end, ortho_line.end_pixel),
        ):
            proj_x, proj_y = _apply_homography(homography, (frame_pt[0], frame_pt[1]))
            proj_e, proj_n = geotiff.pixel_to_geo(PixelsFloat(proj_x), PixelsFloat(proj_y))
            true_e, true_n = geotiff.pixel_to_geo(
                PixelsFloat(float(ortho_pt[0])), PixelsFloat(float(ortho_pt[1]))
            )
            errors.append(
                float(np.hypot(float(proj_e) - float(true_e), float(proj_n) - float(true_n)))
            )
    return errors


def _stats(errors: Sequence[float]) -> ReprojectionStats:
    """Summarise a list of meter errors as RMS / 90th-percentile / max.

    Args:
        errors: Per-endpoint errors in meters (must be non-empty).

    Returns:
        A :class:`ReprojectionStats` over ``errors``.

    Raises:
        ValueError: If ``errors`` is empty.
    """
    if not errors:
        raise ValueError("Cannot compute reprojection stats over zero correspondences")
    arr = np.asarray(errors, dtype=np.float64)
    return ReprojectionStats(
        rms_m=float(np.sqrt(np.mean(arr**2))),
        p90_m=float(np.percentile(arr, _TAIL_PERCENTILE)),
        max_m=float(np.max(arr)),
        n_points=int(arr.size),
    )


def _split_seed(
    seed: Mapping[int, str],
    holdout_fraction: float,
    rng_seed: int,
) -> tuple[dict[int, str], dict[int, str]]:
    """Split a seed into (fit, hold-out) subsets deterministically.

    The number held out is ``round(n * holdout_fraction)``, clamped so the fit
    set keeps at least :data:`_MIN_FIT_LINES` correspondences and at least one
    line is held out. Selection uses a ``numpy`` RNG seeded with ``rng_seed``
    over the frame indices sorted ascending, so the split is reproducible.

    Args:
        seed: Full frame-line index → ortho id correspondence.
        holdout_fraction: Fraction of correspondences to hold out, in (0, 1).
        rng_seed: Seed for the permutation RNG.

    Returns:
        ``(fit_seed, holdout_seed)``.

    Raises:
        ValueError: If ``holdout_fraction`` is outside (0, 1) or the seed is too
            small to yield both a >= 2-line fit set and a >= 1-line hold-out set.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1); got {holdout_fraction}")
    n = len(seed)
    if n < _MIN_HOLDOUT_SEED:
        raise ValueError(
            f"Hold-out validation needs at least {_MIN_HOLDOUT_SEED} "
            f"correspondences (>= {_MIN_FIT_LINES} fit + >= 1 hold-out), got {n}"
        )

    n_holdout = round(n * holdout_fraction)
    n_holdout = max(1, min(n_holdout, n - _MIN_FIT_LINES))

    indices = sorted(seed.keys())
    rng = np.random.default_rng(rng_seed)
    permuted = rng.permutation(np.asarray(indices, dtype=np.int64))
    holdout_keys = {int(k) for k in permuted[:n_holdout]}

    fit_seed = {k: v for k, v in seed.items() if k not in holdout_keys}
    holdout_seed = {k: v for k, v in seed.items() if k in holdout_keys}
    return fit_seed, holdout_seed


def validate_with_holdout(
    *,
    frame_lines: Sequence[CandidateLine],
    ortho_lines: Sequence[OrthoLine],
    seed: Mapping[int, str],
    calibration_table: CameraCalibrationTable,
    commanded_zoom: float,
    commanded_tilt: float,
    map_id: str,
    geotiff: GeoTiff,
    rms_threshold_m: float,
    holdout_fraction: float = 0.3,
    rng_seed: int = 0,
    run_id: str | None = None,
    camera_id: str | None = None,
    ransac_threshold: float = 50.0,
    min_inlier_ratio: float = 0.5,
) -> HoldoutValidationResult:
    """Fit a registration on a subset and validate it on held-out lines.

    Splits ``seed`` into a fit set and a hold-out set, solves the homography on
    the fit set only via :func:`register_frame_lines`, then measures
    reprojection error (in meters via ``geotiff``) on BOTH sets. The hold-out
    RMS is the generalisation metric; if it exceeds ``rms_threshold_m`` a
    :class:`HoldoutValidationError` is raised.

    Args:
        frame_lines: Camera frame lines in DISTORTED pixel space.
        ortho_lines: Orthophoto lines, keyed positionally as ``ortho_{i}``.
        seed: Frame-line index → ortho line id correspondence.
        calibration_table: Per-zoom camera calibration table.
        commanded_zoom: Zoom the frame was captured at (intrinsics lookup key).
        commanded_tilt: Tilt the frame was captured at (provenance only).
        map_id: Orthophoto map identifier for the homography provider.
        geotiff: Georeference for converting map-pixel error to ground meters.
        rms_threshold_m: Maximum acceptable hold-out RMS error in meters.
        holdout_fraction: Fraction of correspondences to hold out, in (0, 1).
        rng_seed: Seed for the reproducible fit/hold-out split.
        run_id: Survey run for provenance (optional).
        camera_id: Camera id for provenance (optional).
        ransac_threshold: Max reprojection error (map pixels) for inliers.
        min_inlier_ratio: Minimum inlier ratio considered a valid fit.

    Returns:
        A :class:`HoldoutValidationResult` with the fit registration and both
        fit and hold-out reprojection statistics.

    Raises:
        ValueError: If the seed is too small to split or inputs are invalid.
        HoldoutValidationError: If the hold-out RMS exceeds ``rms_threshold_m``.
        RuntimeError: If the underlying homography computation fails.
    """
    fit_seed, holdout_seed = _split_seed(seed, holdout_fraction, rng_seed)

    registration = register_frame_lines(
        frame_lines=frame_lines,
        ortho_lines=ortho_lines,
        seed=fit_seed,
        calibration_table=calibration_table,
        commanded_zoom=commanded_zoom,
        commanded_tilt=commanded_tilt,
        map_id=map_id,
        run_id=run_id,
        camera_id=camera_id,
        ransac_threshold=ransac_threshold,
        min_inlier_ratio=min_inlier_ratio,
    )

    homography = registration.homography_matrix
    fit_pairs = _correspondence_pairs(frame_lines, ortho_lines, fit_seed)
    holdout_pairs = _correspondence_pairs(frame_lines, ortho_lines, holdout_seed)
    fit_stats = _stats(_reprojection_errors_m(homography, fit_pairs, geotiff))
    holdout_stats = _stats(_reprojection_errors_m(homography, holdout_pairs, geotiff))

    if holdout_stats.rms_m > rms_threshold_m:
        raise HoldoutValidationError(holdout_stats.rms_m, rms_threshold_m, holdout_stats)

    return HoldoutValidationResult(
        registration=registration,
        fit_stats=fit_stats,
        holdout_stats=holdout_stats,
        rms_threshold_m=rms_threshold_m,
    )


def consolidate_ptz_frames(
    *,
    frames: Sequence[FrameCorrespondence],
    ortho_lines: Sequence[OrthoLine],
    calibration_table: CameraCalibrationTable,
    commanded_zoom: float,
    commanded_tilt: float,
    map_id: str,
    geotiff: GeoTiff,
    run_id: str | None = None,
    camera_id: str | None = None,
    ransac_threshold: float = 50.0,
    min_inlier_ratio: float = 0.5,
) -> MultiFrameConsolidationResult:
    """Consolidate several same-PTZ-state frames into one registration.

    Pools every frame's line correspondences into a single
    :meth:`MapPointHomography.compute_from_lines` solve at ``commanded_zoom``'s
    intrinsics. Because more endpoints average out per-frame line-detection
    noise, the consolidated fit reduces or matches the typical single-frame
    reprojection error. This reuses the existing ICP/RANSAC line solver and is
    NOT a general N-pose bundle adjuster.

    Args:
        frames: One :class:`FrameCorrespondence` per frame at this PTZ state.
        ortho_lines: Orthophoto lines, keyed positionally as ``ortho_{i}``.
        calibration_table: Per-zoom camera calibration table.
        commanded_zoom: Zoom the frames were captured at (intrinsics lookup key).
        commanded_tilt: Tilt the frames were captured at (provenance only).
        map_id: Orthophoto map identifier for the homography provider.
        geotiff: Georeference for converting map-pixel error to ground meters.
        run_id: Survey run for provenance (optional).
        camera_id: Camera id for provenance (optional).
        ransac_threshold: Max reprojection error (map pixels) for inliers.
        min_inlier_ratio: Minimum inlier ratio considered a valid fit.

    Returns:
        A :class:`MultiFrameConsolidationResult` with the consolidated
        homography, solver metrics, and a meters-unit reprojection summary.

    Raises:
        ValueError: If no frames are supplied or a frame's seed is invalid.
        RuntimeError: If the underlying homography computation fails.
    """
    if not frames:
        raise ValueError("consolidate_ptz_frames needs at least one frame")

    pooled_annotations: list[dict[str, Any]] = []
    pooled_registry: dict[str, dict[str, float]] = {}
    for frame in frames:
        annotations, registry = build_line_inputs(frame.frame_lines, ortho_lines, frame.seed)
        pooled_annotations.extend(annotations)
        pooled_registry.update(registry)

    provider = MapPointHomography(map_id, **intrinsic_kwargs(calibration_table, commanded_zoom))
    result = provider.compute_from_lines(
        pooled_annotations,
        pooled_registry,
        ransac_threshold=ransac_threshold,
        min_inlier_ratio=min_inlier_ratio,
    )

    pairs: list[tuple[CandidateLine, OrthoLine]] = []
    for frame in frames:
        pairs.extend(_correspondence_pairs(frame.frame_lines, ortho_lines, frame.seed))
    reprojection_stats = _stats(_reprojection_errors_m(result.homography_matrix, pairs, geotiff))

    return MultiFrameConsolidationResult(
        homography_matrix=result.homography_matrix,
        inverse_matrix=result.inverse_matrix,
        num_frames=len(frames),
        num_lines=result.num_lines,
        num_inliers=result.num_inliers,
        inlier_ratio=result.inlier_ratio,
        mean_perp_error=result.mean_perp_error,
        max_perp_error=result.max_perp_error,
        rmse=result.rmse,
        reprojection_stats=reprojection_stats,
        run_id=run_id,
        camera_id=camera_id,
        commanded_zoom=commanded_zoom,
        commanded_tilt=commanded_tilt,
    )


__all__ = [
    "FrameCorrespondence",
    "HoldoutValidationError",
    "HoldoutValidationResult",
    "MultiFrameConsolidationResult",
    "ReprojectionStats",
    "consolidate_ptz_frames",
    "validate_with_holdout",
]
