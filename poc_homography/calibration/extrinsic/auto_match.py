"""Automatic frame-line ↔ ortho-line RANSAC matcher (the "auto-GCP" core).

This module REPLACES the operator-supplied seed of the SEEDED leaf
(:func:`poc_homography.calibration.extrinsic.ptz_registration.register_frame_lines`)
by *discovering* the frame-line ↔ ortho-line correspondence automatically, then
feeding the discovered seed to that very function. It does NOT implement a new
homography estimator: it only SELECTS correspondences and delegates the refined
solve to the existing ICP/RANSAC core in
:meth:`poc_homography.homography.map_points.MapPointHomography.compute_from_lines`.

Algorithm overview
------------------
1. **Candidate generation.** For each frame line, find ortho lines that are
   plausible matches by descriptor: orientation agreement (angle modulo 180°)
   and length-ratio compatibility. When a ``pose_prior`` is supplied it is used
   to *predict* where each frame line projects in ortho space and to gate
   candidates by midpoint proximity + projected-angle agreement; this is what
   disambiguates REPETITIVE near-parallel parking lines that pure descriptor
   gating would happily swap. Without a prior we fall back to pure descriptor
   (angle/length) gating — line-only matching.
2. **Seeded RANSAC** over minimal 2-line candidate sets. All sampling uses a
   :func:`numpy.random.default_rng` seeded with ``rng_seed`` so runs are bit-for-bit
   reproducible. Each minimal hypothesis is built with ``cv2.getPerspectiveTransform``
   on the 4 endpoint correspondences (trying both endpoint orientations of the
   second line and keeping the better) — deliberately NOT ``cv2.findHomography(...,
   cv2.RANSAC, ...)``, whose internal RNG is not seedable. A (frame, ortho) pair
   scores as an inlier when projecting the frame line's endpoints by the
   hypothesis lands within ``ransac_threshold`` map pixels (PERPENDICULAR
   distance to the ortho *infinite* line — consistent with ``compute_from_lines``).
   Each frame line resolves to at most one ortho line (lowest-error) so the seed
   is a clean, injective mapping.
3. **Fail-fast** BEFORE the refined solve: too few inliers, a "Poor"/"Insufficient"
   :func:`calculate_distribution` quality over the inlier frame endpoints, or a
   too-low inlier count all raise :class:`InsufficientCorrespondenceError`.
4. **Refined solve.** The discovered seed (``frame_index -> "ortho_{i}"``, with
   ``i`` the positional index of the ortho line in the input sequence, matching
   ``register_frame_lines``' convention) is handed to ``register_frame_lines`` and
   wrapped in an :class:`AutoMatchResult`.

pose_prior orientation
----------------------
``pose_prior`` is a 3×3 homography in the SAME orientation as the matcher's
output: **frame-pixel → ortho map-pixel**. A coarse prior of this orientation is
obtained from the commanded PTZ via
:meth:`poc_homography.homography.intrinsic_extrinsic.IntrinsicExtrinsicHomography.compute_from_config`.
The prior need only be approximately correct (a perturbed prior still
disambiguates) since it gates candidates rather than being solved against.

Scope
-----
Line-only matching is in scope. Junction-graph matching (using line
intersections as point features) is explicitly DEFERRED to a later leaf and is
NOT implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from poc_homography.calibration.extrinsic.ptz_registration import register_frame_lines
from poc_homography.validation.gcp_distribution import calculate_distribution

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from poc_homography.calibration.extrinsic.ptz_registration import PtzRegistrationResult
    from poc_homography.calibration.lens_distortion.calibration_table import (
        CameraCalibrationTable,
    )
    from poc_homography.calibration.lens_distortion.line_detection import CandidateLine
    from poc_homography.domain.vo.ortho_line import OrthoLine

# Minimal sample for a perspective hypothesis: 2 lines = 4 endpoint correspondences.
_MINIMAL_SAMPLE_LINES = 2
# A "Poor"/"Insufficient" distribution of inlier endpoints fails fast.
_REJECTED_DISTRIBUTION_QUALITIES = frozenset({"Poor", "Insufficient"})


class InsufficientCorrespondenceError(Exception):
    """Raised when automatic matching cannot yield a trustworthy correspondence.

    This is the typed fail-fast error for the auto-GCP matcher. It carries enough
    context for the caller to report *why* matching was rejected without parsing
    the message string.

    Attributes:
        reason: Human-readable explanation of which gate tripped.
        num_inliers: Inlier (frame, ortho) pairs found by the best hypothesis.
        num_candidates: Total descriptor/prior-gated candidate pairs considered.
        distribution_quality: Distribution quality label of the inlier frame
            endpoints (one of "Good"/"Fair"/"Poor"/"Insufficient"), or None when
            the failure occurred before distribution was assessed.
    """

    def __init__(
        self,
        reason: str,
        *,
        num_inliers: int,
        num_candidates: int,
        distribution_quality: str | None = None,
    ) -> None:
        """Initialize with the fail-fast reason and match provenance.

        Args:
            reason: Human-readable explanation of which gate tripped.
            num_inliers: Inlier (frame, ortho) pairs from the best hypothesis.
            num_candidates: Total candidate pairs considered.
            distribution_quality: Inlier-endpoint distribution quality, if known.
        """
        super().__init__(reason)
        self.reason = reason
        self.num_inliers = num_inliers
        self.num_candidates = num_candidates
        self.distribution_quality = distribution_quality


@dataclass(frozen=True)
class AutoMatchResult:
    """Outcome of automatic matching + refined registration.

    Wraps the refined :class:`PtzRegistrationResult` together with the provenance
    of the automatic match: the discovered seed and the candidate/inlier counts.

    Attributes:
        seed: Discovered correspondence, frame-line index → ``"ortho_{i}"``.
        num_candidates: Total descriptor/prior-gated candidate pairs considered.
        num_inliers: Inlier (frame, ortho) pairs the winning hypothesis explained.
        distribution_quality: Distribution quality label of the inlier frame
            endpoints ("Good"/"Fair"/"Poor"/"Insufficient").
        registration: The refined registration result from
            :func:`register_frame_lines` (carries the homography, inverse, and
            fit metrics).
    """

    seed: Mapping[int, str]
    num_candidates: int
    num_inliers: int
    distribution_quality: str
    registration: PtzRegistrationResult

    @property
    def homography_matrix(self) -> npt.NDArray[np.float64]:
        """3x3 matrix mapping frame pixels → ortho map-pixels (refined fit)."""
        return self.registration.homography_matrix

    @property
    def inverse_matrix(self) -> npt.NDArray[np.float64]:
        """3x3 inverse mapping ortho map-pixels → frame pixels (refined fit)."""
        return self.registration.inverse_matrix


def _line_endpoints(
    line: CandidateLine | OrthoLine,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a line's (start, end) endpoints as float64 (x, y) arrays.

    Works for both :class:`CandidateLine` (``start``/``end``) and
    :class:`OrthoLine` (``start_pixel``/``end_pixel``).
    """
    start = getattr(line, "start", None)
    if start is not None:
        end = line.end  # type: ignore[union-attr]
    else:
        start = line.start_pixel  # type: ignore[union-attr]
        end = line.end_pixel  # type: ignore[union-attr]
    return (
        np.asarray([float(start[0]), float(start[1])], dtype=np.float64),
        np.asarray([float(end[0]), float(end[1])], dtype=np.float64),
    )


def _angle_deg(start: npt.NDArray[np.float64], end: npt.NDArray[np.float64]) -> float:
    """Orientation of a segment in degrees (atan2 of the endpoint delta)."""
    delta = end - start
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _angle_diff_mod180(angle_a: float, angle_b: float) -> float:
    """Smallest direction-independent angle difference, in [0, 90] degrees."""
    diff = abs(angle_a - angle_b) % 180.0
    return min(diff, 180.0 - diff)


def _length(start: npt.NDArray[np.float64], end: npt.NDArray[np.float64]) -> float:
    """Euclidean length of a segment."""
    return float(np.linalg.norm(end - start))


def _perp_distance(
    point: npt.NDArray[np.float64],
    line_start: npt.NDArray[np.float64],
    line_end: npt.NDArray[np.float64],
) -> float:
    """Perpendicular distance from a point to an infinite line through two points.

    Mirrors ``MapPointHomography._point_to_line_distance`` (unclamped projection)
    so the inlier metric here matches the refined solver's metric exactly.
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len_sq = float(np.dot(line_vec, line_vec))
    if line_len_sq < 1e-10:
        return float(np.linalg.norm(point_vec))
    t = float(np.dot(point_vec, line_vec)) / line_len_sq
    projection = line_start + t * line_vec
    return float(np.linalg.norm(point - projection))


@dataclass(frozen=True)
class _LineGeom:
    """Cached geometry for one line (frame or ortho)."""

    start: npt.NDArray[np.float64]
    end: npt.NDArray[np.float64]
    mid: npt.NDArray[np.float64]
    angle_deg: float
    length: float


def _geom(line: CandidateLine | OrthoLine) -> _LineGeom:
    """Build cached :class:`_LineGeom` for a line."""
    start, end = _line_endpoints(line)
    return _LineGeom(
        start=start,
        end=end,
        mid=(start + end) / 2.0,
        angle_deg=_angle_deg(start, end),
        length=_length(start, end),
    )


def _apply_homography(
    homography: npt.NDArray[np.float64], point: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Apply a 3x3 homography to a single (x, y) point."""
    pt = np.asarray([[[point[0], point[1]]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, homography)[0, 0]
    return np.asarray([float(out[0]), float(out[1])], dtype=np.float64)


def _generate_candidates(
    frame_geoms: Sequence[_LineGeom],
    ortho_geoms: Sequence[_LineGeom],
    *,
    pose_prior: npt.NDArray[np.float64] | None,
    angle_tolerance_deg: float,
    length_ratio_tolerance: float,
    prior_distance_tolerance: float,
) -> list[tuple[int, int]]:
    """Enumerate plausible (frame_index, ortho_index) candidate pairs.

    Descriptor gating (always applied): the projected/raw orientation must agree
    within ``angle_tolerance_deg`` (modulo 180°), and the length ratio must lie
    within ``[1 - t, 1 + t]`` for ``t = length_ratio_tolerance``. When
    ``pose_prior`` (frame→ortho) is given, each frame line is projected into ortho
    space; the length ratio and angle are then compared in ortho space and the
    projected midpoint must be within ``prior_distance_tolerance`` map pixels of
    the ortho line's midpoint. This proximity gate is what disambiguates
    repetitive parallel lines.

    Returns:
        List of (frame_index, ortho_index) candidate pairs.
    """
    candidates: list[tuple[int, int]] = []

    for f_idx, f_geom in enumerate(frame_geoms):
        if pose_prior is not None:
            proj_start = _apply_homography(pose_prior, f_geom.start)
            proj_end = _apply_homography(pose_prior, f_geom.end)
            f_angle = _angle_deg(proj_start, proj_end)
            f_length = _length(proj_start, proj_end)
            f_mid = (proj_start + proj_end) / 2.0
        else:
            f_angle = f_geom.angle_deg
            f_length = f_geom.length
            f_mid = f_geom.mid

        for o_idx, o_geom in enumerate(ortho_geoms):
            if _angle_diff_mod180(f_angle, o_geom.angle_deg) > angle_tolerance_deg:
                continue
            if o_geom.length < 1e-9 or f_length < 1e-9:
                continue
            ratio = f_length / o_geom.length
            if not (1.0 - length_ratio_tolerance <= ratio <= 1.0 + length_ratio_tolerance):
                continue
            if pose_prior is not None:
                if float(np.linalg.norm(f_mid - o_geom.mid)) > prior_distance_tolerance:
                    continue
            candidates.append((f_idx, o_idx))

    return candidates


def _minimal_hypothesis(
    frame_a: _LineGeom,
    ortho_a: _LineGeom,
    frame_b: _LineGeom,
    ortho_b: _LineGeom,
) -> npt.NDArray[np.float64] | None:
    """Compute a frame→ortho homography from two line correspondences.

    Builds the 4 endpoint correspondences and solves with
    ``cv2.getPerspectiveTransform`` (deterministic, no RNG). The two lines'
    relative endpoint orientation is ambiguous, so all four orientation
    combinations are tried and the one with the smallest endpoint reprojection
    residual is kept.

    Returns:
        A 3x3 frame→ortho homography, or None if every orientation is degenerate.
    """
    best_h: npt.NDArray[np.float64] | None = None
    best_err = float("inf")

    for flip_a in (False, True):
        for flip_b in (False, True):
            a_o_start, a_o_end = (
                (ortho_a.end, ortho_a.start) if flip_a else (ortho_a.start, ortho_a.end)
            )
            b_o_start, b_o_end = (
                (ortho_b.end, ortho_b.start) if flip_b else (ortho_b.start, ortho_b.end)
            )
            src = np.asarray(
                [frame_a.start, frame_a.end, frame_b.start, frame_b.end], dtype=np.float32
            )
            dst = np.asarray([a_o_start, a_o_end, b_o_start, b_o_end], dtype=np.float32)
            try:
                hyp = cv2.getPerspectiveTransform(src, dst)
            except cv2.error:
                continue
            hyp_arr = np.asarray(hyp, dtype=np.float64)
            if abs(float(np.linalg.det(hyp_arr))) < 1e-12:
                continue
            projected = cv2.perspectiveTransform(
                src.reshape(-1, 1, 2).astype(np.float64), hyp_arr
            ).reshape(-1, 2)
            err = float(np.sum(np.linalg.norm(projected - dst.astype(np.float64), axis=1)))
            if err < best_err:
                best_err = err
                best_h = hyp_arr

    return best_h


def _score_hypothesis(
    homography: npt.NDArray[np.float64],
    candidates: Sequence[tuple[int, int]],
    frame_geoms: Sequence[_LineGeom],
    ortho_geoms: Sequence[_LineGeom],
    ransac_threshold: float,
) -> dict[int, tuple[int, float]]:
    """Score candidates against a hypothesis, resolving one ortho per frame line.

    A (frame, ortho) candidate is an inlier when BOTH frame endpoints, projected
    by ``homography``, lie within ``ransac_threshold`` of the ortho infinite
    line. Each frame line keeps only its lowest-error inlier ortho match.

    Returns:
        Mapping frame_index → (ortho_index, max_endpoint_error) for inliers.
    """
    best_for_frame: dict[int, tuple[int, float]] = {}

    for f_idx, o_idx in candidates:
        f_geom = frame_geoms[f_idx]
        o_geom = ortho_geoms[o_idx]
        proj_start = _apply_homography(homography, f_geom.start)
        proj_end = _apply_homography(homography, f_geom.end)
        dist_start = _perp_distance(proj_start, o_geom.start, o_geom.end)
        dist_end = _perp_distance(proj_end, o_geom.start, o_geom.end)
        worst = max(dist_start, dist_end)
        if worst >= ransac_threshold:
            continue
        prior = best_for_frame.get(f_idx)
        if prior is None or worst < prior[1]:
            best_for_frame[f_idx] = (o_idx, worst)

    return best_for_frame


def _resolve_injective(
    matches: Mapping[int, tuple[int, float]],
) -> dict[int, int]:
    """Resolve frame→ortho matches to an injective mapping (one ortho per line).

    If two frame lines claim the same ortho line, only the lower-error frame line
    keeps it; the loser is dropped. Returns frame_index → ortho_index.
    """
    # Order frame lines by ascending error so the best claim wins each ortho.
    ordered = sorted(matches.items(), key=lambda kv: kv[1][1])
    used_ortho: set[int] = set()
    resolved: dict[int, int] = {}
    for f_idx, (o_idx, _err) in ordered:
        if o_idx in used_ortho:
            continue
        used_ortho.add(o_idx)
        resolved[f_idx] = o_idx
    return resolved


def match_and_register(
    *,
    frame_lines: Sequence[CandidateLine],
    ortho_lines: Sequence[OrthoLine],
    calibration_table: CameraCalibrationTable,
    commanded_zoom: float,
    commanded_tilt: float,
    map_id: str,
    image_width: int,
    image_height: int,
    pose_prior: npt.NDArray[np.float64] | None = None,
    run_id: str | None = None,
    camera_id: str | None = None,
    rng_seed: int = 0,
    ransac_iterations: int = 200,
    ransac_threshold: float = 50.0,
    min_inlier_ratio: float = 0.5,
    min_inliers: int = 4,
    angle_tolerance_deg: float = 10.0,
    length_ratio_tolerance: float = 0.5,
    prior_distance_tolerance: float = 200.0,
) -> AutoMatchResult:
    """Automatically match frame lines to ortho lines, then register them.

    This is the no-manual-seed entry point: it discovers the frame-line ↔
    ortho-line correspondence via seeded RANSAC over line descriptors (optionally
    gated by a coarse ``pose_prior``), fails fast on a weak/poorly-distributed
    inlier set, and delegates the refined solve to
    :func:`register_frame_lines`. No homography/RANSAC/ICP is reimplemented here;
    only candidate SELECTION is.

    Args:
        frame_lines: Camera frame lines in DISTORTED pixel space.
        ortho_lines: Orthophoto lines, keyed positionally as ``ortho_{i}``.
        calibration_table: Per-zoom camera calibration table.
        commanded_zoom: Zoom the frame was captured at (intrinsics lookup key).
        commanded_tilt: Tilt the frame was captured at (provenance only).
        map_id: Orthophoto map identifier for the homography provider.
        image_width: Frame width in pixels (for inlier distribution analysis).
        image_height: Frame height in pixels (for inlier distribution analysis).
        pose_prior: Optional coarse 3x3 homography in **frame-pixel → ortho
            map-pixel** orientation (same as the output). Used only to gate
            candidates by projected proximity + angle, disambiguating repetitive
            parallel lines. A perturbed prior is fine. Obtain a commanded-PTZ
            prior from ``IntrinsicExtrinsicHomography.compute_from_config``.
        run_id: Survey run for provenance (optional).
        camera_id: Camera id for provenance (optional).
        rng_seed: Seed for ``numpy.random.default_rng`` — fixes ALL sampling so
            runs are reproducible.
        ransac_iterations: Number of minimal 2-line samples to try.
        ransac_threshold: Max perpendicular error (map pixels) for an inlier;
            also forwarded to the refined solver.
        min_inlier_ratio: Minimum inlier ratio for a valid refined fit.
        min_inliers: Minimum inlier (frame, ortho) pairs required before solving.
        angle_tolerance_deg: Max orientation difference (mod 180°) for a candidate.
        length_ratio_tolerance: Allowed frame/ortho length-ratio deviation from 1.
        prior_distance_tolerance: Max projected-midpoint distance (map pixels)
            for a candidate when ``pose_prior`` is supplied.

    Returns:
        :class:`AutoMatchResult` with the discovered seed, match provenance, and
        the refined :class:`PtzRegistrationResult`.

    Raises:
        InsufficientCorrespondenceError: If the best inlier set is below
            ``min_inliers``, if too few candidates exist to sample, or if the
            inlier frame endpoints' distribution quality is "Poor"/"Insufficient".
        ValueError: Propagated from the refined solver on an otherwise-poor fit
            (e.g. ``min_inlier_ratio`` not met after ICP).
    """
    frame_geoms = [_geom(line) for line in frame_lines]
    ortho_geoms = [_geom(line) for line in ortho_lines]

    candidates = _generate_candidates(
        frame_geoms,
        ortho_geoms,
        pose_prior=pose_prior,
        angle_tolerance_deg=angle_tolerance_deg,
        length_ratio_tolerance=length_ratio_tolerance,
        prior_distance_tolerance=prior_distance_tolerance,
    )
    num_candidates = len(candidates)

    if num_candidates < _MINIMAL_SAMPLE_LINES:
        raise InsufficientCorrespondenceError(
            f"Too few candidate matches to sample a hypothesis: {num_candidates} "
            f"(need >= {_MINIMAL_SAMPLE_LINES})",
            num_inliers=0,
            num_candidates=num_candidates,
        )

    rng = np.random.default_rng(rng_seed)
    candidate_array = np.asarray(candidates, dtype=np.int64)

    best_resolved: dict[int, int] = {}

    for _ in range(ransac_iterations):
        sample_idx = rng.choice(num_candidates, size=_MINIMAL_SAMPLE_LINES, replace=False)
        (fa, oa), (fb, ob) = candidate_array[sample_idx]
        # Two distinct frame lines AND two distinct ortho lines are required for a
        # non-degenerate minimal hypothesis.
        if fa == fb or oa == ob:
            continue

        hypothesis = _minimal_hypothesis(
            frame_geoms[fa], ortho_geoms[oa], frame_geoms[fb], ortho_geoms[ob]
        )
        if hypothesis is None:
            continue

        matches = _score_hypothesis(
            hypothesis, candidates, frame_geoms, ortho_geoms, ransac_threshold
        )
        resolved = _resolve_injective(matches)
        if len(resolved) > len(best_resolved):
            best_resolved = resolved

    num_inliers = len(best_resolved)

    if num_inliers < min_inliers:
        raise InsufficientCorrespondenceError(
            f"Best inlier set too small: {num_inliers} < min_inliers {min_inliers}",
            num_inliers=num_inliers,
            num_candidates=num_candidates,
        )

    # Distribution fail-fast over the inlier frame endpoints. We use
    # calculate_distribution + an explicit count check rather than
    # validate_ground_control_points: the latter requires GCP-shaped dicts
    # (map_id/map_pixel_x/...) and only WARNS (does not raise) on low count, so it
    # does not fit raw line endpoints nor give us a count fail-fast.
    inlier_endpoints: list[tuple[float, float]] = []
    for f_idx in best_resolved:
        f_geom = frame_geoms[f_idx]
        inlier_endpoints.append((float(f_geom.start[0]), float(f_geom.start[1])))
        inlier_endpoints.append((float(f_geom.end[0]), float(f_geom.end[1])))

    distribution = calculate_distribution(inlier_endpoints, image_width, image_height)
    if distribution.quality in _REJECTED_DISTRIBUTION_QUALITIES:
        raise InsufficientCorrespondenceError(
            f"Inlier endpoint distribution quality is {distribution.quality!r}; "
            f"need better spatial coverage",
            num_inliers=num_inliers,
            num_candidates=num_candidates,
            distribution_quality=distribution.quality,
        )

    seed: dict[int, str] = {
        f_idx: f"ortho_{o_idx}" for f_idx, o_idx in sorted(best_resolved.items())
    }

    registration = register_frame_lines(
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

    return AutoMatchResult(
        seed=seed,
        num_candidates=num_candidates,
        num_inliers=num_inliers,
        distribution_quality=distribution.quality,
        registration=registration,
    )


__all__ = [
    "AutoMatchResult",
    "InsufficientCorrespondenceError",
    "match_and_register",
]
