"""Pose generators for the six survey-pattern families.

Each generator is a pure function producing poses (or phase plans, or
sub-sequences) for one capture pattern. FOV-based generators reuse the helpers
in :mod:`poc_homography.survey.planner.fov`, which in turn reuse the project's
intrinsics computation.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.survey.planner.fov import (
    angular_step_degrees,
    horizontal_fov_degrees,
    vertical_fov_degrees,
)
from poc_homography.survey.planner.phase_plan import PhasePlan
from poc_homography.survey.planner.poses import ApproachDirection, Pose
from poc_homography.types import Degrees, Unitless

if TYPE_CHECKING:
    from collections.abc import Sequence

    from poc_homography.domain.enums.camera_spec import CameraSpec
    from poc_homography.domain.vo.tilt_envelope import TiltEnvelope

# Tolerance (in axis units) for including the inclusive endpoint of a range.
_ENDPOINT_TOLERANCE = 1e-9


class SweepAxis(str, Enum):
    """The PTZ axis swept by :func:`directional_sweep`.

    Members subclass :class:`str` so their ``.value`` is a stable, plain
    string safe for serialisation.
    """

    PAN = "pan"
    TILT = "tilt"
    ZOOM = "zoom"


def _inclusive_values(start: float, end: float, step: float) -> list[float]:
    """Build an inclusive ascending list from ``start`` to ``end`` by ``step``.

    The endpoint ``end`` is always included (clamped exactly), and no value
    exceeds ``end``. ``step`` must be strictly positive.

    Args:
        start: First value (lower bound).
        end: Last value (upper bound); always included.
        step: Positive spacing between consecutive values.

    Returns:
        Strictly ascending list of values covering ``[start, end]`` with no
        gap larger than ``step``.

    Raises:
        ValueError: If ``step`` is not strictly positive or ``end < start``.
    """
    if step <= 0.0:
        raise ValueError(f"step must be > 0; got {step!r}")
    if end < start:
        raise ValueError(f"end ({end!r}) must be >= start ({start!r})")

    values: list[float] = []
    count = math.floor((end - start) / step + _ENDPOINT_TOLERANCE)
    for i in range(count + 1):
        value = start + i * step
        if value > end:
            value = end
        values.append(value)
    # Ensure the endpoint is present and strictly increasing.
    if not values or abs(values[-1] - end) > _ENDPOINT_TOLERANCE:
        values.append(end)
    # De-duplicate any value that coincides with the endpoint.
    deduped: list[float] = []
    for value in values:
        if deduped and abs(value - deduped[-1]) <= _ENDPOINT_TOLERANCE:
            continue
        deduped.append(value)
    return deduped


def _pose_for_axis(
    axis: SweepAxis,
    swept_value: float,
    fixed_a: float,
    fixed_b: float,
    direction: ApproachDirection,
) -> Pose:
    """Build a pose placing ``swept_value`` on ``axis`` and fixed values elsewhere.

    ``fixed_a``/``fixed_b`` are the two non-swept axis values in PAN, TILT,
    ZOOM order skipping the swept axis.
    """
    if axis is SweepAxis.PAN:
        pan, tilt, zoom = swept_value, fixed_a, fixed_b
    elif axis is SweepAxis.TILT:
        pan, tilt, zoom = fixed_a, swept_value, fixed_b
    else:  # ZOOM
        pan, tilt, zoom = fixed_a, fixed_b, swept_value
    return Pose(
        pan=Degrees(pan),
        tilt=Degrees(tilt),
        zoom=Unitless(zoom),
        approach_direction=direction,
    )


def directional_sweep(
    axis: SweepAxis,
    start: float,
    end: float,
    step: float,
    fixed_a: float,
    fixed_b: float,
    *,
    phase: SurveyPhase,
) -> tuple[PhasePlan, PhasePlan]:
    """Generate ascending and descending sweep plans along one axis.

    Args:
        axis: The swept axis.
        start: Lower bound of the swept range.
        end: Upper bound of the swept range.
        step: Positive spacing between consecutive poses.
        fixed_a: First non-swept axis value (PAN, TILT, ZOOM order, skipping
            the swept axis).
        fixed_b: Second non-swept axis value.
        phase: The phase identity to stamp on both plans.

    Returns:
        ``(ascending_plan, descending_plan)``. The ascending pass is strictly
        increasing in the swept value (wide->tele for ZOOM); the descending
        pass is its strict reverse. Each pose is tagged with the matching
        :class:`ApproachDirection`.
    """
    ascending_values = _inclusive_values(start, end, step)
    descending_values = list(reversed(ascending_values))

    ascending_poses = tuple(
        _pose_for_axis(axis, v, fixed_a, fixed_b, ApproachDirection.ASCENDING)
        for v in ascending_values
    )
    descending_poses = tuple(
        _pose_for_axis(axis, v, fixed_a, fixed_b, ApproachDirection.DESCENDING)
        for v in descending_values
    )

    ascending_plan = PhasePlan(
        phase=phase,
        poses=ascending_poses,
        approach_directions=(ApproachDirection.ASCENDING,),
    )
    descending_plan = PhasePlan(
        phase=phase,
        poses=descending_poses,
        approach_directions=(ApproachDirection.DESCENDING,),
    )
    return ascending_plan, descending_plan


def fov_grid(
    spec: CameraSpec,
    pan_range: tuple[float, float],
    tilt_range: tuple[float, float],
    zoom_levels: Sequence[float],
    overlap_fraction: float,
    *,
    phase: SurveyPhase = SurveyPhase.MAIN_SURVEY,
    tilt_envelope: TiltEnvelope | None = None,
) -> list[Pose]:
    """Generate a FOV-overlapping pan x tilt grid for each zoom level.

    For each zoom level the pan/tilt steps are derived from the horizontal and
    vertical FOV at that zoom so adjacent frames overlap by
    ``overlap_fraction``. Both range endpoints are always covered, and no gap
    between consecutive pan (or tilt) values exceeds one step.

    When a ``tilt_envelope`` is supplied, tiles that point above the per-azimuth
    max-up useful tilt are skipped (sky tiles): a pose is kept only when its
    ``tilt >= upper_bound(pan)`` (positive tilt = down, so a smaller tilt points
    further up). With ``tilt_envelope=None`` the output is byte-for-byte
    identical to the unconstrained grid.

    Args:
        spec: Camera specification.
        pan_range: ``(pan_min, pan_max)`` inclusive bounds.
        tilt_range: ``(tilt_min, tilt_max)`` inclusive bounds.
        zoom_levels: Zoom levels to cover, in order.
        overlap_fraction: Desired fractional overlap in ``[0.0, 1.0)``.
        phase: Phase identity (unused for tagging; documents intent).
        tilt_envelope: Optional per-azimuth tilt envelope; when present, sky
            tiles above the per-azimuth bound are omitted.

    Returns:
        Flat pose list ordered zoom, then tilt rows, then pan columns.
    """
    poses: list[Pose] = []
    for zoom in zoom_levels:
        pan_step = angular_step_degrees(horizontal_fov_degrees(spec, zoom), overlap_fraction)
        tilt_step = angular_step_degrees(vertical_fov_degrees(spec, zoom), overlap_fraction)
        pan_values = _inclusive_values(pan_range[0], pan_range[1], float(pan_step))
        tilt_values = _inclusive_values(tilt_range[0], tilt_range[1], float(tilt_step))
        for tilt in tilt_values:
            for pan in pan_values:
                if _is_sky_tile(tilt, pan, tilt_envelope):
                    continue
                poses.append(Pose(pan=Degrees(pan), tilt=Degrees(tilt), zoom=Unitless(zoom)))
    return poses


def _is_sky_tile(tilt: float, pan: float, tilt_envelope: TiltEnvelope | None) -> bool:
    """Return whether ``(pan, tilt)`` points above the per-azimuth useful band.

    With no envelope nothing is sky (preserving unconstrained behaviour). With
    an envelope, a tile is sky when its tilt is strictly above (numerically
    below, since positive tilt = down) the interpolated per-azimuth bound.
    """
    if tilt_envelope is None:
        return False
    return tilt < tilt_envelope.upper_bound(pan) - _ENDPOINT_TOLERANCE


def nadir_region(
    spec: CameraSpec,
    nadir_pan: float,
    nadir_tilt: float,
    radius_deg: float,
    zoom: float,
    overlap_fraction: float,
    *,
    phase: SurveyPhase = SurveyPhase.DENSE_NADIR,
) -> list[Pose]:
    """Generate a dense FOV-overlapping disc of poses around a nadir anchor.

    The grid spans ``+/- radius_deg`` in both pan and tilt at the single given
    zoom, then is filtered to a disc: only poses within ``radius_deg`` of the
    anchor (Euclidean angular distance) are emitted.

    Args:
        spec: Camera specification.
        nadir_pan: Anchor pan in degrees.
        nadir_tilt: Anchor tilt in degrees.
        radius_deg: Disc radius in degrees.
        zoom: Single zoom level for all poses.
        overlap_fraction: Desired fractional overlap in ``[0.0, 1.0)``.
        phase: Phase identity (documents intent).

    Returns:
        Flat pose list, every pose within ``radius_deg`` of the anchor.
    """
    pan_step = angular_step_degrees(horizontal_fov_degrees(spec, zoom), overlap_fraction)
    tilt_step = angular_step_degrees(vertical_fov_degrees(spec, zoom), overlap_fraction)
    pan_values = _inclusive_values(nadir_pan - radius_deg, nadir_pan + radius_deg, float(pan_step))
    tilt_values = _inclusive_values(
        nadir_tilt - radius_deg, nadir_tilt + radius_deg, float(tilt_step)
    )

    poses: list[Pose] = []
    for tilt in tilt_values:
        for pan in pan_values:
            dpan = pan - nadir_pan
            dtilt = tilt - nadir_tilt
            if math.sqrt(dpan * dpan + dtilt * dtilt) <= radius_deg + _ENDPOINT_TOLERANCE:
                poses.append(Pose(pan=Degrees(pan), tilt=Degrees(tilt), zoom=Unitless(zoom)))
    return poses


def cross_zoom(
    anchors: Sequence[tuple[float, float]],
    zoom_levels: Sequence[float],
    *,
    phase: SurveyPhase = SurveyPhase.CROSS_ZOOM,
) -> list[Pose]:
    """Generate cross-zoom poses grouped by anchor.

    For each anchor (in order) one pose per zoom level (in order) is emitted
    before moving to the next anchor. Each pose is tagged with a stable
    ``region_id`` derived from the anchor index.

    Args:
        anchors: ``(pan, tilt)`` anchor positions in degrees, in order.
        zoom_levels: Zoom levels to revisit each anchor at, in order.
        phase: Phase identity (documents intent).

    Returns:
        Flat pose list grouped by anchor.
    """
    poses: list[Pose] = []
    for i, (pan, tilt) in enumerate(anchors):
        region_id = f"region_{i}"
        for zoom in zoom_levels:
            poses.append(
                Pose(
                    pan=Degrees(pan),
                    tilt=Degrees(tilt),
                    zoom=Unitless(zoom),
                    region_id=region_id,
                )
            )
    return poses


def repeatability_sequences(
    target_pan: float,
    target_tilt: float,
    target_zoom: float,
    approach_deltas: Sequence[tuple[float, float]],
    *,
    phase: SurveyPhase = SurveyPhase.REPEATABILITY,
) -> list[tuple[Pose, ...]]:
    """Generate repeatability sub-sequences that all converge on one target.

    For each ``(pan_offset, tilt_offset)`` delta a two-pose sub-sequence is
    produced: an approach-start pose offset from the target, then the identical
    target pose.

    Args:
        target_pan: Target pan in degrees.
        target_tilt: Target tilt in degrees.
        target_zoom: Target zoom factor.
        approach_deltas: ``(pan_offset, tilt_offset)`` offsets in degrees.
        phase: Phase identity (documents intent).

    Returns:
        One sub-sequence per delta; each ends at the identical target pose.
    """
    target = Pose(
        pan=Degrees(target_pan),
        tilt=Degrees(target_tilt),
        zoom=Unitless(target_zoom),
    )
    sequences: list[tuple[Pose, ...]] = []
    for pan_offset, tilt_offset in approach_deltas:
        approach_start = Pose(
            pan=Degrees(target_pan + pan_offset),
            tilt=Degrees(target_tilt + tilt_offset),
            zoom=Unitless(target_zoom),
        )
        sequences.append((approach_start, target))
    return sequences


def partition_holdout(
    poses: Sequence[Pose],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[Pose], list[Pose]]:
    """Deterministically partition poses into training and holdout sets.

    Args:
        poses: Poses to partition.
        holdout_fraction: Fraction to hold out, in ``[0.0, 1.0]``.
        seed: Seed for the deterministic shuffle.

    Returns:
        ``(training_poses, holdout_poses)``. Holdout poses are tagged
        ``is_holdout=True`` and training poses ``is_holdout=False``. The two
        sets are disjoint and deterministic for a given ``seed``.

    Raises:
        ValueError: If ``holdout_fraction`` is outside ``[0.0, 1.0]``.
    """
    if not 0.0 <= holdout_fraction <= 1.0:
        raise ValueError(f"holdout_fraction must be in [0.0, 1.0]; got {holdout_fraction!r}")
    count = math.floor(len(poses) * holdout_fraction)
    indices = list(range(len(poses)))
    random.Random(seed).shuffle(indices)
    holdout_indices = set(indices[:count])

    training: list[Pose] = []
    holdout: list[Pose] = []
    for i, pose in enumerate(poses):
        if i in holdout_indices:
            holdout.append(replace(pose, is_holdout=True))
        else:
            training.append(replace(pose, is_holdout=False))
    return training, holdout
