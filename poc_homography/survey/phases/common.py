"""Shared building blocks for the nine survey phases (C4).

This module holds the persistence port (:class:`PhaseSink`), the per-phase
result container (:class:`PhaseResult`), the pose -> commanded-state adapter,
and :func:`capture_pose_sequence` — the one helper every pose-based phase reuses
to drive the C2 engine over an ordered pose list while stamping the C3-derived
survey context onto each emitted frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

from poc_homography.domain.entities.survey.frame_record import (
    CommandedState,
    FrameRecord,
    SurveyContext,
)
from poc_homography.infrastructure.survey.capture_engine import CaptureContext
from poc_homography.survey.planner.poses import canonical_pose_key
from poc_homography.types import Degrees, Unitless

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from poc_homography.domain.entities.survey.inventory_record import (
        CameraInventoryRecord,
    )
    from poc_homography.domain.entities.survey.video_burst_record import (
        VideoBurstRecord,
    )
    from poc_homography.domain.enums.survey_phase import SurveyPhase
    from poc_homography.infrastructure.survey.capture_engine import (
        SurveyCaptureEngine,
    )
    from poc_homography.survey.planner.poses import Pose


class PhaseSink(Protocol):
    """Destination for the C1 records a phase emits.

    The phase layer is persistence-agnostic: it hands every record to a sink
    and never writes files or rows itself. Implementations may persist to YAML,
    Postgres, or accumulate in memory (tests).
    """

    def save_inventory(self, record: CameraInventoryRecord) -> None:
        """Persist the Phase 1 camera-inventory record."""
        ...

    def save_frame(self, record: FrameRecord) -> None:
        """Persist a single per-frame C1 record."""
        ...

    def save_burst(self, record: VideoBurstRecord) -> None:
        """Persist a Phase 8 video-burst record (segment + frame refs)."""
        ...


@dataclass(frozen=True)
class PhaseResult:
    """The C1 records produced by one phase.

    Exactly one of the populated collections is non-empty for a given phase:
    Phase 1 sets :attr:`inventory`; Phase 8 sets :attr:`bursts` (and the frames
    extracted from them); every other phase sets :attr:`frames`.
    """

    phase: SurveyPhase
    frames: tuple[FrameRecord, ...] = ()
    bursts: tuple[VideoBurstRecord, ...] = ()
    inventory: CameraInventoryRecord | None = None
    commanded_states: tuple[CommandedState, ...] = field(default=())


def pose_to_commanded(pose: Pose) -> CommandedState:
    """Adapt a planner :class:`Pose` to a C2 :class:`CommandedState`.

    Planner poses carry no focus target, so ``commanded_focus`` is ``None``
    (the engine then leaves focus untouched).
    """
    return CommandedState(
        commanded_pan=Degrees(float(pose.pan)),
        commanded_tilt=Degrees(float(pose.tilt)),
        commanded_zoom=Unitless(float(pose.zoom)),
        commanded_focus=None,
    )


def capture_pose_sequence(
    engine: SurveyCaptureEngine,
    poses: Sequence[Pose],
    *,
    run_id: str,
    camera_id: str,
    phase: SurveyPhase,
    output_dir: Path,
    burst_count: int = 1,
    is_repeatability_sequence: bool = False,
    survey_context_for: Callable[[Pose, int], SurveyContext] | None = None,
) -> list[FrameRecord]:
    """Drive the C2 engine over an ordered pose list, emitting tagged frames.

    Threads each pose's commanded state into the next pose's
    :class:`CaptureContext` so the engine can compute per-axis movement
    directions; threading starts fresh (no previous pose) at the first pose so a
    phase's sweep is self-contained and not contaminated by a prior phase. Every
    frame is tagged with ``phase`` (via the engine), with a stable
    ``pose_id`` derived from the pose geometry via
    :func:`~poc_homography.survey.planner.poses.canonical_pose_key` (so the same
    physical pose yields the same id across runs), and, when
    ``survey_context_for`` is supplied, enriched with the C3-derived
    :class:`SurveyContext` (region id / approach direction / sequence index).

    Args:
        engine: The C2 capture engine.
        poses: Ordered poses to capture, in execution order.
        run_id: Owning survey run id.
        camera_id: Stable camera identifier.
        phase: The phase identity stamped on every emitted record.
        output_dir: Directory the engine writes frame images into.
        burst_count: Snapshot frames per pose (``>= 1``).
        is_repeatability_sequence: Recorded verbatim on each frame's movement
            context; never computed here.
        survey_context_for: Optional factory returning the :class:`SurveyContext`
            to stamp on the frames captured at ``(pose, index)``.

    Returns:
        All emitted :class:`FrameRecord` objects, in capture order.
    """
    records: list[FrameRecord] = []
    previous: CommandedState | None = None
    for index, pose in enumerate(poses):
        commanded = pose_to_commanded(pose)
        context = CaptureContext(
            run_id=run_id,
            camera_id=camera_id,
            phase=phase,
            previous_commanded=previous,
            is_repeatability_sequence=is_repeatability_sequence,
        )
        frames = engine.capture_snapshot_burst(
            commanded,
            context,
            burst_count=burst_count,
            output_dir=output_dir,
        )
        base_context = survey_context_for(pose, index) if survey_context_for is not None else None
        survey_context = _stamp_pose_id(base_context, pose)
        frames = [replace(frame, survey_context=survey_context) for frame in frames]
        records.extend(frames)
        previous = commanded
    return records


def _stamp_pose_id(base: SurveyContext | None, pose: Pose) -> SurveyContext:
    """Return ``base`` (or a fresh context) with a stable ``pose_id`` stamped.

    The id is derived purely from the pose geometry via
    :func:`~poc_homography.survey.planner.poses.canonical_pose_key`, so the same
    physical ``(pan, tilt, zoom)`` yields the same id on every run regardless of
    insertion order, randomness, or time.
    """
    pose_id = canonical_pose_key(float(pose.pan), float(pose.tilt), float(pose.zoom))
    if base is None:
        return SurveyContext(pose_id=pose_id)
    return replace(base, pose_id=pose_id)
