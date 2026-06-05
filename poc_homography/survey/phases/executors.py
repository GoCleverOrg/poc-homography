"""The nine concrete survey-phase executors (C4).

Every executor is a thin binding: it asks the C3 planner for poses (or a sweep),
drives the C2 engine to capture them, and returns a :class:`PhaseResult` whose
records all carry this phase's tag (and, where the DoD requires it, the
planner-derived :class:`SurveyContext`). None of them re-implement pose
generation, movement mechanics, or persistence.

The phase identifiers are the C1 :class:`SurveyPhase` enum values — the schema's
source of truth — so e.g. Phase 1 emits ``phase=camera_inventory``, Phase 5
``phase=main_survey``, and Phase 8 ``phase=static_jitter`` (the issue's informal
``inventory`` / ``main_ptz_survey`` / ``jitter`` names map onto these).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.domain.entities.survey.frame_record import SurveyContext
from poc_homography.domain.entities.survey.inventory_record import (
    CameraInventoryRecord,
)
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.infrastructure.survey.capture_engine import (
    CaptureContext,
    CaptureEngineError,
)
from poc_homography.survey.phases.common import (
    PhaseResult,
    capture_pose_sequence,
    pose_to_commanded,
)
from poc_homography.survey.planner.generators import (
    SweepAxis,
    cross_zoom,
    directional_sweep,
    nadir_region,
    repeatability_sequences,
)
from poc_homography.survey.planner.poses import ApproachDirection, Pose
from poc_homography.types import Degrees, Unitless

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime
    from pathlib import Path

    from poc_homography.domain.entities.survey.frame_record import (
        CommandedState,
        FrameRecord,
    )
    from poc_homography.domain.entities.survey.video_burst_record import (
        VideoBurstRecord,
    )
    from poc_homography.domain.enums.camera_spec import CameraSpec
    from poc_homography.domain.protocols.camera_device import CameraDevice
    from poc_homography.infrastructure.survey.capture_engine import (
        SurveyCaptureEngine,
    )

# Phase 8 (static jitter) minimum acceptable target, per the verbatim spec / DoD.
JITTER_MIN_DURATION_S = 10.0
JITTER_MIN_FPS = 5.0
JITTER_MIN_ZOOM_LEVELS = 3
JITTER_MIN_POSES = 3

# Approach-direction labels stamped onto Phase 3 zoom-sweep frames so the two
# directional passes are distinguishable beyond the per-axis ``direction_zoom``.
_ZOOM_LABEL = {
    ApproachDirection.ASCENDING: "wide_to_tele",
    ApproachDirection.DESCENDING: "tele_to_wide",
}


class JitterTargetError(ValueError):
    """Raised when a Phase 8 configuration is below the minimum jitter target."""


def run_inventory(
    camera: CameraDevice,
    *,
    run_id: str,
    camera_id: str,
    clock: Callable[[], datetime],
) -> PhaseResult:
    """Phase 1 — camera inventory survey.

    Collects the camera's full self-report in a single pass via the #256 adapter
    and emits exactly one :class:`CameraInventoryRecord` tagged
    ``phase=camera_inventory``. This is the only phase that does not perform
    pose-based image capture.

    Args:
        camera: The #256 ``CameraDevice`` to interrogate.
        run_id: Owning survey run id.
        camera_id: Stable camera identifier.
        clock: Returns the timezone-aware capture timestamp.

    Returns:
        A :class:`PhaseResult` whose :attr:`PhaseResult.inventory` is set.
    """
    record = CameraInventoryRecord(
        record_id=f"{run_id}_inventory",
        run_id=run_id,
        camera_id=camera_id,
        captured_at=clock(),
        device_info=camera.get_device_info(),
        capabilities=camera.get_capabilities(),
        optics=camera.get_optics(),
        health=camera.get_health(),
        stream_profiles=tuple(camera.get_stream_profiles()),
        presets=tuple(camera.list_presets()),
    )
    return PhaseResult(phase=SurveyPhase.CAMERA_INVENTORY, inventory=record)


def run_ptz_characterization(
    engine: SurveyCaptureEngine,
    *,
    run_id: str,
    camera_id: str,
    output_dir: Path,
    pan_range: tuple[float, float],
    tilt_range: tuple[float, float],
    small_step: float,
    large_step: float,
    fixed_tilt: float,
    fixed_pan: float,
    fixed_zoom: float,
    repeat_count: int = 2,
    burst_count: int = 1,
) -> PhaseResult:
    """Phase 2 — PTZ axis characterization.

    Executes small, large, and repeated pan and tilt sweeps in both directions
    so offline analysis can recover precision, rounding, latency, settling, and
    backlash. Each pose emits a frame tagged ``phase=ptz_characterization`` with
    per-axis movement direction and movement context populated by C2.

    Args:
        engine: The C2 capture engine.
        run_id: Owning survey run id.
        camera_id: Stable camera identifier.
        output_dir: Directory for this phase's frame images.
        pan_range: ``(min, max)`` pan bounds in degrees.
        tilt_range: ``(min, max)`` tilt bounds in degrees.
        small_step: Fine step size (degrees) for small moves.
        large_step: Coarse step size (degrees) for large moves.
        fixed_tilt: Tilt held constant while sweeping pan.
        fixed_pan: Pan held constant while sweeping tilt.
        fixed_zoom: Zoom held constant for the whole phase.
        repeat_count: How many times each sweep family is repeated.
        burst_count: Snapshot frames per pose.

    Returns:
        A :class:`PhaseResult` with the emitted frames and their commanded
        states.
    """
    phase = SurveyPhase.PTZ_CHARACTERIZATION
    records: list[FrameRecord] = []
    for _ in range(max(1, repeat_count)):
        for step in (small_step, large_step):
            pan_asc, pan_desc = directional_sweep(
                SweepAxis.PAN, pan_range[0], pan_range[1], step, fixed_tilt, fixed_zoom, phase=phase
            )
            tilt_asc, tilt_desc = directional_sweep(
                SweepAxis.TILT,
                tilt_range[0],
                tilt_range[1],
                step,
                fixed_pan,
                fixed_zoom,
                phase=phase,
            )
            for plan in (pan_asc, pan_desc, tilt_asc, tilt_desc):
                records.extend(
                    capture_pose_sequence(
                        engine,
                        plan.poses,
                        run_id=run_id,
                        camera_id=camera_id,
                        phase=phase,
                        output_dir=output_dir,
                        burst_count=burst_count,
                        survey_context_for=_approach_context(plan.approach_directions),
                    )
                )
    return PhaseResult(phase=phase, frames=tuple(records), commanded_states=_commanded(records))


def run_zoom_characterization(
    engine: SurveyCaptureEngine,
    *,
    run_id: str,
    camera_id: str,
    output_dir: Path,
    zoom_min: float,
    zoom_max: float,
    zoom_step: float,
    fixed_pan: float,
    fixed_tilt: float,
    burst_count: int = 1,
) -> PhaseResult:
    """Phase 3 — zoom characterization.

    Sweeps the full zoom range wide-to-tele then tele-to-wide as two separate,
    non-interleaved monotonic passes. Each frame carries ``phase=
    zoom_characterization`` plus an ``approach_direction`` of ``wide_to_tele`` or
    ``tele_to_wide`` so the passes (and hysteresis) are distinguishable offline.

    Args:
        engine: The C2 capture engine.
        run_id: Owning survey run id.
        camera_id: Stable camera identifier.
        output_dir: Directory for this phase's frame images.
        zoom_min: Wide end of the zoom range.
        zoom_max: Tele end of the zoom range.
        zoom_step: Positive zoom step between consecutive poses.
        fixed_pan: Pan held constant for the whole phase.
        fixed_tilt: Tilt held constant for the whole phase.
        burst_count: Snapshot frames per pose.

    Returns:
        A :class:`PhaseResult` with both directional passes' frames.
    """
    phase = SurveyPhase.ZOOM_CHARACTERIZATION
    wide_to_tele, tele_to_wide = directional_sweep(
        SweepAxis.ZOOM, zoom_min, zoom_max, zoom_step, fixed_pan, fixed_tilt, phase=phase
    )
    records: list[FrameRecord] = []
    for plan in (wide_to_tele, tele_to_wide):
        label = _ZOOM_LABEL[plan.approach_directions[0]]
        records.extend(
            capture_pose_sequence(
                engine,
                plan.poses,
                run_id=run_id,
                camera_id=camera_id,
                phase=phase,
                output_dir=output_dir,
                burst_count=burst_count,
                survey_context_for=lambda _pose, _i, _label=label: SurveyContext(
                    approach_direction=_label
                ),
            )
        )
    return PhaseResult(phase=phase, frames=tuple(records), commanded_states=_commanded(records))


def run_dense_nadir(
    engine: SurveyCaptureEngine,
    spec: CameraSpec,
    *,
    run_id: str,
    camera_id: str,
    output_dir: Path,
    nadir_pan: float,
    nadir_tilt: float,
    radius_deg: float,
    zoom: float,
    overlap_fraction: float,
    region_id: str = "nadir",
    burst_count: int = 1,
) -> PhaseResult:
    """Phase 4 — dense nadir survey.

    Captures a dense, high-overlap disc of poses around the downward-looking
    region (~18 m AGL). Every frame is tagged ``phase=dense_nadir`` and carries
    the nadir ``region_id``.

    Args:
        engine: The C2 capture engine.
        spec: Camera specification (drives FOV-based overlap spacing).
        run_id: Owning survey run id.
        camera_id: Stable camera identifier.
        output_dir: Directory for this phase's frame images.
        nadir_pan: Anchor pan of the downward-looking region.
        nadir_tilt: Anchor tilt of the downward-looking region.
        radius_deg: Disc radius in degrees.
        zoom: Single zoom level for the nadir grid.
        overlap_fraction: Desired fractional overlap in ``[0.0, 1.0)``.
        region_id: Identifier stamped on every nadir frame.
        burst_count: Snapshot frames per pose.

    Returns:
        A :class:`PhaseResult` with the nadir frames.
    """
    phase = SurveyPhase.DENSE_NADIR
    poses = nadir_region(
        spec, nadir_pan, nadir_tilt, radius_deg, zoom, overlap_fraction, phase=phase
    )
    records = capture_pose_sequence(
        engine,
        poses,
        run_id=run_id,
        camera_id=camera_id,
        phase=phase,
        output_dir=output_dir,
        burst_count=burst_count,
        survey_context_for=lambda _pose, _i: SurveyContext(region_id=region_id),
    )
    return PhaseResult(phase=phase, frames=tuple(records), commanded_states=_commanded(records))


def run_main_survey(
    engine: SurveyCaptureEngine,
    poses: Sequence[Pose],
    *,
    run_id: str,
    camera_id: str,
    output_dir: Path,
    burst_count: int = 1,
) -> PhaseResult:
    """Phase 5 — main pan/tilt/zoom survey.

    Captures a dense, highly-overlapping visual graph across the operational
    pan/tilt area at the configured zoom levels. The pose set is the training
    partition produced by the runner (FOV grid minus the Phase 9 holdout), so
    every frame is tagged ``phase=main_survey``.

    Args:
        engine: The C2 capture engine.
        poses: The training poses to capture (runner-supplied).
        run_id: Owning survey run id.
        camera_id: Stable camera identifier.
        output_dir: Directory for this phase's frame images.
        burst_count: Snapshot frames per pose.

    Returns:
        A :class:`PhaseResult` with the main-survey frames.
    """
    phase = SurveyPhase.MAIN_SURVEY
    records = capture_pose_sequence(
        engine,
        poses,
        run_id=run_id,
        camera_id=camera_id,
        phase=phase,
        output_dir=output_dir,
        burst_count=burst_count,
    )
    return PhaseResult(phase=phase, frames=tuple(records), commanded_states=_commanded(records))


def run_cross_zoom(
    engine: SurveyCaptureEngine,
    *,
    run_id: str,
    camera_id: str,
    output_dir: Path,
    anchors: Sequence[tuple[float, float]],
    zoom_levels: Sequence[float],
    burst_count: int = 1,
) -> PhaseResult:
    """Phase 6 — cross-zoom survey.

    Captures each ground region at multiple zoom levels. All frames observing
    one region share a stable ``region_id`` (determined before capture by the
    C3 generator) so cross-scale observations can be connected offline. Frames
    are tagged ``phase=cross_zoom``.

    Args:
        engine: The C2 capture engine.
        run_id: Owning survey run id.
        camera_id: Stable camera identifier.
        output_dir: Directory for this phase's frame images.
        anchors: ``(pan, tilt)`` region anchors, in order.
        zoom_levels: Zoom levels to revisit each anchor at.
        burst_count: Snapshot frames per pose.

    Returns:
        A :class:`PhaseResult` with the cross-zoom frames.
    """
    phase = SurveyPhase.CROSS_ZOOM
    poses = cross_zoom(anchors, zoom_levels, phase=phase)
    records = capture_pose_sequence(
        engine,
        poses,
        run_id=run_id,
        camera_id=camera_id,
        phase=phase,
        output_dir=output_dir,
        burst_count=burst_count,
        survey_context_for=lambda pose, _i: SurveyContext(region_id=pose.region_id),
    )
    return PhaseResult(phase=phase, frames=tuple(records), commanded_states=_commanded(records))


def run_repeatability(
    engine: SurveyCaptureEngine,
    *,
    run_id: str,
    camera_id: str,
    output_dir: Path,
    target_pan: float,
    target_tilt: float,
    target_zoom: float,
    approach_deltas: Sequence[tuple[float, float]],
    burst_count: int = 1,
) -> PhaseResult:
    """Phase 7 — repeatability survey.

    Visits the same commanded target pose multiple times, approaching it from a
    different direction each time. Each visit's target frame carries
    ``phase=repeatability``, the approach-direction label, and a sequence index
    within the repeat group; ``is_repeatability_sequence`` is set on the C2
    movement context.

    Args:
        engine: The C2 capture engine.
        run_id: Owning survey run id.
        camera_id: Stable camera identifier.
        output_dir: Directory for this phase's frame images.
        target_pan: Target pan in degrees.
        target_tilt: Target tilt in degrees.
        target_zoom: Target zoom factor.
        approach_deltas: ``(pan_offset, tilt_offset)`` approach offsets; at
            least two distinct directions.
        burst_count: Snapshot frames per pose.

    Returns:
        A :class:`PhaseResult` with the repeatability frames.
    """
    phase = SurveyPhase.REPEATABILITY
    sequences = repeatability_sequences(
        target_pan, target_tilt, target_zoom, approach_deltas, phase=phase
    )
    records: list[FrameRecord] = []
    for visit_index, (sub_sequence, (pan_offset, tilt_offset)) in enumerate(
        zip(sequences, approach_deltas)
    ):
        label = _approach_label(pan_offset, tilt_offset)
        records.extend(
            capture_pose_sequence(
                engine,
                sub_sequence,
                run_id=run_id,
                camera_id=camera_id,
                phase=phase,
                output_dir=output_dir,
                burst_count=burst_count,
                is_repeatability_sequence=True,
                survey_context_for=lambda _pose, _i, _label=label, _seq=visit_index: SurveyContext(
                    approach_direction=_label, sequence_index=_seq
                ),
            )
        )
    return PhaseResult(phase=phase, frames=tuple(records), commanded_states=_commanded(records))


def run_jitter(
    engine: SurveyCaptureEngine,
    *,
    run_id: str,
    camera_id: str,
    output_dir: Path,
    rtsp_url: str,
    poses: Sequence[tuple[float, float]],
    zoom_levels: Sequence[float],
    burst_duration_s: float = JITTER_MIN_DURATION_S,
    target_fps: float = JITTER_MIN_FPS,
) -> PhaseResult:
    """Phase 8 — static jitter survey (RTSP/video bursts).

    For each fixed (pose, zoom-level) combination, records an RTSP video segment
    and extracts its frames via the C2 video-burst path. The camera is never
    commanded between burst start and end (the engine guarantees this). Both the
    original segment and addressable per-frame references are persisted; frames
    and bursts are tagged ``phase=static_jitter``.

    Args:
        engine: The C2 capture engine.
        run_id: Owning survey run id.
        camera_id: Stable camera identifier.
        output_dir: Directory for segment files and extracted frames.
        rtsp_url: The RTSP stream URL to record.
        poses: Fixed ``(pan, tilt)`` poses; at least three.
        zoom_levels: Fixed zoom levels; at least three.
        burst_duration_s: Target seconds per burst; at least ten.
        target_fps: Target frame rate; at least five.

    Returns:
        A :class:`PhaseResult` with the bursts and their extracted frames.

    Raises:
        JitterTargetError: If the configuration is below the minimum jitter
            target (10 s, 5 fps, 3 zoom levels, 3 poses).
        CaptureEngineError: If a burst produces no segment or no frames.
    """
    _assert_jitter_target(poses, zoom_levels, burst_duration_s, target_fps)
    phase = SurveyPhase.STATIC_JITTER
    bursts = []
    frames: list[FrameRecord] = []
    states: list[CommandedState] = []
    for pan, tilt in poses:
        for zoom in zoom_levels:
            commanded = pose_to_commanded(Pose(pan=_deg(pan), tilt=_deg(tilt), zoom=_unit(zoom)))
            context = _jitter_context(run_id, camera_id, phase)
            burst, burst_frames = engine.capture_video_burst(
                commanded,
                context,
                rtsp_url,
                burst_duration_s=burst_duration_s,
                output_dir=output_dir,
                target_fps=target_fps,
            )
            _assert_burst_complete(burst)
            bursts.append(burst)
            frames.extend(burst_frames)
            states.append(commanded)
    return PhaseResult(
        phase=phase,
        frames=tuple(frames),
        bursts=tuple(bursts),
        commanded_states=tuple(states),
    )


def run_validation(
    engine: SurveyCaptureEngine,
    holdout_poses: Sequence[Pose],
    captured_states: Sequence[CommandedState],
    *,
    run_id: str,
    camera_id: str,
    output_dir: Path,
    burst_count: int = 1,
) -> PhaseResult:
    """Phase 9 — validation survey (held-out, disjoint pose set).

    Captures the C3 holdout pose set and emits frames tagged
    ``phase=validation``. Before any capture, asserts that no holdout commanded
    state (pan, tilt, zoom) coincides with a state already captured in phases
    4-7, guaranteeing the validation set is disjoint from the training poses.

    Args:
        engine: The C2 capture engine.
        holdout_poses: The C3 holdout partition (runner-supplied).
        captured_states: Commanded states captured in phases 4-7.
        run_id: Owning survey run id.
        camera_id: Stable camera identifier.
        output_dir: Directory for this phase's frame images.
        burst_count: Snapshot frames per pose.

    Returns:
        A :class:`PhaseResult` with the validation frames.

    Raises:
        ValueError: If any holdout pose overlaps a captured training state.
    """
    phase = SurveyPhase.VALIDATION
    _assert_disjoint(holdout_poses, captured_states)
    records = capture_pose_sequence(
        engine,
        holdout_poses,
        run_id=run_id,
        camera_id=camera_id,
        phase=phase,
        output_dir=output_dir,
        burst_count=burst_count,
    )
    return PhaseResult(phase=phase, frames=tuple(records), commanded_states=_commanded(records))


# -- Internal helpers -------------------------------------------------------


def _deg(value: float) -> Degrees:
    return Degrees(float(value))


def _unit(value: float) -> Unitless:
    return Unitless(float(value))


def _commanded(records: Sequence[FrameRecord]) -> tuple[CommandedState, ...]:
    """Collect the distinct commanded states across a phase's frames."""
    seen: dict[tuple[float, float, float], CommandedState] = {}
    for record in records:
        seen[_state_key(record.commanded)] = record.commanded
    return tuple(seen.values())


def _state_key(state: CommandedState) -> tuple[float, float, float]:
    """A hashable (pan, tilt, zoom) key for commanded-state set membership."""
    return (
        float(state.commanded_pan),
        float(state.commanded_tilt),
        float(state.commanded_zoom),
    )


def _approach_context(
    approach_directions: tuple[ApproachDirection, ...],
) -> Callable[[Pose, int], SurveyContext]:
    """Build a context factory stamping the sweep's approach direction."""
    label = approach_directions[0].value if approach_directions else None
    return lambda _pose, _i: SurveyContext(approach_direction=label)


def _approach_label(pan_offset: float, tilt_offset: float) -> str:
    """Human-readable approach-direction label from a repeatability offset."""
    parts: list[str] = []
    if pan_offset < 0:
        parts.append("from_lower_pan")
    elif pan_offset > 0:
        parts.append("from_higher_pan")
    if tilt_offset < 0:
        parts.append("from_lower_tilt")
    elif tilt_offset > 0:
        parts.append("from_higher_tilt")
    return "+".join(parts) if parts else "from_target"


def _assert_jitter_target(
    poses: Sequence[tuple[float, float]],
    zoom_levels: Sequence[float],
    burst_duration_s: float,
    target_fps: float,
) -> None:
    """Reject Phase 8 configurations below the minimum jitter target."""
    if len(poses) < JITTER_MIN_POSES:
        raise JitterTargetError(f"jitter requires >= {JITTER_MIN_POSES} poses, got {len(poses)}")
    if len(zoom_levels) < JITTER_MIN_ZOOM_LEVELS:
        raise JitterTargetError(
            f"jitter requires >= {JITTER_MIN_ZOOM_LEVELS} zoom levels, got {len(zoom_levels)}"
        )
    if burst_duration_s < JITTER_MIN_DURATION_S:
        raise JitterTargetError(
            f"jitter burst duration must be >= {JITTER_MIN_DURATION_S}s, got {burst_duration_s}"
        )
    if target_fps < JITTER_MIN_FPS:
        raise JitterTargetError(f"jitter fps must be >= {JITTER_MIN_FPS}, got {target_fps}")


def _assert_burst_complete(burst: VideoBurstRecord) -> None:
    """Confirm a burst meets the jitter target and persisted its data.

    Validates the *achieved* burst, not just the requested configuration: the
    C2 engine stops recording early when the RTSP stream ends, so a burst can
    come back shorter than the requested ``burst_duration_s``. The DoD requires
    every burst to *meet* the minimum target (10 s, 5 fps) with both the segment
    and addressable per-frame references stored, so a short or empty burst fails
    loudly here rather than being silently marked complete.

    Raises:
        CaptureEngineError: If the segment path or frame references are missing.
        JitterTargetError: If the realized duration or stream fps is below the
            minimum jitter target.
    """
    if not burst.frame_refs:
        raise CaptureEngineError(f"jitter burst {burst.burst_id} has no frame references")
    if str(burst.segment_path) == "":
        raise CaptureEngineError(f"jitter burst {burst.burst_id} has no segment path")
    if float(burst.duration_s) < JITTER_MIN_DURATION_S:
        raise JitterTargetError(
            f"jitter burst {burst.burst_id} achieved {float(burst.duration_s):.2f}s, "
            f"below the {JITTER_MIN_DURATION_S}s minimum"
        )
    if float(burst.fps) < JITTER_MIN_FPS:
        raise JitterTargetError(
            f"jitter burst {burst.burst_id} fps {float(burst.fps):.2f} is below the "
            f"{JITTER_MIN_FPS} minimum"
        )


def _assert_disjoint(
    holdout_poses: Sequence[Pose],
    captured_states: Sequence[CommandedState],
) -> None:
    """Assert no holdout pose shares a commanded state with phases 4-7."""
    captured = {_state_key(state) for state in captured_states}
    overlap = [pose for pose in holdout_poses if _state_key(pose_to_commanded(pose)) in captured]
    if overlap:
        raise ValueError(
            f"validation set overlaps {len(overlap)} training commanded state(s); "
            "holdout must be disjoint from phases 4-7"
        )


def _jitter_context(run_id: str, camera_id: str, phase: SurveyPhase) -> CaptureContext:
    """Build the per-burst capture context for the jitter phase."""
    return CaptureContext(run_id=run_id, camera_id=camera_id, phase=phase)
