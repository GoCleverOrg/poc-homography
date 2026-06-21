"""Sequences the nine survey phases for one camera (C4 orchestration).

:func:`execute_survey` runs Phase 1 through Phase 9 in order, routing every
emitted C1 record to a :class:`PhaseSink`. It owns the cross-phase concerns the
individual executors cannot: generating the main FOV grid and partitioning it
into a training set (Phase 5) and a disjoint holdout set (Phase 9) via the C3
planner, accumulating the commanded states captured in phases 4-7, and asserting
the validation set is disjoint from them. It builds and returns the C1
:class:`SurveyRun` header for persistence compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus
from poc_homography.infrastructure.survey.capture_engine import SurveyCaptureEngine
from poc_homography.survey.phases import executors
from poc_homography.survey.planner.generators import fov_grid, partition_holdout
from poc_homography.survey.ptz_bracketing import preserve_ptz_position

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from poc_homography.domain.entities.survey.frame_record import CommandedState
    from poc_homography.domain.protocols.camera_device import CameraDevice
    from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig
    from poc_homography.domain.vo.tilt_envelope import TiltEnvelope
    from poc_homography.survey.phases.common import PhaseResult, PhaseSink
    from poc_homography.survey.planner.poses import Pose

# Clean-plate default snapshot-burst frame count (#276): a stable clean plate
# needs at least two frames per pose; five is a sensible default. Phases absent
# from a plan-config's ``burst_frame_count`` fall back to this.
DEFAULT_BURST_FRAME_COUNT = 5

# Minimum frames per pose for a usable clean-plate burst (#276).
MIN_BURST_FRAME_COUNT = 2

# Pose-based phases that capture a snapshot burst per pose (phase 8 uses video
# bursts; phase 1 is inventory and captures no frames).
_POSE_BURST_PHASES = (2, 3, 4, 5, 6, 7, 9)


@dataclass(frozen=True)
class SurveyPlan:
    """Per-camera survey parameters consumed by :func:`execute_survey`.

    These are the knobs C5 will later populate from config; the defaults here
    deliberately satisfy the DoD minimums (e.g. Phase 8: 3 poses x 3 zoom
    levels, 10 s, 5 fps) and keep the training and holdout sets disjoint by
    construction. Pan/tilt ranges are intentionally small so a full run is cheap.
    """

    burst_count: int = 1

    # Optional per-phase override of ``burst_count`` (phase number -> frames),
    # sourced from :class:`SurveyPlanConfig` via :meth:`from_plan_config`. A
    # phase absent here falls back to :attr:`burst_count`.
    phase_burst_counts: dict[int, int] = field(default_factory=dict)

    # Phase 2 — PTZ characterization.
    ptz_pan_range: tuple[float, float] = (0.0, 20.0)
    ptz_tilt_range: tuple[float, float] = (-20.0, 0.0)
    ptz_small_step: float = 5.0
    ptz_large_step: float = 20.0
    ptz_fixed_pan: float = 0.0
    ptz_fixed_tilt: float = -10.0
    ptz_fixed_zoom: float = 1.0
    ptz_repeat_count: int = 2

    # Phase 3 — zoom characterization.
    zoom_min: float = 1.0
    zoom_max: float = 16.0
    zoom_step: float = 5.0
    zoom_fixed_pan: float = 0.0
    zoom_fixed_tilt: float = -10.0

    # Phase 4 — dense nadir.
    nadir_pan: float = 180.0
    nadir_tilt: float = -85.0
    nadir_radius_deg: float = 3.0
    nadir_zoom: float = 8.0
    nadir_overlap_fraction: float = 0.8
    nadir_region_id: str = "nadir"

    # Phase 5 — main survey (and Phase 9 holdout source).
    main_pan_range: tuple[float, float] = (0.0, 30.0)
    main_tilt_range: tuple[float, float] = (-30.0, -10.0)
    main_zoom_levels: tuple[float, ...] = (2.0, 8.0)
    main_overlap_fraction: float = 0.5

    # Phase 6 — cross-zoom.
    cross_zoom_anchors: tuple[tuple[float, float], ...] = (
        (90.0, -20.0),
        (120.0, -25.0),
    )
    cross_zoom_levels: tuple[float, ...] = (2.0, 8.0, 16.0)

    # Phase 7 — repeatability.
    repeat_target_pan: float = 200.0
    repeat_target_tilt: float = -30.0
    repeat_target_zoom: float = 4.0
    repeat_approach_deltas: tuple[tuple[float, float], ...] = (
        (-5.0, 0.0),
        (5.0, 0.0),
        (0.0, -5.0),
    )

    # Phase 8 — static jitter.
    jitter_rtsp_url: str = ""
    jitter_poses: tuple[tuple[float, float], ...] = (
        (10.0, -15.0),
        (90.0, -20.0),
        (200.0, -30.0),
    )
    jitter_zoom_levels: tuple[float, ...] = (1.0, 8.0, 16.0)
    jitter_burst_duration_s: float = 10.0
    jitter_target_fps: float = 5.0

    # Phase 9 — validation holdout partition.
    holdout_fraction: float = 0.25
    holdout_seed: int = 261

    # Phase-0 horizon calibration (#275). When set, the main-survey FOV grid
    # skips sky tiles above the per-azimuth bound; ``None`` reproduces the
    # pre-horizon behaviour exactly.
    tilt_envelope: TiltEnvelope | None = None

    def burst_for(self, phase_number: int) -> int:
        """Return the per-pose snapshot-burst frame count for ``phase_number``.

        Falls back to :attr:`burst_count` when the phase is not in
        :attr:`phase_burst_counts`.
        """
        return self.phase_burst_counts.get(phase_number, self.burst_count)

    @classmethod
    def from_plan_config(
        cls,
        config: SurveyPlanConfig,
        *,
        default_burst_frame_count: int = DEFAULT_BURST_FRAME_COUNT,
    ) -> SurveyPlan:
        """Build a :class:`SurveyPlan` whose burst counts come from ``config``.

        Bridges the camera-free :class:`SurveyPlanConfig` sidecar onto the
        runner's in-memory plan. Each pose-based phase's per-pose snapshot-burst
        frame count is taken from ``config.burst_frame_count[phase]`` (falling
        back to ``default_burst_frame_count``) and clamped to at least
        :data:`MIN_BURST_FRAME_COUNT`, so every clean-plate capture emits two or
        more frames per pose. Only the burst counts are bridged here; the other
        plan knobs keep their DoD-satisfying defaults.

        Args:
            config: The reproducible plan-config sidecar.
            default_burst_frame_count: Burst frame count for pose-based phases
                absent from ``config.burst_frame_count``.

        Returns:
            A :class:`SurveyPlan` with per-phase burst counts populated.
        """
        default = max(MIN_BURST_FRAME_COUNT, int(default_burst_frame_count))
        phase_burst_counts = {
            phase: max(MIN_BURST_FRAME_COUNT, int(config.burst_frame_count.get(phase, default)))
            for phase in _POSE_BURST_PHASES
        }
        return cls(burst_count=default, phase_burst_counts=phase_burst_counts)


@dataclass(frozen=True)
class SurveyExecution:
    """The outcome of a full multi-phase run."""

    run: SurveyRun
    results: tuple[PhaseResult, ...] = field(default=())


def execute_survey(
    camera: CameraDevice,
    *,
    run_id: str,
    camera_id: str,
    sink: PhaseSink,
    base_output_dir: Path,
    plan: SurveyPlan | None = None,
    spec: CameraSpec = CameraSpec.HIKVISION_DS_2DF8425IX,
    clock: Callable[[], datetime] | None = None,
    uuid_factory: Callable[[], str] | None = None,
) -> SurveyExecution:
    """Run all nine survey phases for one camera, in order.

    Args:
        camera: The #256 ``CameraDevice`` to drive.
        run_id: Unique survey run id.
        camera_id: Stable camera identifier.
        sink: Destination for every emitted C1 record.
        base_output_dir: Root directory; each phase writes under a sub-directory
            named for its phase value.
        plan: Survey parameters; defaults to a DoD-satisfying :class:`SurveyPlan`.
        spec: Camera specification driving FOV-based pose spacing.
        clock: Returns timezone-aware UTC timestamps; injectable for tests.
        uuid_factory: Returns fresh id strings for bursts; injectable for tests.

    Returns:
        A :class:`SurveyExecution` with the C1 :class:`SurveyRun` header and the
        per-phase results.

    Raises:
        JitterTargetError: If Phase 8 is below the minimum jitter target.
        ValueError: If the Phase 9 holdout overlaps a captured training state.
    """
    plan = plan or SurveyPlan()
    now = clock or _utcnow
    engine = SurveyCaptureEngine(camera, clock=clock, uuid_factory=uuid_factory)
    started_at = now()

    training, holdout = _partition_main_grid(spec, plan)

    results: list[PhaseResult] = []
    captured_4_to_7: list[CommandedState] = []

    # Bracket the whole camera-driving sweep: snapshot the pre-sweep PTZ
    # position and restore it on normal completion, exception, or abort (#342).
    with preserve_ptz_position(camera):
        inventory = executors.run_inventory(camera, run_id=run_id, camera_id=camera_id, clock=now)
        results.append(inventory)

        results.append(
            executors.run_ptz_characterization(
                engine,
                run_id=run_id,
                camera_id=camera_id,
                output_dir=_phase_dir(base_output_dir, SurveyPhase.PTZ_CHARACTERIZATION),
                pan_range=plan.ptz_pan_range,
                tilt_range=plan.ptz_tilt_range,
                small_step=plan.ptz_small_step,
                large_step=plan.ptz_large_step,
                fixed_tilt=plan.ptz_fixed_tilt,
                fixed_pan=plan.ptz_fixed_pan,
                fixed_zoom=plan.ptz_fixed_zoom,
                repeat_count=plan.ptz_repeat_count,
                burst_count=plan.burst_for(2),
            )
        )

        results.append(
            executors.run_zoom_characterization(
                engine,
                run_id=run_id,
                camera_id=camera_id,
                output_dir=_phase_dir(base_output_dir, SurveyPhase.ZOOM_CHARACTERIZATION),
                zoom_min=plan.zoom_min,
                zoom_max=plan.zoom_max,
                zoom_step=plan.zoom_step,
                fixed_pan=plan.zoom_fixed_pan,
                fixed_tilt=plan.zoom_fixed_tilt,
                burst_count=plan.burst_for(3),
            )
        )

        nadir = executors.run_dense_nadir(
            engine,
            spec,
            run_id=run_id,
            camera_id=camera_id,
            output_dir=_phase_dir(base_output_dir, SurveyPhase.DENSE_NADIR),
            nadir_pan=plan.nadir_pan,
            nadir_tilt=plan.nadir_tilt,
            radius_deg=plan.nadir_radius_deg,
            zoom=plan.nadir_zoom,
            overlap_fraction=plan.nadir_overlap_fraction,
            region_id=plan.nadir_region_id,
            burst_count=plan.burst_for(4),
        )
        results.append(nadir)
        captured_4_to_7.extend(nadir.commanded_states)

        main = executors.run_main_survey(
            engine,
            training,
            run_id=run_id,
            camera_id=camera_id,
            output_dir=_phase_dir(base_output_dir, SurveyPhase.MAIN_SURVEY),
            burst_count=plan.burst_for(5),
        )
        results.append(main)
        captured_4_to_7.extend(main.commanded_states)

        cross = executors.run_cross_zoom(
            engine,
            run_id=run_id,
            camera_id=camera_id,
            output_dir=_phase_dir(base_output_dir, SurveyPhase.CROSS_ZOOM),
            anchors=plan.cross_zoom_anchors,
            zoom_levels=plan.cross_zoom_levels,
            burst_count=plan.burst_for(6),
        )
        results.append(cross)
        captured_4_to_7.extend(cross.commanded_states)

        repeat = executors.run_repeatability(
            engine,
            run_id=run_id,
            camera_id=camera_id,
            output_dir=_phase_dir(base_output_dir, SurveyPhase.REPEATABILITY),
            target_pan=plan.repeat_target_pan,
            target_tilt=plan.repeat_target_tilt,
            target_zoom=plan.repeat_target_zoom,
            approach_deltas=plan.repeat_approach_deltas,
            burst_count=plan.burst_for(7),
        )
        results.append(repeat)
        captured_4_to_7.extend(repeat.commanded_states)

        results.append(
            executors.run_jitter(
                engine,
                run_id=run_id,
                camera_id=camera_id,
                output_dir=_phase_dir(base_output_dir, SurveyPhase.STATIC_JITTER),
                rtsp_url=plan.jitter_rtsp_url,
                poses=plan.jitter_poses,
                zoom_levels=plan.jitter_zoom_levels,
                burst_duration_s=plan.jitter_burst_duration_s,
                target_fps=plan.jitter_target_fps,
            )
        )

        results.append(
            executors.run_validation(
                engine,
                holdout,
                captured_4_to_7,
                run_id=run_id,
                camera_id=camera_id,
                output_dir=_phase_dir(base_output_dir, SurveyPhase.VALIDATION),
                burst_count=plan.burst_for(9),
            )
        )

    _emit(results, sink)
    _assert_no_cross_tagging(results)

    run = SurveyRun(
        run_id=run_id,
        camera_id=camera_id,
        phases=frozenset(result.phase for result in results),
        started_at=started_at,
        finished_at=now(),
        status=SurveyRunStatus.COMPLETED,
    )
    return SurveyExecution(run=run, results=tuple(results))


# -- Internal helpers -------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def _phase_dir(base: Path, phase: SurveyPhase) -> Path:
    """Return the per-phase output sub-directory."""
    return base / phase.value


def _partition_main_grid(spec: CameraSpec, plan: SurveyPlan) -> tuple[list[Pose], list[Pose]]:
    """Build the main FOV grid and split it into training and holdout sets.

    When ``plan.tilt_envelope`` is set (Phase-0 calibration ran), sky tiles
    above the per-azimuth bound are skipped; otherwise the grid is unchanged.
    """
    grid = fov_grid(
        spec,
        plan.main_pan_range,
        plan.main_tilt_range,
        plan.main_zoom_levels,
        plan.main_overlap_fraction,
        tilt_envelope=plan.tilt_envelope,
    )
    training, holdout = partition_holdout(grid, plan.holdout_fraction, plan.holdout_seed)
    return training, holdout


def _emit(results: list[PhaseResult], sink: PhaseSink) -> None:
    """Route every record produced by every phase to the sink."""
    for result in results:
        if result.inventory is not None:
            sink.save_inventory(result.inventory)
        for burst in result.bursts:
            sink.save_burst(burst)
        for frame in result.frames:
            sink.save_frame(frame)


def _assert_no_cross_tagging(results: list[PhaseResult]) -> None:
    """Assert no phase emitted a record tagged with another phase's identifier."""
    for result in results:
        for frame in result.frames:
            if frame.capture.phase is not result.phase:
                raise ValueError(
                    f"phase {result.phase.value} emitted a frame tagged {frame.capture.phase.value}"
                )
        for burst in result.bursts:
            if burst.phase is not result.phase:
                raise ValueError(
                    f"phase {result.phase.value} emitted a burst tagged {burst.phase.value}"
                )
        if result.inventory is not None and result.inventory.phase is not result.phase:
            raise ValueError(
                f"phase {result.phase.value} emitted an inventory record tagged "
                f"{result.inventory.phase.value}"
            )
