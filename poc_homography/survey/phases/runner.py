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

import logging
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

# Phase number of the main survey (the ground-spanning FOV-grid sweep). Its
# pan/tilt extent and zoom levels are sourced from the plan-config sidecar.
_MAIN_SURVEY_PHASE = 5

# Every phase number (1..9). A :class:`SurveyPlan` with the default
# :attr:`~SurveyPlan.enabled_phases` runs every one; the execution order itself
# is fixed by the sequence of phase blocks in :func:`execute_survey`.
_ALL_PHASE_NUMBERS = frozenset(range(1, 10))

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurveyPlan:
    """Per-camera survey parameters consumed by :func:`execute_survey`.

    These are the knobs C5 will later populate from config; the defaults here
    deliberately satisfy the DoD minimums (e.g. Phase 8: 3 poses x 3 zoom
    levels, 10 s, 5 fps) and keep the training and holdout sets disjoint by
    construction. Pan/tilt ranges are intentionally small so a full run is cheap.
    """

    burst_count: int = 1

    # Survey phases (1..9, the :class:`SurveyPhase` numbering) to execute; a
    # phase absent here is skipped entirely by :func:`execute_survey`. Defaults
    # to all nine and is sourced from :attr:`SurveyPlanConfig.enabled_phases`
    # via :meth:`from_plan_config` (#372).
    enabled_phases: frozenset[int] = _ALL_PHASE_NUMBERS

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

    def is_enabled(self, phase_number: int) -> bool:
        """Return whether ``phase_number`` (1..9) is in :attr:`enabled_phases`."""
        return phase_number in self.enabled_phases

    @classmethod
    def from_plan_config(
        cls,
        config: SurveyPlanConfig,
        *,
        default_burst_frame_count: int = DEFAULT_BURST_FRAME_COUNT,
    ) -> SurveyPlan:
        """Build a :class:`SurveyPlan` from the ``config`` sidecar.

        Bridges the camera-free :class:`SurveyPlanConfig` sidecar onto the
        runner's in-memory plan:

        * **Enabled phases** (#372) — ``config.enabled_phases`` is forwarded so
          :func:`execute_survey` runs only the configured subset of phases
          (e.g. ``{1, 5}`` runs inventory + main survey alone). The default
          (all nine) reproduces the pre-gating behaviour.
        * **Burst counts** — each pose-based phase's per-pose snapshot-burst
          frame count is taken from ``config.burst_frame_count[phase]`` (falling
          back to ``default_burst_frame_count``) and clamped to at least
          :data:`MIN_BURST_FRAME_COUNT`, so every clean-plate capture emits two
          or more frames per pose.
        * **Main-survey ground span** (#343) — when the config configures the
          main survey (phase 5) via ``phase_pan_range``/``phase_tilt_range``,
          the full main-survey FOV-grid parameter set is adopted from the
          sidecar atomically: ``phase_pan_range[5]``/``phase_tilt_range[5]``
          become :attr:`main_pan_range`/:attr:`main_tilt_range`,
          ``config.zoom_levels`` becomes :attr:`main_zoom_levels`, and
          ``grid_overlap_pct[5]`` (a percentage) becomes
          :attr:`main_overlap_fraction` (a fraction). These drive the FOV grid
          built in :func:`_partition_main_grid` (and thus
          :func:`~poc_homography.survey.planner.generators.fov_grid`) over the
          configured visible-ground extent, zoom levels, and tile overlap. A
          config that does not configure phase 5 leaves every main-survey knob
          at its default.
        * **Tilt envelope** — ``config.tilt_envelope`` (the Phase-0 horizon
          calibration product) is forwarded so the FOV grid skips sky tiles;
          ``None`` (the default) reproduces the unconstrained grid exactly.

        Args:
            config: The reproducible plan-config sidecar.
            default_burst_frame_count: Burst frame count for pose-based phases
                absent from ``config.burst_frame_count``.

        Returns:
            A :class:`SurveyPlan` with per-phase burst counts, the main-survey
            ground span, and the optional tilt envelope populated from
            ``config``.
        """
        default = max(MIN_BURST_FRAME_COUNT, int(default_burst_frame_count))
        phase_burst_counts = {
            phase: max(MIN_BURST_FRAME_COUNT, int(config.burst_frame_count.get(phase, default)))
            for phase in _POSE_BURST_PHASES
        }
        defaults = cls()
        # The main survey (phase 5) is "configured" when its pan or tilt extent
        # is pinned by the sidecar; only then do we adopt the sidecar's
        # ground-span knobs (extent, zoom levels, overlap) as one atomic set, so
        # a partial config never silently swaps just one of them.
        if (
            _MAIN_SURVEY_PHASE in config.phase_pan_range
            or _MAIN_SURVEY_PHASE in config.phase_tilt_range
        ):
            main_pan_range = config.phase_pan_range.get(_MAIN_SURVEY_PHASE, defaults.main_pan_range)
            main_tilt_range = config.phase_tilt_range.get(
                _MAIN_SURVEY_PHASE, defaults.main_tilt_range
            )
            main_zoom_levels = tuple(config.zoom_levels)
            overlap_pct = config.grid_overlap_pct.get(_MAIN_SURVEY_PHASE)
            main_overlap_fraction = (
                overlap_pct / 100.0 if overlap_pct is not None else defaults.main_overlap_fraction
            )
        else:
            main_pan_range = defaults.main_pan_range
            main_tilt_range = defaults.main_tilt_range
            main_zoom_levels = defaults.main_zoom_levels
            main_overlap_fraction = defaults.main_overlap_fraction
        return cls(
            burst_count=default,
            enabled_phases=frozenset(config.enabled_phases),
            phase_burst_counts=phase_burst_counts,
            main_pan_range=main_pan_range,
            main_tilt_range=main_tilt_range,
            main_zoom_levels=main_zoom_levels,
            main_overlap_fraction=main_overlap_fraction,
            tilt_envelope=config.tilt_envelope,
        )


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
    rtsp_url_resolver: Callable[[], str | None] | None = None,
) -> SurveyExecution:
    """Run the enabled survey phases for one camera, in order.

    Only the phases in ``plan.enabled_phases`` (the :class:`SurveyPhase`
    numbering 1..9) execute; the default plan enables all nine (#372). Each
    phase's records are persisted to ``sink`` as soon as that phase completes,
    so a later-phase failure never discards the frames already captured (#372).

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
        rtsp_url_resolver: Resolves the Phase 8 jitter RTSP URL for this camera
            (the live wiring wraps ``camera_config.get_rtsp_url``). Used only
            when ``plan.jitter_rtsp_url`` is empty; returning ``None`` (or
            omitting the resolver) makes Phase 8 skip gracefully instead of
            driving the engine with an empty URL (#372).

    Returns:
        A :class:`SurveyExecution` with the C1 :class:`SurveyRun` header and the
        per-phase results.

    Raises:
        JitterTargetError: If Phase 8 runs and is below the minimum jitter
            target (Phase 8 is skipped — never raising — when it is disabled or
            its RTSP URL is unresolvable).
        ValueError: If Phase 9 runs and its holdout overlaps a captured
            training state.
    """
    plan = plan or SurveyPlan()
    now = clock or _utcnow
    engine = SurveyCaptureEngine(camera, clock=clock, uuid_factory=uuid_factory)
    started_at = now()

    training, holdout = _partition_main_grid(spec, plan)

    results: list[PhaseResult] = []
    captured_4_to_7: list[CommandedState] = []

    def record(result: PhaseResult) -> None:
        """Validate, accumulate, and immediately persist one phase's result.

        Emitting per phase (rather than once after the whole sweep) keeps
        persistence as failure-resilient as the PTZ restore: a later-phase
        failure cannot discard the frames already captured (#372).
        """
        _assert_result_not_cross_tagged(result)
        results.append(result)
        _emit_result(result, sink)

    # Bracket the whole camera-driving sweep: snapshot the pre-sweep PTZ
    # position and restore it on normal completion, exception, or abort (#342).
    with preserve_ptz_position(camera):
        if plan.is_enabled(1):
            record(executors.run_inventory(camera, run_id=run_id, camera_id=camera_id, clock=now))

        if plan.is_enabled(2):
            record(
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

        if plan.is_enabled(3):
            record(
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

        if plan.is_enabled(4):
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
            record(nadir)
            captured_4_to_7.extend(nadir.commanded_states)

        if plan.is_enabled(5):
            main = executors.run_main_survey(
                engine,
                training,
                run_id=run_id,
                camera_id=camera_id,
                output_dir=_phase_dir(base_output_dir, SurveyPhase.MAIN_SURVEY),
                burst_count=plan.burst_for(5),
            )
            record(main)
            captured_4_to_7.extend(main.commanded_states)

        if plan.is_enabled(6):
            cross = executors.run_cross_zoom(
                engine,
                run_id=run_id,
                camera_id=camera_id,
                output_dir=_phase_dir(base_output_dir, SurveyPhase.CROSS_ZOOM),
                anchors=plan.cross_zoom_anchors,
                zoom_levels=plan.cross_zoom_levels,
                burst_count=plan.burst_for(6),
            )
            record(cross)
            captured_4_to_7.extend(cross.commanded_states)

        if plan.is_enabled(7):
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
            record(repeat)
            captured_4_to_7.extend(repeat.commanded_states)

        if plan.is_enabled(8):
            jitter_rtsp_url = _resolve_jitter_rtsp_url(plan, rtsp_url_resolver)
            if jitter_rtsp_url:
                record(
                    executors.run_jitter(
                        engine,
                        run_id=run_id,
                        camera_id=camera_id,
                        output_dir=_phase_dir(base_output_dir, SurveyPhase.STATIC_JITTER),
                        rtsp_url=jitter_rtsp_url,
                        poses=plan.jitter_poses,
                        zoom_levels=plan.jitter_zoom_levels,
                        burst_duration_s=plan.jitter_burst_duration_s,
                        target_fps=plan.jitter_target_fps,
                    )
                )
            else:
                _logger.warning(
                    "Phase 8 (static jitter) skipped for camera %s: no RTSP URL resolvable",
                    camera_id,
                )

        if plan.is_enabled(9):
            record(
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


def _resolve_jitter_rtsp_url(
    plan: SurveyPlan, resolver: Callable[[], str | None] | None
) -> str | None:
    """Resolve the Phase 8 jitter RTSP URL, preferring the plan's explicit one.

    Falls back to ``resolver`` (the live per-camera/tenant credential lookup)
    when ``plan.jitter_rtsp_url`` is empty. Returns ``None`` when neither
    yields a usable URL so the caller skips jitter rather than driving the
    engine with an empty URL (#372). The resolver contract is to return
    ``None`` (not raise) when no URL is resolvable.
    """
    if plan.jitter_rtsp_url:
        return plan.jitter_rtsp_url
    if resolver is not None:
        return resolver()
    return None


def _emit_result(result: PhaseResult, sink: PhaseSink) -> None:
    """Route every record produced by one phase to the sink."""
    if result.inventory is not None:
        sink.save_inventory(result.inventory)
    for burst in result.bursts:
        sink.save_burst(burst)
    for frame in result.frames:
        sink.save_frame(frame)


def _assert_result_not_cross_tagged(result: PhaseResult) -> None:
    """Assert one phase's records are all tagged with its own identifier."""
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
