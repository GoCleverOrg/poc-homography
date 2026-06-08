"""Capture/phase-wiring tests for #276 clean-plate capture.

Covers the per-pose burst-count bridge from :class:`SurveyPlanConfig`, the
additive per-frame metadata stamped by the capture engine (intrinsics,
full_optics) and the phase layer (stable pose_id), and the deliberate
``None`` of the downstream-only fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.camera_config import (
    DEFAULT_BASE_FOCAL_LENGTH_MM,
    DEFAULT_SENSOR_WIDTH_MM,
)
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig
from poc_homography.infrastructure.survey.capture_engine import SurveyCaptureEngine
from poc_homography.survey.phases import executors
from poc_homography.survey.phases.runner import (
    MIN_BURST_FRAME_COUNT,
    SurveyPlan,
)
from poc_homography.survey.planner.poses import canonical_pose_key

if TYPE_CHECKING:
    from pathlib import Path

    from tests.survey.conftest import CountingUuid, FakeCamera, IncrementingClock

    from poc_homography.domain.entities.survey.frame_record import FrameRecord

_SPEC_PAN = 90.0
_SPEC_TILT = -20.0
_SPEC_ZOOM = 4.0


def _engine(
    camera: FakeCamera, clock: IncrementingClock, uuid_factory: CountingUuid
) -> SurveyCaptureEngine:
    return SurveyCaptureEngine(camera, clock=clock, uuid_factory=uuid_factory)


def _cross_zoom_frames(
    camera: FakeCamera,
    clock: IncrementingClock,
    uuid_factory: CountingUuid,
    out: Path,
    *,
    burst_count: int = 1,
    run_id: str = "run-1",
) -> list[FrameRecord]:
    """Drive a small pose-based phase (cross-zoom) and return its frames."""
    result = executors.run_cross_zoom(
        _engine(camera, clock, uuid_factory),
        run_id=run_id,
        camera_id="cam01",
        output_dir=out,
        anchors=((_SPEC_PAN, _SPEC_TILT),),
        zoom_levels=(_SPEC_ZOOM,),
        burst_count=burst_count,
    )
    return list(result.frames)


class TestPerPoseBursts:
    def test_pose_phase_emits_n_frames_per_pose_sharing_one_burst_id(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        frames = _cross_zoom_frames(camera, clock, uuid_factory, tmp_path, burst_count=3)
        # One pose (single anchor x single zoom) -> exactly N frames.
        assert len(frames) == 3
        assert len({f.capture.burst_id for f in frames}) == 1
        assert [f.capture.frame_index for f in frames] == [0, 1, 2]


class TestStablePoseId:
    def test_same_pose_yields_same_pose_id_across_runs(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        run_a = _cross_zoom_frames(camera, clock, uuid_factory, tmp_path / "a", run_id="run-a")
        # A second, independent run with fresh factories and a distinct run id.
        from tests.survey.conftest import (
            CountingUuid as _Uuid,
        )
        from tests.survey.conftest import (
            FakeCamera as _Cam,
        )
        from tests.survey.conftest import (
            IncrementingClock as _Clock,
        )

        run_b = _cross_zoom_frames(_Cam(), _Clock(), _Uuid(), tmp_path / "b", run_id="run-b")

        expected = canonical_pose_key(_SPEC_PAN, _SPEC_TILT, _SPEC_ZOOM)
        # Distinct runs (different run ids) yield the SAME stable pose id.
        assert run_a[0].capture.run_id != run_b[0].capture.run_id
        assert {f.survey_context.pose_id for f in run_a} == {expected}
        assert {f.survey_context.pose_id for f in run_b} == {expected}

    def test_pose_id_preserves_other_survey_context_fields(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        frames = _cross_zoom_frames(camera, clock, uuid_factory, tmp_path)
        # Cross-zoom stamps region_id; pose_id must be added, not replace it.
        assert all(f.survey_context.region_id == "region_0" for f in frames)
        assert all(f.survey_context.pose_id is not None for f in frames)


class TestIntrinsicsStamped:
    def test_every_frame_has_intrinsics_with_k_matrix(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        frames = _cross_zoom_frames(camera, clock, uuid_factory, tmp_path)
        assert frames
        for frame in frames:
            intrinsics = frame.intrinsics
            assert intrinsics is not None
            assert float(intrinsics.zoom) == _SPEC_ZOOM
            assert float(intrinsics.sensor_width_mm) == DEFAULT_SENSOR_WIDTH_MM
            assert float(intrinsics.base_focal_length_mm) == DEFAULT_BASE_FOCAL_LENGTH_MM
            assert intrinsics.k_matrix is not None
            assert len(intrinsics.k_matrix) == 3
            assert all(len(row) == 3 for row in intrinsics.k_matrix)
            # Principal point is the image centre; fx == fy on the diagonal.
            assert intrinsics.k_matrix[0][2] == float(frame.image_data.width) / 2.0
            assert intrinsics.k_matrix[0][0] == intrinsics.k_matrix[1][1]


class TestFullOpticsStamped:
    def test_every_frame_has_full_optics_from_camera(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        frames = _cross_zoom_frames(camera, clock, uuid_factory, tmp_path)
        assert frames
        for frame in frames:
            optics = frame.full_optics
            assert optics is not None
            # Mapped from the FakeCamera's ImageOptics (see conftest.make_optics).
            assert optics.exposure_type == "auto"
            assert optics.iris == 50
            assert optics.white_balance == "auto"
            assert optics.focus == 150
            # Unmapped (no adapter source) fields stay None.
            assert optics.shutter is None
            assert optics.gain is None


class TestDownstreamFieldsLeftNone:
    def test_ground_homography_and_floor_mask_are_none_at_capture(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        frames = _cross_zoom_frames(camera, clock, uuid_factory, tmp_path)
        assert frames
        # These require deployment extrinsics / an external masker; the survey
        # leaves them None and they are populated downstream.
        assert all(f.ground_homography is None for f in frames)
        assert all(f.floor_mask_reference is None for f in frames)


class TestFromPlanConfigBridge:
    def test_maps_burst_frame_count_per_phase(self) -> None:
        config = SurveyPlanConfig(burst_frame_count={5: 8, 2: 4})
        plan = SurveyPlan.from_plan_config(config)
        assert plan.burst_for(5) == 8
        assert plan.burst_for(2) == 4

    def test_default_applies_to_unconfigured_phases(self) -> None:
        config = SurveyPlanConfig(burst_frame_count={})
        plan = SurveyPlan.from_plan_config(config, default_burst_frame_count=7)
        # Every pose-based phase falls back to the default.
        for phase in (2, 3, 4, 5, 6, 7, 9):
            assert plan.burst_for(phase) == 7

    def test_clamps_below_minimum_to_two(self) -> None:
        config = SurveyPlanConfig(burst_frame_count={5: 1})
        plan = SurveyPlan.from_plan_config(config, default_burst_frame_count=1)
        assert plan.burst_for(5) == MIN_BURST_FRAME_COUNT
        assert plan.burst_for(3) == MIN_BURST_FRAME_COUNT

    def test_default_is_at_least_two(self) -> None:
        plan = SurveyPlan.from_plan_config(SurveyPlanConfig())
        assert plan.burst_for(5) >= MIN_BURST_FRAME_COUNT


class TestRunnerHonoursPlanBursts:
    def test_runner_emits_configured_burst_per_pose(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        patch_rtsp,
    ) -> None:
        from poc_homography.survey.phases.runner import execute_survey

        patch_rtsp(20)

        class _Sink:
            def __init__(self) -> None:
                self.frames: list[FrameRecord] = []

            def save_inventory(self, record: object) -> None:
                return None

            def save_frame(self, record: FrameRecord) -> None:
                self.frames.append(record)

            def save_burst(self, record: object) -> None:
                return None

        sink = _Sink()
        plan = SurveyPlan.from_plan_config(SurveyPlanConfig(), default_burst_frame_count=2)
        execute_survey(
            camera,
            run_id="run-1",
            camera_id="cam01",
            sink=sink,  # type: ignore[arg-type]
            base_output_dir=tmp_path,
            plan=plan,
            clock=clock,
            uuid_factory=uuid_factory,
        )

        # Every snapshot-burst (non-jitter) pose emits >= 2 frames sharing a
        # burst id; group the pose-based frames by burst id and assert size.
        pose_frames = [f for f in sink.frames if f.capture.phase is not SurveyPhase.STATIC_JITTER]
        assert pose_frames
        by_burst: dict[str | None, list[FrameRecord]] = {}
        for frame in pose_frames:
            by_burst.setdefault(frame.capture.burst_id, []).append(frame)
        assert all(len(group) == 2 for group in by_burst.values())
        # And every emitted frame carries the additive clean-plate metadata.
        assert all(f.intrinsics is not None for f in pose_frames)
        assert all(f.full_optics is not None for f in pose_frames)
        assert all(f.survey_context.pose_id is not None for f in pose_frames)
