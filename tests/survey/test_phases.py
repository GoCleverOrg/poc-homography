"""Per-phase DoD tests for the nine C4 survey-phase executors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.infrastructure.survey.capture_engine import SurveyCaptureEngine
from poc_homography.survey.phases import executors
from poc_homography.survey.phases.common import pose_to_commanded
from poc_homography.survey.planner.generators import fov_grid, partition_holdout
from poc_homography.survey.planner.poses import Pose
from poc_homography.types import Degrees, Unitless

if TYPE_CHECKING:
    from pathlib import Path

    from tests.survey.conftest import CountingUuid, FakeCamera, IncrementingClock

_SPEC = CameraSpec.HIKVISION_DS_2DF8425IX


def _engine(
    camera: FakeCamera, clock: IncrementingClock, uuid_factory: CountingUuid
) -> SurveyCaptureEngine:
    return SurveyCaptureEngine(camera, clock=clock, uuid_factory=uuid_factory)


class TestPhase1Inventory:
    def test_emits_one_inventory_record_with_full_self_report(
        self, camera: FakeCamera, clock: IncrementingClock
    ) -> None:
        result = executors.run_inventory(camera, run_id="run-1", camera_id="cam01", clock=clock)
        assert result.phase is SurveyPhase.CAMERA_INVENTORY
        assert result.inventory is not None
        record = result.inventory
        assert record.phase is SurveyPhase.CAMERA_INVENTORY
        assert record.device_info.firmware_version == "V5.7.3"
        assert record.capabilities.zoom_max == 25.0
        assert record.optics.focus.style == "SEMIAUTOMATIC"
        assert record.stream_profiles and record.presets
        # Single-pass query across all capability/state surfaces.
        for method in (
            "get_device_info",
            "get_capabilities",
            "get_optics",
            "get_health",
            "get_stream_profiles",
            "list_presets",
        ):
            assert method in camera.calls


class TestPhase2PtzCharacterization:
    def test_small_large_repeated_moves_both_axes(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        result = executors.run_ptz_characterization(
            _engine(camera, clock, uuid_factory),
            run_id="run-1",
            camera_id="cam01",
            output_dir=tmp_path,
            pan_range=(0.0, 20.0),
            tilt_range=(-20.0, 0.0),
            small_step=5.0,
            large_step=20.0,
            fixed_tilt=-10.0,
            fixed_pan=0.0,
            fixed_zoom=1.0,
            repeat_count=2,
        )
        assert result.frames
        assert all(f.capture.phase is SurveyPhase.PTZ_CHARACTERIZATION for f in result.frames)
        pan_dirs = {f.movement.direction_pan for f in result.frames}
        tilt_dirs = {f.movement.direction_tilt for f in result.frames}
        assert {"cw", "ccw"} <= pan_dirs
        assert {"up", "down"} <= tilt_dirs


class TestPhase3ZoomCharacterization:
    def test_both_directional_passes_distinguishable(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        result = executors.run_zoom_characterization(
            _engine(camera, clock, uuid_factory),
            run_id="run-1",
            camera_id="cam01",
            output_dir=tmp_path,
            zoom_min=1.0,
            zoom_max=16.0,
            zoom_step=5.0,
            fixed_pan=0.0,
            fixed_tilt=-10.0,
        )
        assert all(f.capture.phase is SurveyPhase.ZOOM_CHARACTERIZATION for f in result.frames)
        labels = [f.survey_context.approach_direction for f in result.frames]
        assert "wide_to_tele" in labels
        assert "tele_to_wide" in labels
        # Passes are not interleaved: all wide_to_tele precede all tele_to_wide.
        first_t2w = labels.index("tele_to_wide")
        assert all(label == "wide_to_tele" for label in labels[:first_t2w])


class TestPhase4DenseNadir:
    def test_region_id_on_every_frame(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        result = executors.run_dense_nadir(
            _engine(camera, clock, uuid_factory),
            _SPEC,
            run_id="run-1",
            camera_id="cam01",
            output_dir=tmp_path,
            nadir_pan=180.0,
            nadir_tilt=-85.0,
            radius_deg=3.0,
            zoom=8.0,
            overlap_fraction=0.8,
        )
        assert result.frames
        assert all(f.capture.phase is SurveyPhase.DENSE_NADIR for f in result.frames)
        assert all(f.survey_context.region_id == "nadir" for f in result.frames)


class TestPhase5MainSurvey:
    def test_spans_zoom_levels(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        poses = fov_grid(_SPEC, (0.0, 10.0), (-10.0, 0.0), (2.0, 8.0), 0.5)
        result = executors.run_main_survey(
            _engine(camera, clock, uuid_factory),
            poses,
            run_id="run-1",
            camera_id="cam01",
            output_dir=tmp_path,
        )
        assert result.frames
        assert all(f.capture.phase is SurveyPhase.MAIN_SURVEY for f in result.frames)
        commanded_zooms = {float(f.commanded.commanded_zoom) for f in result.frames}
        assert {2.0, 8.0} <= commanded_zooms


class TestPhase6CrossZoom:
    def test_shared_region_id_across_zoom_levels(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        result = executors.run_cross_zoom(
            _engine(camera, clock, uuid_factory),
            run_id="run-1",
            camera_id="cam01",
            output_dir=tmp_path,
            anchors=((90.0, -20.0), (120.0, -25.0)),
            zoom_levels=(2.0, 8.0, 16.0),
        )
        assert all(f.capture.phase is SurveyPhase.CROSS_ZOOM for f in result.frames)
        region_0 = [f for f in result.frames if f.survey_context.region_id == "region_0"]
        assert len(region_0) == 3
        assert {float(f.commanded.commanded_zoom) for f in region_0} == {2.0, 8.0, 16.0}


class TestPhase7Repeatability:
    def test_multiple_visits_varying_approach(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        result = executors.run_repeatability(
            _engine(camera, clock, uuid_factory),
            run_id="run-1",
            camera_id="cam01",
            output_dir=tmp_path,
            target_pan=200.0,
            target_tilt=-30.0,
            target_zoom=4.0,
            approach_deltas=((-5.0, 0.0), (5.0, 0.0), (0.0, -5.0)),
        )
        assert result.frames
        assert all(f.capture.phase is SurveyPhase.REPEATABILITY for f in result.frames)
        assert all(f.movement.is_repeatability_sequence for f in result.frames)
        approaches = {f.survey_context.approach_direction for f in result.frames}
        assert len({a for a in approaches if a}) >= 2
        sequences = {f.survey_context.sequence_index for f in result.frames}
        assert len({s for s in sequences if s is not None}) == 3


class TestPhase8Jitter:
    def test_bursts_meet_minimum_target(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        patch_rtsp,
    ) -> None:
        patch_rtsp(20)
        result = executors.run_jitter(
            _engine(camera, clock, uuid_factory),
            run_id="run-1",
            camera_id="cam01",
            output_dir=tmp_path,
            rtsp_url="rtsp://fake/stream",
            poses=((10.0, -15.0), (90.0, -20.0), (200.0, -30.0)),
            zoom_levels=(1.0, 8.0, 16.0),
            burst_duration_s=10.0,
            target_fps=5.0,
        )
        assert len(result.bursts) == 9  # 3 poses x 3 zoom levels
        assert all(b.phase is SurveyPhase.STATIC_JITTER for b in result.bursts)
        for burst in result.bursts:
            assert burst.segment_path.exists()
            assert burst.frame_refs  # addressable per-frame references
        assert result.frames
        assert all(f.capture.phase is SurveyPhase.STATIC_JITTER for f in result.frames)

    def test_too_few_poses_rejected(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(executors.JitterTargetError):
            executors.run_jitter(
                _engine(camera, clock, uuid_factory),
                run_id="run-1",
                camera_id="cam01",
                output_dir=tmp_path,
                rtsp_url="rtsp://fake/stream",
                poses=((10.0, -15.0), (90.0, -20.0)),
                zoom_levels=(1.0, 8.0, 16.0),
            )

    def test_short_achieved_burst_rejected(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        patch_rtsp,
    ) -> None:
        # A stream that ends after a single frame yields a burst far shorter
        # than the 10s minimum; the achieved-target guard must reject it.
        patch_rtsp(1)
        with pytest.raises(executors.JitterTargetError):
            executors.run_jitter(
                _engine(camera, clock, uuid_factory),
                run_id="run-1",
                camera_id="cam01",
                output_dir=tmp_path,
                rtsp_url="rtsp://fake/stream",
                poses=((10.0, -15.0), (90.0, -20.0), (200.0, -30.0)),
                zoom_levels=(1.0, 8.0, 16.0),
            )

    def test_too_short_duration_rejected(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(executors.JitterTargetError):
            executors.run_jitter(
                _engine(camera, clock, uuid_factory),
                run_id="run-1",
                camera_id="cam01",
                output_dir=tmp_path,
                rtsp_url="rtsp://fake/stream",
                poses=((10.0, -15.0), (90.0, -20.0), (200.0, -30.0)),
                zoom_levels=(1.0, 8.0, 16.0),
                burst_duration_s=5.0,
            )


class TestPhase9Validation:
    def test_disjoint_holdout_captured(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        grid = fov_grid(_SPEC, (0.0, 10.0), (-10.0, 0.0), (2.0, 8.0), 0.5)
        training, holdout = partition_holdout(grid, 0.3, seed=7)
        captured = [pose_to_commanded(p) for p in training]
        result = executors.run_validation(
            _engine(camera, clock, uuid_factory),
            holdout,
            captured,
            run_id="run-1",
            camera_id="cam01",
            output_dir=tmp_path,
        )
        assert result.frames
        assert all(f.capture.phase is SurveyPhase.VALIDATION for f in result.frames)

    def test_overlap_raises(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
    ) -> None:
        holdout = [Pose(pan=Degrees(5.0), tilt=Degrees(-5.0), zoom=Unitless(2.0), is_holdout=True)]
        captured = [pose_to_commanded(holdout[0])]  # identical commanded state
        with pytest.raises(ValueError, match="disjoint"):
            executors.run_validation(
                _engine(camera, clock, uuid_factory),
                holdout,
                captured,
                run_id="run-1",
                camera_id="cam01",
                output_dir=tmp_path,
            )
