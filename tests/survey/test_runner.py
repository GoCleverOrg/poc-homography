"""End-to-end tests for the nine-phase survey runner and the YAML sink."""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus
from poc_homography.infrastructure.repositories.repo_yaml_survey_run import (
    RepoYamlSurveyRun,
)
from poc_homography.infrastructure.survey.yaml_phase_sink import YamlPhaseSink
from poc_homography.survey.phases.runner import SurveyExecution, execute_survey

if TYPE_CHECKING:
    from pathlib import Path

    from tests.survey.conftest import CountingUuid, FakeCamera, IncrementingClock

    from poc_homography.domain.entities.survey.frame_record import FrameRecord
    from poc_homography.domain.entities.survey.inventory_record import (
        CameraInventoryRecord,
    )
    from poc_homography.domain.entities.survey.video_burst_record import (
        VideoBurstRecord,
    )

_ALL_NINE = set(SurveyPhase)


class _RecordingSink:
    """In-memory :class:`PhaseSink` collecting every emitted record."""

    def __init__(self) -> None:
        self.frames: list[FrameRecord] = []
        self.bursts: list[VideoBurstRecord] = []
        self.inventories: list[CameraInventoryRecord] = []

    def save_inventory(self, record: CameraInventoryRecord) -> None:
        self.inventories.append(record)

    def save_frame(self, record: FrameRecord) -> None:
        self.frames.append(record)

    def save_burst(self, record: VideoBurstRecord) -> None:
        self.bursts.append(record)


def _run(
    camera: FakeCamera,
    clock: IncrementingClock,
    uuid_factory: CountingUuid,
    sink: object,
    base: Path,
) -> SurveyExecution:
    return execute_survey(
        camera,
        run_id="run-1",
        camera_id="icozee-camptz-04",
        sink=sink,  # type: ignore[arg-type]
        base_output_dir=base,
        clock=clock,
        uuid_factory=uuid_factory,
    )


class TestExecuteSurvey:
    def test_all_nine_phases_run(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        patch_rtsp,
    ) -> None:
        patch_rtsp(20)
        sink = _RecordingSink()
        execution = _run(camera, clock, uuid_factory, sink, tmp_path)
        assert {r.phase for r in execution.results} == _ALL_NINE
        assert execution.run.phases == frozenset(_ALL_NINE)
        assert execution.run.status is SurveyRunStatus.COMPLETED

    def test_one_inventory_record(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        patch_rtsp,
    ) -> None:
        patch_rtsp(20)
        sink = _RecordingSink()
        _run(camera, clock, uuid_factory, sink, tmp_path)
        assert len(sink.inventories) == 1
        assert sink.inventories[0].phase is SurveyPhase.CAMERA_INVENTORY

    def test_no_record_tagged_with_another_phase(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        patch_rtsp,
    ) -> None:
        patch_rtsp(20)
        sink = _RecordingSink()
        execution = _run(camera, clock, uuid_factory, sink, tmp_path)
        for result in execution.results:
            assert all(f.capture.phase is result.phase for f in result.frames)
            assert all(b.phase is result.phase for b in result.bursts)

    def test_jitter_present_and_complete(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        patch_rtsp,
    ) -> None:
        patch_rtsp(20)
        sink = _RecordingSink()
        _run(camera, clock, uuid_factory, sink, tmp_path)
        assert len(sink.bursts) == 9
        for burst in sink.bursts:
            assert burst.phase is SurveyPhase.STATIC_JITTER
            assert burst.segment_path.exists()
            assert burst.frame_refs

    def test_validation_disjoint_from_training(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        patch_rtsp,
    ) -> None:
        patch_rtsp(20)
        sink = _RecordingSink()
        _run(camera, clock, uuid_factory, sink, tmp_path)

        def keys(frames: list[FrameRecord]) -> set[tuple[float, float, float]]:
            return {
                (
                    float(f.commanded.commanded_pan),
                    float(f.commanded.commanded_tilt),
                    float(f.commanded.commanded_zoom),
                )
                for f in frames
            }

        training_phases = {
            SurveyPhase.DENSE_NADIR,
            SurveyPhase.MAIN_SURVEY,
            SurveyPhase.CROSS_ZOOM,
            SurveyPhase.REPEATABILITY,
        }
        training = [f for f in sink.frames if f.capture.phase in training_phases]
        validation = [f for f in sink.frames if f.capture.phase is SurveyPhase.VALIDATION]
        assert validation
        assert keys(training).isdisjoint(keys(validation))


class TestYamlPhaseSinkDataset:
    def test_dataset_queryable_by_phase(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        patch_rtsp,
    ) -> None:
        patch_rtsp(20)
        survey_root = tmp_path / "survey"
        sink = YamlPhaseSink(survey_root)
        _run(camera, clock, uuid_factory, sink, tmp_path / "images")

        repo = RepoYamlSurveyRun(tmp_path / "survey_runs", frames_dir=survey_root)
        all_frames = repo.get_frames_by_run("run-1")
        assert all_frames
        image_bearing = _ALL_NINE - {SurveyPhase.CAMERA_INVENTORY}
        for phase in image_bearing:
            assert repo.get_frames_by_phase(phase), f"no frames for {phase.value}"
