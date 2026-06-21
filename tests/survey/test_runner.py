"""End-to-end tests for the nine-phase survey runner and the YAML sink."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus
from poc_homography.infrastructure.repositories.repo_yaml_survey_run import (
    RepoYamlSurveyRun,
)
from poc_homography.infrastructure.survey.yaml_phase_sink import YamlPhaseSink
from poc_homography.survey.phases import executors
from poc_homography.survey.phases.common import PhaseResult
from poc_homography.survey.phases.runner import SurveyExecution, SurveyPlan, execute_survey

# Offline jitter (Phase 8) drives the engine through the ``patch_rtsp`` seam, so
# the URL value is never dialled — any non-empty string enables the phase (#372).
_FAKE_RTSP_URL = "rtsp://fake/stream"

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
    plan: SurveyPlan | None = None,
) -> SurveyExecution:
    return execute_survey(
        camera,
        run_id="run-1",
        camera_id="icozee-camptz-04",
        sink=sink,  # type: ignore[arg-type]
        base_output_dir=base,
        plan=plan or SurveyPlan(jitter_rtsp_url=_FAKE_RTSP_URL),
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

    def test_sweep_restores_original_ptz_position(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        patch_rtsp,
    ) -> None:
        patch_rtsp(20)
        _run(camera, clock, uuid_factory, _RecordingSink(), tmp_path)
        # The sweep snapshots the pre-sweep PTZ once and the final move restores
        # exactly that (pan, tilt, zoom) — the FakeCamera reports (1.0, -1.0, 2.0).
        assert camera.calls.count("get_ptz_status") == 1
        assert camera.calls[-1] == "move_absolute(1.0,-1.0,2.0)"

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


# Executor function name -> the SurveyPhase it produces, in phase-number order.
_EXECUTOR_BY_PHASE = {
    1: ("run_inventory", SurveyPhase.CAMERA_INVENTORY),
    2: ("run_ptz_characterization", SurveyPhase.PTZ_CHARACTERIZATION),
    3: ("run_zoom_characterization", SurveyPhase.ZOOM_CHARACTERIZATION),
    4: ("run_dense_nadir", SurveyPhase.DENSE_NADIR),
    5: ("run_main_survey", SurveyPhase.MAIN_SURVEY),
    6: ("run_cross_zoom", SurveyPhase.CROSS_ZOOM),
    7: ("run_repeatability", SurveyPhase.REPEATABILITY),
    8: ("run_jitter", SurveyPhase.STATIC_JITTER),
    9: ("run_validation", SurveyPhase.VALIDATION),
}


def _spy_all_executors(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every phase executor with a spy; return the invocation log.

    Each spy records its name and returns an empty :class:`PhaseResult` tagged
    with its own phase (so the runner's per-phase cross-tag check passes without
    driving the C2 engine or a camera).
    """
    invoked: list[str] = []

    def make_spy(name: str, phase: SurveyPhase):
        def spy(*_args: object, **_kwargs: object) -> PhaseResult:
            invoked.append(name)
            return PhaseResult(phase=phase)

        return spy

    for name, phase in _EXECUTOR_BY_PHASE.values():
        monkeypatch.setattr(executors, name, make_spy(name, phase))
    return invoked


class TestEnabledPhasesGating:
    def test_only_enabled_phase_executors_run(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invoked = _spy_all_executors(monkeypatch)
        sink = _RecordingSink()
        plan = SurveyPlan(enabled_phases=frozenset({1, 5}))
        execution = _run(camera, clock, uuid_factory, sink, tmp_path, plan=plan)

        assert invoked == ["run_inventory", "run_main_survey"]
        assert execution.run.phases == frozenset(
            {SurveyPhase.CAMERA_INVENTORY, SurveyPhase.MAIN_SURVEY}
        )

    def test_default_plan_runs_all_nine(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invoked = _spy_all_executors(monkeypatch)
        plan = SurveyPlan(jitter_rtsp_url=_FAKE_RTSP_URL)
        _run(camera, clock, uuid_factory, _RecordingSink(), tmp_path, plan=plan)
        assert invoked == [name for name, _ in _EXECUTOR_BY_PHASE.values()]


class TestJitterRtspResolution:
    def test_jitter_uses_resolver_when_plan_url_empty(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def spy_jitter(*_args: object, **kwargs: object) -> PhaseResult:
            captured["rtsp_url"] = kwargs["rtsp_url"]
            return PhaseResult(phase=SurveyPhase.STATIC_JITTER)

        monkeypatch.setattr(executors, "run_jitter", spy_jitter)
        plan = SurveyPlan(enabled_phases=frozenset({8}), jitter_rtsp_url="")
        execute_survey(
            camera,
            run_id="run-1",
            camera_id="cam-x",
            sink=_RecordingSink(),  # type: ignore[arg-type]
            base_output_dir=tmp_path,
            plan=plan,
            clock=clock,
            uuid_factory=uuid_factory,
            rtsp_url_resolver=lambda: "rtsp://resolved/stream",
        )
        assert captured["rtsp_url"] == "rtsp://resolved/stream"

    def test_jitter_skipped_with_warning_when_unresolvable(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> PhaseResult:
            raise AssertionError("run_jitter must not be called when the URL is unresolvable")

        monkeypatch.setattr(executors, "run_jitter", boom)
        sink = _RecordingSink()
        plan = SurveyPlan(enabled_phases=frozenset({8}), jitter_rtsp_url="")
        with caplog.at_level("WARNING", logger="poc_homography.survey.phases.runner"):
            execution = execute_survey(
                camera,
                run_id="run-1",
                camera_id="cam-x",
                sink=sink,  # type: ignore[arg-type]
                base_output_dir=tmp_path,
                plan=plan,
                clock=clock,
                uuid_factory=uuid_factory,
                rtsp_url_resolver=lambda: None,
            )

        assert sink.bursts == []
        assert execution.run.phases == frozenset()
        messages = [record.getMessage().lower() for record in caplog.records]
        assert any("static jitter" in message and "skipped" in message for message in messages)


class TestPersistenceRobustness:
    def test_phase_failure_persists_earlier_phases(
        self,
        camera: FakeCamera,
        clock: IncrementingClock,
        uuid_factory: CountingUuid,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def failing_ptz(*_args: object, **_kwargs: object) -> PhaseResult:
            raise RuntimeError("phase 2 capture failed mid-run")

        monkeypatch.setattr(executors, "run_ptz_characterization", failing_ptz)
        sink = _RecordingSink()
        plan = SurveyPlan(enabled_phases=frozenset({1, 2}))

        with pytest.raises(RuntimeError, match="phase 2 capture failed"):
            _run(camera, clock, uuid_factory, sink, tmp_path, plan=plan)

        # Phase 1 completed before phase 2 raised, so its inventory was already
        # persisted — the failure did not discard the captured data (#372).
        assert len(sink.inventories) == 1
        assert sink.inventories[0].phase is SurveyPhase.CAMERA_INVENTORY


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
