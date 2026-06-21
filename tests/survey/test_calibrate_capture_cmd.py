"""Tests for the ``hom cleanplate calibrate-capture`` command (#344).

Exercise the extracted ``_run_calibration_capture`` helper end-to-end with a
fake camera (driving the real :func:`execute_survey`), a stub sink collecting
``save_frame`` records, and a fake session + repo capturing the saved
:class:`SurveyRun` — so nothing touches MinIO, Neon, or RTSP.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from poc_homography.cli import calibrate_capture as cmd
from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig
from poc_homography.survey.phases.runner import SurveyPlan

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from poc_homography.domain.entities.survey.frame_record import FrameRecord

    from .conftest import CountingUuid, FakeCamera, IncrementingClock


class _StubSink:
    """A :class:`PhaseSink` double recording the frames routed to it."""

    def __init__(self) -> None:
        self.frames: list[FrameRecord] = []

    def save_inventory(self, record: object) -> None:
        return None

    def save_frame(self, record: FrameRecord) -> None:
        self.frames.append(record)

    def save_burst(self, record: object) -> None:
        return None


class _FakeRepo:
    """Captures the entities and plan-config saved through it."""

    saved: list[SurveyRun] = []
    plan_configs: list[tuple[str, SurveyPlanConfig]] = []

    def __init__(self, session: object, entity_factory: object) -> None:
        self._session = session
        self._factory = entity_factory

    def save(self, entity: SurveyRun) -> bool:
        type(self).saved.append(entity)
        return True

    def save_plan_config(self, run_id: str, config: SurveyPlanConfig) -> bool:
        type(self).plan_configs.append((run_id, config))
        return True


def test_rtsp_url_resolver_returns_url_when_resolvable(monkeypatch) -> None:
    monkeypatch.setattr(cmd, "get_rtsp_url", lambda name: f"rtsp://{name}/stream")
    assert cmd.make_rtsp_url_resolver("cam01")() == "rtsp://cam01/stream"


def test_rtsp_url_resolver_swallows_config_shape_errors(monkeypatch) -> None:
    # get_rtsp_url raises ValueError on missing credentials and KeyError when a
    # camera row lacks an expected field (e.g. ``ip``); both must resolve to
    # None so the runner skips jitter rather than crashing (#372).
    def raise_value_error(_name: str) -> str:
        raise ValueError("credentials not set")

    def raise_key_error(_name: str) -> str:
        raise KeyError("ip")

    monkeypatch.setattr(cmd, "get_rtsp_url", raise_value_error)
    assert cmd.make_rtsp_url_resolver("cam01")() is None

    monkeypatch.setattr(cmd, "get_rtsp_url", raise_key_error)
    assert cmd.make_rtsp_url_resolver("cam01")() is None


def test_run_calibration_capture_routes_frames_and_saves_run(
    camera: FakeCamera,
    clock: IncrementingClock,
    uuid_factory: CountingUuid,
    tmp_path: Path,
    patch_rtsp,
    monkeypatch,
) -> None:
    patch_rtsp(20)

    # Replace the real Postgres repo with an in-memory capture and the session
    # factory with a no-op context manager so nothing touches Neon.
    _FakeRepo.saved = []
    _FakeRepo.plan_configs = []
    monkeypatch.setattr(cmd, "RepoPostgresSurveyRun", _FakeRepo)

    @contextmanager
    def fake_session_factory() -> Generator[object, None, None]:
        yield object()

    config = SurveyPlanConfig()
    runner_plan = SurveyPlan.from_plan_config(config, default_burst_frame_count=2)
    sink = _StubSink()

    execution = cmd._run_calibration_capture(
        camera=camera,
        plan_config=config,
        runner_plan=runner_plan,
        camera_id="cam01",
        run_id="run-1",
        base_output_dir=tmp_path,
        sink=sink,
        session_factory=fake_session_factory,
        clock=clock,
        uuid_factory=uuid_factory,
    )

    # Frames were routed to the sink.
    assert len(sink.frames) > 0
    # The run header was saved with a SurveyRun for this run/camera.
    assert len(_FakeRepo.saved) == 1
    saved_run = _FakeRepo.saved[0]
    assert isinstance(saved_run, SurveyRun)
    assert saved_run.run_id == "run-1"
    assert saved_run.camera_id == "cam01"
    # The plan-config sidecar was persisted under the same run id.
    assert _FakeRepo.plan_configs == [("run-1", config)]
    # The returned execution echoes the persisted run.
    assert execution.run is saved_run
