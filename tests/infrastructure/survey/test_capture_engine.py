"""Unit tests for :class:`SurveyCaptureEngine` (offline, mocked adapters)."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image

from poc_homography.domain.entities.survey.frame_record import CommandedState
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.vo.device_info import DeviceInfo
from poc_homography.domain.vo.image_optics import (
    ExposureState,
    FocusState,
    ImageOptics,
    IrisState,
    WhiteBalanceState,
)
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.domain.vo.stream_profile import StreamProfile
from poc_homography.infrastructure.survey import capture_engine
from poc_homography.infrastructure.survey.capture_engine import (
    CaptureContext,
    CaptureEngineError,
    SurveyCaptureEngine,
)
from poc_homography.types import FPS, Degrees, Pixels, Unitless

if TYPE_CHECKING:
    from pathlib import Path

# PTZ-mutating command names that MUST NOT be issued during a burst.
_PTZ_COMMANDS = frozenset(
    {
        "move_absolute",
        "move_relative",
        "move_continuous",
        "set_focus",
        "stop",
        "goto_preset",
        "position3d",
    }
)


def _make_jpeg(width: int, height: int) -> bytes:
    """Return JPEG bytes of a solid image with the given dimensions."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _make_optics() -> ImageOptics:
    return ImageOptics(
        focus=FocusState(style="SEMIAUTOMATIC", focus_limited=150),
        iris=IrisState(level=50, min_level=0, max_level=100),
        exposure=ExposureState(exposure_type="auto", overexpose_suppress=False),
        white_balance=WhiteBalanceState(style="auto", red=50, blue=50),
    )


def _make_device_info() -> DeviceInfo:
    return DeviceInfo(
        model="DS-2DF8425IX-AELW",
        serial_number="SN123456",
        mac_address="00:11:22:33:44:55",
        firmware_version="V5.7.3",
        device_name="cam",
        device_type="IPDome",
        device_id="dev-1",
        device_description="desc",
        encoder_version="E1",
        boot_version="B1",
        hardware_version="H1",
        platform_name="H8",
        manufacturer="Hikvision",
    )


def _make_stream_profile() -> StreamProfile:
    return StreamProfile(
        channel_id=101,
        codec="H.264",
        width=Pixels(2560),
        height=Pixels(1440),
        fps=25.0,
        bitrate_kbps=4096,
        quality_control="VBR",
        transports=["RTSP"],
    )


class FakeCamera:
    """A recording :class:`CameraDevice` double for offline engine tests."""

    def __init__(
        self,
        *,
        snapshot_bytes: bytes | None = None,
        reported: PTZState | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._snapshot_bytes = (
            snapshot_bytes if snapshot_bytes is not None else _make_jpeg(640, 480)
        )
        self._reported = reported or PTZState(
            pan_raw=Degrees(120.1), tilt_deg=Degrees(-15.1), zoom=Unitless(4.0), focus=510
        )

    def move_absolute(
        self, pan: float | None = None, tilt: float | None = None, zoom: float | None = None
    ) -> PTZState:
        self.calls.append("move_absolute")
        return self._reported

    def set_focus(self, focus: int) -> None:
        self.calls.append("set_focus")

    def wait_for_stabilization(self, timeout_s: float = 5.0, threshold: float = 0.1) -> PTZState:
        self.calls.append("wait_for_stabilization")
        return self._reported

    def get_device_info(self) -> DeviceInfo:
        self.calls.append("get_device_info")
        return _make_device_info()

    def get_optics(self) -> ImageOptics:
        self.calls.append("get_optics")
        return _make_optics()

    def get_stream_profiles(self) -> list[StreamProfile]:
        self.calls.append("get_stream_profiles")
        return [_make_stream_profile()]

    def capture_snapshot(self) -> bytes:
        self.calls.append("capture_snapshot")
        return self._snapshot_bytes


class _IncrementingClock:
    """Deterministic UTC clock advancing one second per call."""

    def __init__(self) -> None:
        self._base = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        self._n = 0

    def __call__(self) -> datetime:
        value = self._base + timedelta(seconds=self._n)
        self._n += 1
        return value


def _commanded() -> CommandedState:
    return CommandedState(
        commanded_pan=Degrees(120.0),
        commanded_tilt=Degrees(-15.0),
        commanded_zoom=Unitless(4.0),
        commanded_focus=512,
    )


def _context(previous: CommandedState | None = None) -> CaptureContext:
    return CaptureContext(
        run_id="run-0001",
        camera_id="icozee-camptz-04",
        phase=SurveyPhase.MAIN_SURVEY,
        previous_commanded=previous,
        is_repeatability_sequence=False,
    )


def _engine(camera: FakeCamera) -> SurveyCaptureEngine:
    return SurveyCaptureEngine(
        camera,  # type: ignore[arg-type]
        clock=_IncrementingClock(),
        uuid_factory=lambda: "burst-test",
    )


class TestSnapshotBurst:
    def test_single_snapshot_populates_all_record_fields(self, tmp_path: Path) -> None:
        jpeg = _make_jpeg(640, 480)
        camera = FakeCamera(snapshot_bytes=jpeg)
        prev = CommandedState(
            commanded_pan=Degrees(110.0),
            commanded_tilt=Degrees(-10.0),
            commanded_zoom=Unitless(2.0),
            commanded_focus=500,
        )
        records = _engine(camera).capture_snapshot_burst(
            _commanded(), _context(previous=prev), burst_count=1, output_dir=tmp_path
        )

        assert len(records) == 1
        record = records[0]

        # Commanded vs reported are distinct, non-null fields.
        assert record.commanded.commanded_pan == Degrees(120.0)
        assert record.reported.reported_pan == Degrees(120.1)
        assert record.commanded.commanded_focus == 512
        assert record.reported.reported_focus == 510
        assert record.reported.ptz_settled is True

        # Three distinct timestamps.
        cap = record.capture
        assert cap.timestamp_before_move < cap.timestamp_after_move < cap.timestamp_at_capture
        assert cap.timestamp_at_capture.tzinfo is not None

        # Burst identity.
        assert cap.burst_id == "burst-test"
        assert cap.frame_index == 0
        assert cap.capture_id == "burst-test_0000"

        # Checksum + dimensions correct.
        assert record.image_data.checksum == hashlib.sha256(jpeg).hexdigest()
        assert record.image_data.width == Pixels(640)
        assert record.image_data.height == Pixels(480)
        assert record.image_data.capture_format == "jpeg"
        assert record.image_data.image_path.exists()
        assert record.image_data.image_path.read_bytes() == jpeg

        # Movement context: 120>110 -> cw, -15<-10 -> down, 4>2 -> tele.
        assert record.movement.direction_pan == "cw"
        assert record.movement.direction_tilt == "down"
        assert record.movement.direction_zoom == "tele"
        assert record.movement.prev_pan == Degrees(110.0)
        assert record.movement.settling_delay_s >= 0

        # Pipeline state from stream profile + optics.
        assert record.pipeline.resolution_width == Pixels(2560)
        assert record.pipeline.codec == "H.264"
        assert record.pipeline.fps == FPS(25.0)
        assert record.pipeline.exposure_mode == "auto"
        assert record.pipeline.focus_mode == "SEMIAUTOMATIC"

        # Camera identity from device info + stream profile.
        assert record.camera.brand == "Hikvision"
        assert record.camera.model == "DS-2DF8425IX-AELW"
        assert record.camera.stream_id == "101"

        # Round-trips through the C1 serializer.
        assert record.to_dict()["schema_version"] == record.schema_version

    def test_first_pose_has_none_directions(self, tmp_path: Path) -> None:
        camera = FakeCamera()
        records = _engine(camera).capture_snapshot_burst(
            _commanded(), _context(previous=None), burst_count=1, output_dir=tmp_path
        )
        movement = records[0].movement
        assert movement.direction_pan == "none"
        assert movement.direction_tilt == "none"
        assert movement.direction_zoom == "none"

    def test_burst_shares_burst_id_with_sequential_frame_index(self, tmp_path: Path) -> None:
        camera = FakeCamera()
        records = _engine(camera).capture_snapshot_burst(
            _commanded(), _context(), burst_count=4, output_dir=tmp_path
        )
        assert len(records) == 4
        assert {r.capture.burst_id for r in records} == {"burst-test"}
        assert [r.capture.frame_index for r in records] == [0, 1, 2, 3]
        # Four addressable, distinct files.
        paths = {r.image_data.image_path for r in records}
        assert len(paths) == 4
        assert all(p.exists() for p in paths)

    def test_no_ptz_command_during_burst(self, tmp_path: Path) -> None:
        camera = FakeCamera()
        _engine(camera).capture_snapshot_burst(
            _commanded(), _context(), burst_count=5, output_dir=tmp_path
        )
        first_snapshot = camera.calls.index("capture_snapshot")
        during_burst = camera.calls[first_snapshot:]
        assert not _PTZ_COMMANDS.intersection(during_burst)
        # The per-burst optics/profile snapshot happens before the burst.
        assert "get_optics" in camera.calls[:first_snapshot]
        assert "get_stream_profiles" in camera.calls[:first_snapshot]

    def test_zero_burst_count_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="burst_count must be >= 1"):
            _engine(FakeCamera()).capture_snapshot_burst(
                _commanded(), _context(), burst_count=0, output_dir=tmp_path
            )

    def test_empty_snapshot_raises(self, tmp_path: Path) -> None:
        camera = FakeCamera(snapshot_bytes=b"")
        with pytest.raises(CaptureEngineError, match="no data"):
            _engine(camera).capture_snapshot_burst(
                _commanded(), _context(), burst_count=1, output_dir=tmp_path
            )


class _FakeCapture:
    """A fake ``cv2.VideoCapture`` yielding a fixed sequence of frames."""

    def __init__(self, frames: list[Any]) -> None:
        self._frames = list(frames)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, Any]:
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


class _FakeWriter:
    """A fake ``cv2.VideoWriter`` that records writes and creates the file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.frames_written = 0
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # minimal placeholder segment

    def write(self, frame: Any) -> None:
        self.frames_written += 1

    def release(self) -> None:
        pass


@pytest.fixture()
def _patch_rtsp(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the cv2 I/O seams; return handles the test can inspect."""
    frames = [np.full((48, 64, 3), i * 10, dtype=np.uint8) for i in range(3)]
    capture = _FakeCapture(frames)
    created: dict[str, Any] = {"capture": capture, "writer": None, "n_frames": len(frames)}

    def fake_open_capture(rtsp_url: str) -> _FakeCapture:
        return capture

    def fake_open_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> _FakeWriter:
        writer = _FakeWriter(path)
        created["writer"] = writer
        return writer

    monkeypatch.setattr(capture_engine, "_open_capture", fake_open_capture)
    monkeypatch.setattr(capture_engine, "_open_writer", fake_open_writer)
    return created


class TestVideoBurst:
    def test_video_burst_emits_segment_and_frame_records(
        self, tmp_path: Path, _patch_rtsp: dict[str, Any]
    ) -> None:
        camera = FakeCamera()
        burst_record, records = _engine(camera).capture_video_burst(
            _commanded(),
            _context(),
            rtsp_url="rtsp://camera/stream",
            burst_duration_s=100.0,
            output_dir=tmp_path,
        )

        n = _patch_rtsp["n_frames"]
        assert len(records) == n
        assert [r.capture.frame_index for r in records] == list(range(n))
        assert {r.capture.burst_id for r in records} == {"burst-test"}

        # Segment file written and referenced.
        assert burst_record.segment_path.exists()
        assert burst_record.segment_path == tmp_path / "burst-test.mp4"
        assert _patch_rtsp["writer"].frames_written == n
        assert _patch_rtsp["capture"].released is True

        # Each extracted frame is an addressable file with a valid checksum.
        for record in records:
            assert record.image_data.image_path.exists()
            data = record.image_data.image_path.read_bytes()
            assert record.image_data.checksum == hashlib.sha256(data).hexdigest()
            assert record.image_data.width == Pixels(64)
            assert record.image_data.height == Pixels(48)

        # Burst-level record references every frame.
        assert len(burst_record.frame_refs) == n
        assert burst_record.commanded_state == _commanded()
        assert burst_record.codec == "H.264"
        assert burst_record.frame_by_index(0) is not None

        # duration_s reflects the ACTUAL recorded span (stream ended early),
        # not the requested 100 s target.
        assert 0 < burst_record.duration_s < 100.0

    def test_video_burst_issues_no_ptz_during_capture(
        self, tmp_path: Path, _patch_rtsp: dict[str, Any]
    ) -> None:
        camera = FakeCamera()
        _engine(camera).capture_video_burst(
            _commanded(),
            _context(),
            rtsp_url="rtsp://camera/stream",
            burst_duration_s=100.0,
            output_dir=tmp_path,
        )
        # All PTZ commands precede the per-burst optics snapshot (burst start).
        burst_start = camera.calls.index("get_stream_profiles")
        assert not _PTZ_COMMANDS.intersection(camera.calls[burst_start:])

    def test_video_burst_no_frames_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(capture_engine, "_open_capture", lambda url: _FakeCapture([]))
        with pytest.raises(CaptureEngineError, match="no frames"):
            _engine(FakeCamera()).capture_video_burst(
                _commanded(),
                _context(),
                rtsp_url="rtsp://camera/stream",
                burst_duration_s=100.0,
                output_dir=tmp_path,
            )

    def test_video_burst_open_failure_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _ClosedCapture(_FakeCapture):
            def isOpened(self) -> bool:
                return False

        monkeypatch.setattr(capture_engine, "_open_capture", lambda url: _ClosedCapture([]))
        with pytest.raises(CaptureEngineError, match="Failed to open RTSP"):
            _engine(FakeCamera()).capture_video_burst(
                _commanded(),
                _context(),
                rtsp_url="rtsp://camera/stream",
                burst_duration_s=100.0,
                output_dir=tmp_path,
            )
