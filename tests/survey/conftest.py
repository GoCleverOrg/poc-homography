"""Shared fixtures for the C4 survey-phase tests.

Provides a recording :class:`CameraDevice` double, deterministic clock/uuid
factories, and an RTSP monkeypatch so the Phase 8 video-burst path runs fully
offline (mirroring ``tests/infrastructure/survey/test_capture_engine.py``).
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image

from poc_homography.domain.vo.camera_capabilities import CameraCapabilities
from poc_homography.domain.vo.camera_preset import CameraPreset
from poc_homography.domain.vo.device_health import DeviceHealth, LensOdometry
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
from poc_homography.types import Degrees, FocusSteps, Pixels, Unitless

if TYPE_CHECKING:
    from collections.abc import Callable


def make_jpeg(width: int = 64, height: int = 48) -> bytes:
    """Return JPEG bytes of a solid image with the given dimensions."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_optics() -> ImageOptics:
    """Build a representative :class:`ImageOptics`."""
    return ImageOptics(
        focus=FocusState(style="SEMIAUTOMATIC", focus_limited=150),
        iris=IrisState(level=50, min_level=0, max_level=100),
        exposure=ExposureState(exposure_type="auto", overexpose_suppress=False),
        white_balance=WhiteBalanceState(style="auto", red=50, blue=50),
    )


def make_device_info() -> DeviceInfo:
    """Build a representative :class:`DeviceInfo`."""
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


def make_capabilities() -> CameraCapabilities:
    """Build a representative :class:`CameraCapabilities`."""
    return CameraCapabilities(
        pan_min=Degrees(0.0),
        pan_max=Degrees(360.0),
        tilt_min=Degrees(-90.0),
        tilt_max=Degrees(90.0),
        zoom_min=Unitless(1.0),
        zoom_max=Unitless(25.0),
        focus_min=FocusSteps(0),
        focus_max=FocusSteps(1000),
        pan_speed_min=0.1,
        pan_speed_max=100.0,
        tilt_speed_min=0.1,
        tilt_speed_max=100.0,
    )


def make_health() -> DeviceHealth:
    """Build a representative :class:`DeviceHealth`."""
    return DeviceHealth(
        uptime_seconds=3600,
        cpu_utilization=12,
        memory_usage=34,
        fan_state=1,
        heat_state=0,
        total_reboot_count=2,
        odometry=LensOdometry(
            zoom_reverse_times=1,
            zoom_total_steps=10,
            focus_reverse_times=2,
            focus_total_steps=20,
            iris_shift_times=3,
            iris_total_steps=30,
            icr_shift_times=4,
            lens_intir_times=5,
            camera_run_total_time=6,
            pan_total_rounds=7,
            tilt_total_rounds=8,
        ),
    )


def make_stream_profile() -> StreamProfile:
    """Build a representative :class:`StreamProfile`."""
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


def make_preset() -> CameraPreset:
    """Build a representative :class:`CameraPreset`."""
    return CameraPreset(
        preset_id=1,
        name="home",
        ptz=PTZState(pan_raw=Degrees(0.0), tilt_deg=Degrees(0.0), zoom=Unitless(1.0), focus=100),
    )


class IncrementingClock:
    """A deterministic clock advancing by a fixed step per call."""

    def __init__(self, step_s: float = 1.0) -> None:
        self._now = datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc)
        self._step = timedelta(seconds=step_s)

    def __call__(self) -> datetime:
        current = self._now
        self._now = self._now + self._step
        return current


class CountingUuid:
    """A deterministic uuid factory yielding stable, unique ids."""

    def __init__(self) -> None:
        self._n = 0

    def __call__(self) -> str:
        self._n += 1
        return f"burst-{self._n:04d}"


class FakeCamera:
    """A recording :class:`CameraDevice` double for offline phase tests."""

    def __init__(self, *, reported: PTZState | None = None) -> None:
        self.calls: list[str] = []
        self._reported = reported or PTZState(
            pan_raw=Degrees(1.0), tilt_deg=Degrees(-1.0), zoom=Unitless(2.0), focus=510
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
        return make_device_info()

    def get_capabilities(self) -> CameraCapabilities:
        self.calls.append("get_capabilities")
        return make_capabilities()

    def get_health(self) -> DeviceHealth:
        self.calls.append("get_health")
        return make_health()

    def get_optics(self) -> ImageOptics:
        self.calls.append("get_optics")
        return make_optics()

    def get_stream_profiles(self) -> list[StreamProfile]:
        self.calls.append("get_stream_profiles")
        return [make_stream_profile()]

    def list_presets(self) -> list[CameraPreset]:
        self.calls.append("list_presets")
        return [make_preset()]

    def capture_snapshot(self) -> bytes:
        self.calls.append("capture_snapshot")
        return make_jpeg()


@pytest.fixture()
def camera() -> FakeCamera:
    """Return a fresh :class:`FakeCamera`."""
    return FakeCamera()


@pytest.fixture()
def clock() -> IncrementingClock:
    """Return a deterministic incrementing clock."""
    return IncrementingClock()


@pytest.fixture()
def uuid_factory() -> CountingUuid:
    """Return a deterministic uuid factory."""
    return CountingUuid()


class _FakeCapture:
    """A fake :class:`cv2.VideoCapture` yielding a fixed list of frames."""

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
    """A fake :class:`cv2.VideoWriter` counting written frames."""

    def __init__(self, path: Any) -> None:
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        self.frames_written = 0

    def write(self, frame: Any) -> None:
        self.frames_written += 1

    def release(self) -> None:
        return None


@pytest.fixture()
def patch_rtsp(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Patch the engine's RTSP open/write seams with in-memory fakes.

    Returns a configurator that (re)installs fakes producing ``n`` frames per
    burst; call it before driving the jitter phase.
    """

    def configure(n_frames: int) -> None:
        def fake_open_capture(rtsp_url: str) -> _FakeCapture:
            frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(n_frames)]
            return _FakeCapture(frames)

        def fake_open_writer(path: Any, fps: float, frame_size: tuple[int, int]) -> _FakeWriter:
            return _FakeWriter(path)

        monkeypatch.setattr(capture_engine, "_open_capture", fake_open_capture)
        monkeypatch.setattr(capture_engine, "_open_writer", fake_open_writer)

    return configure
