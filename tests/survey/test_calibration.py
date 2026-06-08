"""Phase-0 horizon-calibration tests (offline, with a recording FakeCamera)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.domain.vo.tilt_envelope import TiltEnvelope
from poc_homography.horizon.geometry import (
    DEFAULT_TILT_OFFSET_DEG,
    all_ground_tilt_threshold,
)
from poc_homography.survey.calibration import calibrate_horizon_envelope
from poc_homography.types import Degrees, Unitless

SPEC = CameraSpec.HIKVISION_DS_2DF8425IX


def _jpeg(width: int, height: int) -> bytes:
    """Solid JPEG of the given size (CV refinement falls back to geometry)."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(20, 40, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


class _EchoCamera:
    """A CameraDevice double that echoes the commanded tilt and records pans."""

    def __init__(self) -> None:
        self.commanded_pans: list[float] = []
        self._tilt = 0.0

    def move_absolute(
        self, pan: float | None = None, tilt: float | None = None, zoom: float | None = None
    ) -> PTZState:
        if pan is not None:
            self.commanded_pans.append(pan)
        if tilt is not None:
            self._tilt = tilt
        return PTZState(
            pan_raw=Degrees(pan or 0.0),
            tilt_deg=Degrees(self._tilt),
            zoom=Unitless(zoom or 1.0),
            focus=100,
        )

    def wait_for_stabilization(self, timeout_s: float = 5.0, threshold: float = 0.1) -> PTZState:
        return PTZState(
            pan_raw=Degrees(0.0), tilt_deg=Degrees(self._tilt), zoom=Unitless(1.0), focus=100
        )

    def capture_snapshot(self) -> bytes:
        return _jpeg(int(SPEC.image_width), int(SPEC.image_height))


class TestCalibrateHorizonEnvelope:
    def test_empty_pans_raises(self) -> None:
        with pytest.raises(ValueError, match="pan"):
            calibrate_horizon_envelope(_EchoCamera(), pans=[])

    def test_one_bound_per_pan(self) -> None:
        camera = _EchoCamera()
        env = calibrate_horizon_envelope(camera, pans=[0.0, 90.0, 180.0, 270.0])
        assert isinstance(env, TiltEnvelope)
        assert set(env.bounds) == {0.0, 90.0, 180.0, 270.0}

    def test_pans_are_commanded(self) -> None:
        camera = _EchoCamera()
        calibrate_horizon_envelope(camera, pans=[0.0, 120.0, 240.0])
        assert camera.commanded_pans == [0.0, 120.0, 240.0]

    def test_envelope_metadata_recorded(self) -> None:
        env = calibrate_horizon_envelope(_EchoCamera(), pans=[0.0], zoom=1.0)
        assert env.tilt_offset_deg == float(DEFAULT_TILT_OFFSET_DEG)
        assert env.vfov_deg > 0.0
        assert env.zoom == 1.0

    def test_bound_is_sane(self) -> None:
        # Aimed at the horizon (reported tilt == offset), the detected boundary
        # sits near frame centre, so the bound lands near the flat all-ground
        # threshold (tilt_offset + VFOV/2).
        env = calibrate_horizon_envelope(_EchoCamera(), pans=[0.0], zoom=1.0)
        flat_threshold = float(
            all_ground_tilt_threshold(
                Unitless(1.0),
                SPEC.image_width,
                SPEC.image_height,
                SPEC.sensor_width,
                SPEC.base_focal_length,
                DEFAULT_TILT_OFFSET_DEG,
            )
        )
        assert env.bounds[0.0] == pytest.approx(flat_threshold, abs=1.0)
        # The bound must be a valid, useful (down-ward) tilt above the offset.
        assert env.tilt_offset_deg < env.bounds[0.0] < env.tilt_offset_deg + env.vfov_deg

    def test_pan_reduced_modulo_360(self) -> None:
        env = calibrate_horizon_envelope(_EchoCamera(), pans=[370.0])
        assert set(env.bounds) == {10.0}
