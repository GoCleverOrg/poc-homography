"""Live integration tests for the Hikvision ISAPI adapter.

Gated behind the ``ICO_CAMERA_IP`` environment variable: the whole module is
skipped when it is unset, so the default ``poe test`` run stays offline. Write
operations require a SECOND opt-in flag, ``ICO_CAMERA_WRITE``, because they
physically move the camera.

Required env:
- ``ICO_CAMERA_IP`` -- camera host/IP (presence enables read tests).
- ``ICO_CAMERA_USERNAME`` / ``ICO_CAMERA_PASSWORD`` -- credentials.
- ``ICO_CAMERA_WRITE`` -- set (any value) to enable the write/restore test.
"""

from __future__ import annotations

import io
import os

import pytest

from poc_homography.infrastructure.clients.hikvision.isapi_client import HikvisionISAPIClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("ICO_CAMERA_IP"),
        reason="ICO_CAMERA_IP not set (live camera unavailable)",
    ),
]


def make_live_client() -> HikvisionISAPIClient:
    """Build a client from the ICO_CAMERA_* environment variables."""
    host = os.environ["ICO_CAMERA_IP"]
    username = os.getenv("ICO_CAMERA_USERNAME", "admin")
    password = os.getenv("ICO_CAMERA_PASSWORD", "")
    return HikvisionISAPIClient(host, username, password)


def test_live_get_device_info() -> None:
    info = make_live_client().get_device_info()
    assert info.model
    assert info.serial_number


def test_live_get_capabilities() -> None:
    caps = make_live_client().get_capabilities()
    assert caps.pan_max > caps.pan_min
    assert caps.tilt_max > caps.tilt_min
    assert caps.zoom_max > caps.zoom_min


def test_live_get_ptz_status() -> None:
    state = make_live_client().get_ptz_status()
    assert state.pan_raw is not None
    assert state.tilt_deg is not None
    assert state.zoom is not None


def test_live_get_optics() -> None:
    optics = make_live_client().get_optics()
    assert optics.focus is not None
    assert optics.iris is not None
    assert optics.exposure is not None
    assert optics.white_balance is not None


def test_live_get_health() -> None:
    health = make_live_client().get_health()
    assert health.uptime_seconds >= 0


def test_live_list_presets() -> None:
    presets = make_live_client().list_presets()
    assert isinstance(presets, list)


def test_live_capture_snapshot() -> None:
    data = make_live_client().capture_snapshot()
    assert data[:2] == b"\xff\xd8"
    assert len(data) > 0
    width, height = _decode_dimensions(data)
    assert width == 2560
    assert height == 1440


def _decode_dimensions(data: bytes) -> tuple[int, int]:
    """Decode a JPEG to (width, height) using opencv, falling back to Pillow."""
    try:
        import cv2
        import numpy as np

        array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if array is None:
            raise ValueError("cv2 failed to decode snapshot")
        height, width = array.shape[:2]
        return int(width), int(height)
    except ImportError:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return image.width, image.height


@pytest.mark.skipif(
    not os.getenv("ICO_CAMERA_WRITE"),
    reason="ICO_CAMERA_WRITE not set (write/move test disabled)",
)
def test_live_move_relative_and_restore() -> None:
    client = make_live_client()
    original = client.get_ptz_status()
    try:
        client.move_relative(pan_delta=1.0)
        client.wait_for_stabilization()
    finally:
        client.move_absolute(
            pan=original.pan_raw,
            tilt=original.tilt_deg,
            zoom=original.zoom,
        )
        client.wait_for_stabilization()
