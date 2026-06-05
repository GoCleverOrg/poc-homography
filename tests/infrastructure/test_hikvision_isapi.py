"""Mocked, offline tests for the Hikvision ISAPI adapter and value objects.

These tests run by default under ``poe test``. They never touch the network:
fixture XML is loaded from ``tests/fixtures/hikvision/<name>.txt`` and a
``FakeTransport`` substitutes for ``IsapiTransport`` so adapter parsing is
exercised without a live socket.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from defusedxml import ElementTree as ET

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
from poc_homography.infrastructure.clients.hikvision import isapi_endpoints as ep
from poc_homography.infrastructure.clients.hikvision import isapi_units as units
from poc_homography.infrastructure.clients.hikvision.isapi_client import HikvisionISAPIClient
from poc_homography.infrastructure.clients.hikvision.isapi_errors import (
    HikvisionHTTPError,
    HikvisionUnsupportedError,
)
from poc_homography.infrastructure.clients.hikvision.isapi_transport import classify_error_response

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "hikvision"


def load_fixture(name: str) -> Element:
    """Load ``tests/fixtures/hikvision/<name>.txt`` and parse it with defusedxml."""
    text = (FIXTURE_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return ET.fromstring(text)


def load_fixture_text(name: str) -> str:
    """Load the raw text of a fixture (no parsing)."""
    return (FIXTURE_DIR / f"{name}.txt").read_text(encoding="utf-8")


def load_fixture_bytes(name: str) -> bytes:
    """Load the raw bytes of a binary fixture (e.g. snapshot.jpg)."""
    return (FIXTURE_DIR / name).read_bytes()


# --- FakeTransport -------------------------------------------------------------


class FakeTransport:
    """In-memory stand-in for ``IsapiTransport`` returning fixture Elements.

    Routes a request path to the matching fixture by substring match on the
    distinctive ISAPI path segment. ``put_xml`` records every call so XML-body
    building can be asserted.
    """

    def __init__(self) -> None:
        self.put_calls: list[tuple[str, str]] = []

    def get_xml(self, path: str) -> Element:
        """Return the fixture Element matching ``path``."""
        return load_fixture(self._fixture_for(path))

    def get_bytes(self, path: str) -> bytes:
        """Return snapshot bytes for a picture path."""
        if "picture" in path:
            return load_fixture_bytes("snapshot.jpg")
        raise AssertionError(f"unexpected get_bytes path: {path}")

    def put_xml(self, path: str, body: object) -> Element:
        """Record the PUT and return a minimal ResponseStatus element."""
        self.put_calls.append((path, str(body)))
        return ET.fromstring(
            '<ResponseStatus version="2.0" '
            f'xmlns="{ep.HIKVISION_XML_NS}"><statusCode>1</statusCode>'
            "<statusString>OK</statusString></ResponseStatus>"
        )

    @staticmethod
    def _fixture_for(path: str) -> str:
        """Map an ISAPI path to a fixture file name."""
        if path.endswith("/absoluteEx/capabilities"):
            return "ISAPI__PTZCtrl__channels__1__absoluteEx__capabilities"
        if path.endswith("/PTZCtrl/channels/1/status"):
            return "ISAPI__PTZCtrl__channels__1__status"
        if path.endswith("/PTZCtrl/channels/1/presets"):
            return "ISAPI__PTZCtrl__channels__1__presets"
        if path.endswith("/System/deviceInfo"):
            return "ISAPI__System__deviceInfo"
        if path.endswith("/System/status"):
            return "ISAPI__System__status"
        if "/focusConfiguration" in path:
            return "ISAPI__Image__channels__1__focusConfiguration"
        if "/iris" in path:
            return "ISAPI__Image__channels__1__iris"
        if "/exposure" in path:
            return "ISAPI__Image__channels__1__exposure"
        if "/whiteBalance" in path:
            return "ISAPI__Image__channels__1__whiteBalance"
        if "/Streaming/channels/101" in path:
            return "ISAPI__Streaming__channels__101"
        if "/Streaming/channels/102" in path:
            return "ISAPI__Streaming__channels__102"
        raise KeyError(f"no fixture mapped for path: {path}")


def make_client() -> tuple[HikvisionISAPIClient, FakeTransport]:
    """Build a client with its transport replaced by a FakeTransport."""
    client = HikvisionISAPIClient("192.0.2.1", "u", "p")
    fake = FakeTransport()
    client._transport = fake  # type: ignore[assignment]
    return client, fake


# --- VO parsing ----------------------------------------------------------------


def test_device_info_parses() -> None:
    info = DeviceInfo.from_element(load_fixture("ISAPI__System__deviceInfo"))
    assert info.model == "DS-2DF8425IX-AELW"
    assert info.serial_number == "DS-2DF8425IX-AELW20240712CCWRFH3535597"
    assert info.mac_address == "74:3f:c2:fc:a1:9b"
    assert info.firmware_version == "V5.8.0"
    assert info.device_name == "icozee-camptz-04"
    assert info.device_type == "IPDome"
    assert info.platform_name == "H8"
    assert info.manufacturer == "hikvision"


def test_capabilities_real_ranges() -> None:
    caps = CameraCapabilities.from_absolute_ex_element(
        load_fixture("ISAPI__PTZCtrl__channels__1__absoluteEx__capabilities")
    )
    assert caps.tilt_min == -50
    assert caps.tilt_max == 60
    assert caps.pan_min == 0
    assert caps.pan_max == 360
    assert caps.zoom_min == 1
    assert caps.zoom_max == 25
    assert caps.focus_min == 4096
    assert caps.focus_max == 2576990208
    assert caps.pan_speed_min == 0.2
    assert caps.pan_speed_max == 210.8
    assert caps.tilt_speed_min == 0.2
    assert caps.tilt_speed_max == 151.8


def test_focus_state_parses() -> None:
    focus = FocusState.from_element(load_fixture("ISAPI__Image__channels__1__focusConfiguration"))
    assert focus.style == "SEMIAUTOMATIC"
    assert focus.focus_limited == 600


def test_iris_state_parses() -> None:
    iris = IrisState.from_element(load_fixture("ISAPI__Image__channels__1__iris"))
    assert iris.level == 160
    assert iris.min_level == 0
    assert iris.max_level == 100


def test_exposure_state_parses() -> None:
    exposure = ExposureState.from_element(load_fixture("ISAPI__Image__channels__1__exposure"))
    assert exposure.exposure_type == "auto"
    assert exposure.overexpose_suppress is False


def test_white_balance_state_parses() -> None:
    wb = WhiteBalanceState.from_element(load_fixture("ISAPI__Image__channels__1__whiteBalance"))
    assert wb.style == "auto"
    assert wb.red == 50
    assert wb.blue == 50


def test_device_health_parses() -> None:
    health = DeviceHealth.from_element(load_fixture("ISAPI__System__status"))
    assert health.uptime_seconds == 126519
    assert health.cpu_utilization == 56
    assert health.memory_usage == 43
    assert health.heat_state == 0


def test_lens_odometry_parses() -> None:
    odo = LensOdometry.from_element(load_fixture("ISAPI__System__status"))
    assert odo.zoom_reverse_times == 724
    assert odo.zoom_total_steps == 621
    assert odo.focus_reverse_times == 4811
    assert odo.iris_shift_times == 3951
    assert odo.lens_intir_times == 11
    assert odo.camera_run_total_time == 182826
    assert odo.pan_total_rounds == 206
    assert odo.tilt_total_rounds == 134


def test_stream_profile_parses() -> None:
    profile = StreamProfile.from_element(load_fixture("ISAPI__Streaming__channels__101"))
    assert profile.channel_id == 101
    assert profile.codec == "H.264"
    assert profile.width == 2560
    assert profile.height == 1440
    assert profile.fps == 25.0
    assert profile.bitrate_kbps == 6144
    assert profile.quality_control == "VBR"
    assert profile.transports == ["RTSP", "HTTP", "SHTTP", "SRTP"]


def test_ptz_state_from_status_fixture() -> None:
    root = load_fixture("ISAPI__PTZCtrl__channels__1__status")
    azimuth = int(root.findtext(".//h:azimuth", default="0", namespaces=ep.NS))
    elevation = int(root.findtext(".//h:elevation", default="0", namespaces=ep.NS))
    zoom = int(root.findtext(".//h:absoluteZoom", default="0", namespaces=ep.NS))
    state = PTZState(
        pan_raw=units.raw_to_degrees(azimuth),
        tilt_deg=units.raw_to_degrees(elevation),
        zoom=units.raw_to_zoom(zoom),
    )
    assert state.pan_raw == 320.2
    assert state.tilt_deg == 18.2
    assert state.zoom == 17.2


# --- to_dict / from_dict round trips ------------------------------------------


def test_roundtrip_device_info() -> None:
    info = DeviceInfo.from_element(load_fixture("ISAPI__System__deviceInfo"))
    assert DeviceInfo.from_dict(info.to_dict()) == info


def test_roundtrip_capabilities() -> None:
    caps = CameraCapabilities.from_absolute_ex_element(
        load_fixture("ISAPI__PTZCtrl__channels__1__absoluteEx__capabilities")
    )
    assert CameraCapabilities.from_dict(caps.to_dict()) == caps


def test_roundtrip_optics_parts() -> None:
    focus = FocusState.from_element(load_fixture("ISAPI__Image__channels__1__focusConfiguration"))
    iris = IrisState.from_element(load_fixture("ISAPI__Image__channels__1__iris"))
    exposure = ExposureState.from_element(load_fixture("ISAPI__Image__channels__1__exposure"))
    wb = WhiteBalanceState.from_element(load_fixture("ISAPI__Image__channels__1__whiteBalance"))
    assert FocusState.from_dict(focus.to_dict()) == focus
    assert IrisState.from_dict(iris.to_dict()) == iris
    assert ExposureState.from_dict(exposure.to_dict()) == exposure
    assert WhiteBalanceState.from_dict(wb.to_dict()) == wb
    optics = ImageOptics(focus=focus, iris=iris, exposure=exposure, white_balance=wb)
    assert ImageOptics.from_dict(optics.to_dict()) == optics


def test_roundtrip_device_health() -> None:
    health = DeviceHealth.from_element(load_fixture("ISAPI__System__status"))
    assert DeviceHealth.from_dict(health.to_dict()) == health
    odo = LensOdometry.from_element(load_fixture("ISAPI__System__status"))
    assert LensOdometry.from_dict(odo.to_dict()) == odo


def test_roundtrip_stream_profile() -> None:
    profile = StreamProfile.from_element(load_fixture("ISAPI__Streaming__channels__101"))
    assert StreamProfile.from_dict(profile.to_dict()) == profile


def test_roundtrip_ptz_state() -> None:
    state = PTZState(pan_raw=320.2, tilt_deg=18.2, zoom=17.2)
    assert PTZState.from_dict(state.to_dict()) == state


def test_roundtrip_camera_preset() -> None:
    preset = CameraPreset(
        preset_id=1,
        name="Preset 1",
        ptz=PTZState(pan_raw=209.9, tilt_deg=51.1, zoom=1.0),
    )
    assert CameraPreset.from_dict(preset.to_dict()) == preset


# --- isapi_units symmetry ------------------------------------------------------


def test_units_symmetry() -> None:
    for tenth in range(-900, 901):
        x = tenth / 10.0
        assert units.raw_to_degrees(units.degrees_to_raw(x)) == pytest.approx(x)
    for tenth in range(10, 251):
        x = tenth / 10.0
        assert units.raw_to_zoom(units.zoom_to_raw(x)) == pytest.approx(x)


# --- error classification ------------------------------------------------------


def test_classify_403_method_not_allowed() -> None:
    body = load_fixture_text("ISAPI__403__methodNotAllowed")
    err = classify_error_response(403, body)
    assert isinstance(err, HikvisionUnsupportedError)
    assert err.sub_status_code == "methodNotAllowed"


def test_classify_500_is_http_not_unsupported() -> None:
    err = classify_error_response(500, "")
    assert isinstance(err, HikvisionHTTPError)
    assert not isinstance(err, HikvisionUnsupportedError)


# --- adapter parsing via FakeTransport ----------------------------------------


def test_adapter_get_device_info() -> None:
    client, _ = make_client()
    info = client.get_device_info()
    assert info.model == "DS-2DF8425IX-AELW"


def test_adapter_get_capabilities() -> None:
    client, _ = make_client()
    caps = client.get_capabilities()
    assert caps.tilt_min == -50
    assert caps.tilt_max == 60


def test_adapter_get_ptz_status() -> None:
    client, _ = make_client()
    state = client.get_ptz_status()
    assert state.pan_raw == 320.2
    assert state.tilt_deg == 18.2
    assert state.zoom == 17.2
    assert client.last_ptz_state == state


def test_adapter_get_optics() -> None:
    client, _ = make_client()
    optics = client.get_optics()
    assert optics.focus.style == "SEMIAUTOMATIC"
    assert optics.iris.level == 160
    assert optics.exposure.exposure_type == "auto"
    assert optics.white_balance.style == "auto"


def test_adapter_get_health() -> None:
    client, _ = make_client()
    health = client.get_health()
    assert health.uptime_seconds == 126519
    assert health.odometry.zoom_reverse_times == 724


def test_adapter_get_stream_profiles() -> None:
    client, _ = make_client()
    profiles = client.get_stream_profiles()
    assert len(profiles) >= 1
    channel_ids = {p.channel_id for p in profiles}
    assert 101 in channel_ids
    primary = next(p for p in profiles if p.channel_id == 101)
    assert primary.width == 2560


def test_adapter_list_presets_count_and_first() -> None:
    client, _ = make_client()
    presets = client.list_presets()
    assert len(presets) == 36
    first = presets[0]
    assert first.preset_id == 1
    assert first.name == "Preset 1"
    assert first.ptz.tilt_deg == 51.1
    assert first.ptz.pan_raw == 209.9
    assert first.ptz.zoom == 1.0


def test_adapter_capture_snapshot() -> None:
    client, _ = make_client()
    data = client.capture_snapshot()
    assert data[:2] == b"\xff\xd8"
    assert len(data) > 0
