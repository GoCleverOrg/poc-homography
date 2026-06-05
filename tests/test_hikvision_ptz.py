"""XML-body-building tests for HikvisionISAPIClient PTZ writes.

Replaces the legacy ``ptz_discovery_and_control.HikvisionPTZ`` suite. A capturing
fake transport records ``put_xml`` calls so the absolute-move and position3D XML
bodies (and their raw x10 encoding) can be asserted offline, with no network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.domain.vo.ptz_state import PTZState
from poc_homography.infrastructure.clients.hikvision import isapi_endpoints as ep
from poc_homography.infrastructure.clients.hikvision.isapi_client import HikvisionISAPIClient

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element


class CapturingTransport:
    """Fake transport that records ``put_xml`` calls and serves a PTZ status.

    ``move_absolute`` reads the current status (so unset axes can be filled) and
    re-reads after the PUT; this fake answers both ``get_xml`` calls with a fixed
    status while capturing every PUT path and body.
    """

    def __init__(self, pan: float = 0.0, tilt: float = 0.0, zoom: float = 1.0) -> None:
        self._status = (
            f'<PTZStatus version="2.0" xmlns="{ep.HIKVISION_XML_NS}">'
            "<AbsoluteHigh>"
            f"<elevation>{round(tilt * 10)}</elevation>"
            f"<azimuth>{round(pan * 10)}</azimuth>"
            f"<absoluteZoom>{round(zoom * 10)}</absoluteZoom>"
            "</AbsoluteHigh></PTZStatus>"
        )
        self.put_calls: list[tuple[str, str]] = []

    def get_xml(self, path: str) -> Element:
        from defusedxml import ElementTree as ET

        return ET.fromstring(self._status)

    def put_xml(self, path: str, body: object) -> Element:
        from defusedxml import ElementTree as ET

        self.put_calls.append((path, str(body)))
        return ET.fromstring(
            f'<ResponseStatus version="2.0" xmlns="{ep.HIKVISION_XML_NS}">'
            "<statusString>OK</statusString></ResponseStatus>"
        )


def _make_client(transport: CapturingTransport) -> HikvisionISAPIClient:
    client = HikvisionISAPIClient("192.0.2.1", "u", "p")
    client._transport = transport  # type: ignore[assignment]
    return client


def test_move_absolute_builds_raw_x10_xml() -> None:
    transport = CapturingTransport()
    client = _make_client(transport)

    client.move_absolute(pan=60.0, tilt=-2.0, zoom=3.2)

    path, body = transport.put_calls[0]
    assert path == ep.ptz_absolute()
    assert "<azimuth>600</azimuth>" in body
    assert "<elevation>-20</elevation>" in body
    assert "<absoluteZoom>32</absoluteZoom>" in body


def test_move_absolute_fills_unset_axes_from_current() -> None:
    transport = CapturingTransport(pan=10.5, tilt=4.0, zoom=2.0)
    client = _make_client(transport)

    # Only pan supplied; tilt/zoom must be filled from current status.
    client.move_absolute(pan=10.5)

    _, body = transport.put_calls[0]
    assert "<azimuth>105</azimuth>" in body
    assert "<elevation>40</elevation>" in body
    assert "<absoluteZoom>20</absoluteZoom>" in body


def test_move_relative_applies_delta_to_current() -> None:
    transport = CapturingTransport(pan=10.0, tilt=5.0, zoom=2.0)
    client = _make_client(transport)
    client.get_ptz_status()  # prime the cache

    client.move_relative(pan_delta=1.0)

    _, body = transport.put_calls[0]
    assert "<azimuth>110</azimuth>" in body
    assert "<elevation>50</elevation>" in body
    assert "<absoluteZoom>20</absoluteZoom>" in body


def test_position3d_builds_expected_xml() -> None:
    transport = CapturingTransport()
    client = _make_client(transport)

    client.position3d(0.1, 0.2, 0.3, 0.4)

    path, body = transport.put_calls[0]
    assert path == ep.ptz_position3d()
    assert "<positionX>0.1</positionX>" in body
    assert "<positionY>0.2</positionY>" in body
    assert "<positionX>0.3</positionX>" in body
    assert "<positionY>0.4</positionY>" in body


def test_move_absolute_returns_ptz_state() -> None:
    transport = CapturingTransport(pan=60.0, tilt=-2.0, zoom=3.2)
    client = _make_client(transport)

    state = client.move_absolute(pan=60.0, tilt=-2.0, zoom=3.2)

    assert isinstance(state, PTZState)
    assert state.pan_raw == 60.0
    assert state.tilt_deg == -2.0
    assert state.zoom == 3.2
