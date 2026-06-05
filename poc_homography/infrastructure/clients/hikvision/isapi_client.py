"""Hikvision ISAPI camera adapter implementing :class:`CameraDevice`.

This is the single concrete adapter for Hikvision PTZ cameras. It composes the
Phase A building blocks:

- :mod:`isapi_endpoints` for every ISAPI path and the XML namespace,
- :mod:`isapi_units` for the raw x10 <-> degrees/zoom conversions,
- :class:`IsapiTransport` for HTTP digest auth, timeouts, and error mapping,

and the Phase B value objects for parsing responses. No ISAPI path literal, no
namespace literal, and no raw divide/multiply-by-10 unit math appear in this
module; all of that lives in the helper modules above.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from poc_homography.domain.vo.camera_capabilities import CameraCapabilities
from poc_homography.domain.vo.camera_preset import CameraPreset
from poc_homography.domain.vo.device_health import DeviceHealth
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
from poc_homography.infrastructure.clients.hikvision.isapi_errors import HikvisionError
from poc_homography.infrastructure.clients.hikvision.isapi_transport import (
    IsapiTransport,
    find,
    findtext,
)

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

    from poc_homography.domain.entities.camera_config import CameraConfig
    from poc_homography.domain.protocols.camera_device import CameraDevice


def _ptz_state_from_absolute_high(elem: Element) -> PTZState:
    """Build a :class:`PTZState` from an ``AbsoluteHigh`` raw-x10 element.

    The ``azimuth`` (pan), ``elevation`` (tilt), and ``absoluteZoom`` values are
    raw integers scaled x10; the unit conversion is delegated to
    :mod:`isapi_units`.

    Args:
        elem: An element containing an ``AbsoluteHigh`` block (status, preset).

    Returns:
        The decoded PTZ state (focus left ``None``).

    Raises:
        HikvisionError: If any of the three PTZ values is missing.
    """
    azimuth = findtext(elem, ".//h:azimuth")
    elevation = findtext(elem, ".//h:elevation")
    absolute_zoom = findtext(elem, ".//h:absoluteZoom")
    if azimuth is None or elevation is None or absolute_zoom is None:
        raise HikvisionError("Missing PTZ values in AbsoluteHigh element")
    return PTZState(
        pan_raw=units.raw_to_degrees(int(azimuth)),
        tilt_deg=units.raw_to_degrees(int(elevation)),
        zoom=units.raw_to_zoom(int(absolute_zoom)),
    )


class HikvisionISAPIClient:
    """Hikvision ISAPI adapter implementing :class:`CameraDevice`.

    Wraps a single :class:`IsapiTransport` against one camera. Reads and writes
    PTZ position, optics, presets, identity, health, stream profiles, and
    snapshots over the Hikvision ISAPI protocol.

    Attributes:
        channel: PTZ / image channel (default 1).
    """

    POLL_INTERVAL = 0.1
    """Seconds between PTZ status polls during stabilization."""

    STABLE_THRESHOLD = 0.5
    """Seconds of sub-threshold change required to consider the position stable."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        channel: int = 1,
        timeout: float = 5.0,
    ) -> None:
        """Initialize the adapter and its transport.

        Args:
            host: Camera host or IP address (no scheme).
            username: ISAPI account username.
            password: ISAPI account password.
            channel: PTZ / image channel to operate on.
            timeout: Per-request timeout in seconds.
        """
        self.channel = channel
        self._transport = IsapiTransport(host, username, password, timeout=timeout)
        self._last_ptz_state: PTZState | None = None

    @classmethod
    def from_config(
        cls,
        camera_config: CameraConfig,
        *,
        timeout: float = 5.0,
    ) -> HikvisionISAPIClient:
        """Build a client from a :class:`CameraConfig`.

        Args:
            camera_config: Camera configuration with IP address and credentials.
            timeout: Per-request timeout in seconds.

        Returns:
            A configured :class:`HikvisionISAPIClient`.

        Raises:
            ValueError: If ``camera_config`` has no IP address.
        """
        if not camera_config.ip_address:
            raise ValueError(f"Camera '{camera_config.name}' has no IP address")
        return cls(
            camera_config.ip_address,
            camera_config.credential.username,
            camera_config.credential.password,
            timeout=timeout,
        )

    # --- XML helpers -------------------------------------------------------

    @staticmethod
    def _xml(body: str) -> str:
        """Wrap a body fragment in a namespaced ``PTZData`` document.

        The namespace string is sourced from :mod:`isapi_endpoints`; it is never
        hardcoded in this module.

        Args:
            body: The inner XML (e.g. an ``AbsoluteHigh`` block).

        Returns:
            A complete ``PTZData`` XML document as a string.
        """
        return f'<PTZData version="2.0" xmlns="{ep.HIKVISION_XML_NS}">{body}</PTZData>'

    # --- Identity / health -------------------------------------------------

    def get_device_info(self) -> DeviceInfo:
        """Read device identity and firmware metadata."""
        root = self._transport.get_xml(ep.device_info())
        return DeviceInfo.from_element(root)

    def get_capabilities(self) -> CameraCapabilities:
        """Read the camera's degree-based PTZ capability envelope."""
        root = self._transport.get_xml(ep.ptz_absolute_ex_capabilities(self.channel))
        return CameraCapabilities.from_absolute_ex_element(root)

    def get_health(self) -> DeviceHealth:
        """Read runtime health metrics and lens odometry."""
        root = self._transport.get_xml(ep.system_status())
        return DeviceHealth.from_element(root)

    def get_stream_profiles(self) -> list[StreamProfile]:
        """Read available streaming profiles (channel 101, plus 102 if present)."""
        profiles = [StreamProfile.from_element(self._transport.get_xml(ep.streaming_channel(101)))]
        try:
            profiles.append(
                StreamProfile.from_element(self._transport.get_xml(ep.streaming_channel(102)))
            )
        except HikvisionError:
            pass
        return profiles

    # --- PTZ read ----------------------------------------------------------

    @property
    def last_ptz_state(self) -> PTZState | None:
        """Get the last known PTZ state (in-memory cache)."""
        return self._last_ptz_state

    def get_ptz_status(self) -> PTZState:
        """Read current PTZ status and update the cache."""
        root = self._transport.get_xml(ep.ptz_status(self.channel))
        state = _ptz_state_from_absolute_high(root)
        self._last_ptz_state = state
        return state

    def wait_for_stabilization(
        self,
        timeout_s: float = 5.0,
        threshold: float = 0.1,
    ) -> PTZState:
        """Poll PTZ status until the position is stable or ``timeout_s`` elapses.

        The position is stable once consecutive readings differ by less than
        ``threshold`` on every axis for ``STABLE_THRESHOLD`` seconds.
        """
        start = time.time()
        last: PTZState | None = None
        stable_since: float | None = None

        while time.time() - start < timeout_s:
            current = self.get_ptz_status()
            if last is not None:
                changed = (
                    abs(current.pan_raw - last.pan_raw) > threshold
                    or abs(current.tilt_deg - last.tilt_deg) > threshold
                    or abs(current.zoom - last.zoom) > threshold
                )
                if not changed:
                    if stable_since is None:
                        stable_since = time.time()
                    elif time.time() - stable_since >= self.STABLE_THRESHOLD:
                        return current
                else:
                    stable_since = None
            last = current
            time.sleep(self.POLL_INTERVAL)

        return self.get_ptz_status()

    # --- PTZ write ---------------------------------------------------------

    def move_absolute(
        self,
        pan: float | None = None,
        tilt: float | None = None,
        zoom: float | None = None,
    ) -> PTZState:
        """Move to an absolute PTZ position, filling unset axes from current."""
        current = self._last_ptz_state or self.get_ptz_status()
        target_pan = pan if pan is not None else current.pan_raw
        target_tilt = tilt if tilt is not None else current.tilt_deg
        target_zoom = zoom if zoom is not None else current.zoom

        body = (
            "<AbsoluteHigh>"
            f"<elevation>{units.degrees_to_raw(target_tilt)}</elevation>"
            f"<azimuth>{units.degrees_to_raw(target_pan)}</azimuth>"
            f"<absoluteZoom>{units.zoom_to_raw(target_zoom)}</absoluteZoom>"
            "</AbsoluteHigh>"
        )
        self._transport.put_xml(ep.ptz_absolute(self.channel), self._xml(body))
        return self.get_ptz_status()

    def move_relative(
        self,
        pan_delta: float = 0.0,
        tilt_delta: float = 0.0,
        zoom_delta: float = 0.0,
    ) -> PTZState:
        """Move relative to the current position via :meth:`move_absolute`."""
        current = self._last_ptz_state or self.get_ptz_status()
        return self.move_absolute(
            pan=current.pan_raw + pan_delta,
            tilt=current.tilt_deg + tilt_delta,
            zoom=current.zoom + zoom_delta,
        )

    def move_continuous(
        self,
        pan_speed: int = 0,
        tilt_speed: int = 0,
        zoom_speed: int = 0,
    ) -> None:
        """Start continuous movement; speeds are clamped to -100..100."""
        pan = max(-100, min(100, pan_speed))
        tilt = max(-100, min(100, tilt_speed))
        zoom = max(-100, min(100, zoom_speed))
        body = f"<pan>{pan}</pan><tilt>{tilt}</tilt><zoom>{zoom}</zoom>"
        self._transport.put_xml(ep.ptz_continuous(self.channel), self._xml(body))

    def stop(self) -> None:
        """Stop all PTZ movement (continuous with zero speeds)."""
        self.move_continuous(0, 0, 0)

    def goto_preset(self, preset_id: int) -> None:
        """Recall a stored preset position."""
        self._transport.put_xml(
            ep.ptz_preset_goto(preset_id, self.channel),
            self._xml(""),
        )

    def position3d(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> None:
        """Issue a 3D drag-zoom positioning command."""
        body = (
            f'<Position3D version="2.0" xmlns="{ep.HIKVISION_XML_NS}">'
            "<StartPoint>"
            f"<positionX>{start_x}</positionX>"
            f"<positionY>{start_y}</positionY>"
            "</StartPoint>"
            "<EndPoint>"
            f"<positionX>{end_x}</positionX>"
            f"<positionY>{end_y}</positionY>"
            "</EndPoint>"
            "</Position3D>"
        )
        self._transport.put_xml(ep.ptz_position3d(self.channel), body)

    # --- Optics read / write ----------------------------------------------

    def get_optics(self) -> ImageOptics:
        """Read focus, iris, exposure, and white-balance state."""
        focus = FocusState.from_element(
            self._transport.get_xml(ep.image_focus_configuration(self.channel))
        )
        iris = IrisState.from_element(self._transport.get_xml(ep.image_iris(self.channel)))
        exposure = ExposureState.from_element(
            self._transport.get_xml(ep.image_exposure(self.channel))
        )
        white_balance = WhiteBalanceState.from_element(
            self._transport.get_xml(ep.image_white_balance(self.channel))
        )
        return ImageOptics(
            focus=focus,
            iris=iris,
            exposure=exposure,
            white_balance=white_balance,
        )

    def set_focus(self, focus: int) -> None:
        """Set the focus position (may be unsupported by firmware)."""
        body = (
            f'<FocusConfiguration xmlns="{ep.HIKVISION_XML_NS}">'
            f"<focus>{focus}</focus>"
            "</FocusConfiguration>"
        )
        self._transport.put_xml(ep.image_focus_configuration(self.channel), body)

    def set_iris(self, level: int) -> None:
        """Set the iris level (may be unsupported by firmware)."""
        body = f'<Iris xmlns="{ep.HIKVISION_XML_NS}"><IrisLevel>{level}</IrisLevel></Iris>'
        self._transport.put_xml(ep.image_iris(self.channel), body)

    def set_exposure(self, exposure_type: str) -> None:
        """Set the exposure mode (may be unsupported by firmware)."""
        body = (
            f'<Exposure xmlns="{ep.HIKVISION_XML_NS}">'
            f"<ExposureType>{exposure_type}</ExposureType>"
            "</Exposure>"
        )
        self._transport.put_xml(ep.image_exposure(self.channel), body)

    # --- Presets -----------------------------------------------------------

    def list_presets(self) -> list[CameraPreset]:
        """List stored PTZ presets, converting raw values to degrees/zoom here."""
        root = self._transport.get_xml(ep.ptz_presets(self.channel))
        presets: list[CameraPreset] = []
        for preset_elem in root.findall(".//h:PTZPreset", ep.NS):
            preset_id = findtext(preset_elem, "h:id")
            name = findtext(preset_elem, "h:presetName")
            absolute_high = find(preset_elem, "h:AbsoluteHigh")
            if preset_id is None or absolute_high is None:
                continue
            presets.append(
                CameraPreset(
                    preset_id=int(preset_id),
                    name=name or "",
                    ptz=_ptz_state_from_absolute_high(absolute_high),
                )
            )
        return presets

    # --- Capture -----------------------------------------------------------

    def capture_snapshot(self) -> bytes:
        """Capture a JPEG still from the primary streaming channel."""
        return self._transport.get_bytes(ep.streaming_picture(101))


if TYPE_CHECKING:
    # Static-only structural conformance check: pyright errors here if
    # HikvisionISAPIClient stops satisfying the CameraDevice protocol. No
    # runtime instantiation occurs.
    def _assert_camera_device(client: HikvisionISAPIClient) -> CameraDevice:
        return client
