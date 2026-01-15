"""Hikvision PTZ camera controller implementation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import requests
from requests.auth import HTTPDigestAuth

from poc_homography.domain.protocols import CameraControllerError
from poc_homography.domain.vo.ptz_state import PTZState

if TYPE_CHECKING:
    from poc_homography.domain.entities.camera_config import CameraConfig


class HikvisionCameraController:
    """PTZ camera controller for Hikvision cameras using ISAPI.

    This client implements the CameraController protocol, providing
    hardware communication with Hikvision PTZ cameras.

    Attributes:
        camera_config: Camera configuration with IP and credentials.
        timeout: Request timeout in seconds.
    """

    _HIKVISION_NS = {"h": "http://www.hikvision.com/ver20/XMLSchema"}

    def __init__(
        self,
        camera_config: CameraConfig,
        timeout: float = 5.0,
    ) -> None:
        """Initialize the Hikvision camera controller.

        Args:
            camera_config: Camera configuration entity.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If camera_config has no IP address.
        """
        if not camera_config.ip_address:
            raise ValueError(f"Camera '{camera_config.name}' has no IP address")

        self._config = camera_config
        self._timeout = timeout
        self._last_ptz_state: PTZState | None = None

    @property
    def last_ptz_state(self) -> PTZState | None:
        """Get the last known PTZ state."""
        return self._last_ptz_state

    def get_ptz_status(self) -> PTZState:
        """Read current PTZ status from camera."""
        url = f"http://{self._config.ip_address}/ISAPI/PTZCtrl/channels/1/status"

        try:
            response = requests.get(
                url,
                auth=HTTPDigestAuth(
                    self._config.credential.username,
                    self._config.credential.password,
                ),
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            raise CameraControllerError(f"Failed to connect: {e}") from e

        if response.status_code != 200:
            raise CameraControllerError(f"Camera returned status {response.status_code}")

        state = self._parse_ptz_status(response.text)
        self._last_ptz_state = state
        return state

    def move_absolute(
        self,
        pan: float | None = None,
        tilt: float | None = None,
        zoom: float | None = None,
    ) -> PTZState:
        """Move camera to absolute position."""
        url = f"http://{self._config.ip_address}/ISAPI/PTZCtrl/channels/1/absolute"

        # Get current state if any value is None
        current = self._last_ptz_state or self.get_ptz_status()

        target_pan = pan if pan is not None else current.pan_raw
        target_tilt = tilt if tilt is not None else current.tilt_deg
        target_zoom = zoom if zoom is not None else current.zoom

        xml_body = self._build_absolute_xml(target_pan, target_tilt, target_zoom)

        try:
            response = requests.put(
                url,
                data=xml_body,
                auth=HTTPDigestAuth(
                    self._config.credential.username,
                    self._config.credential.password,
                ),
                headers={"Content-Type": "application/xml"},
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            raise CameraControllerError(f"Move failed: {e}") from e

        if response.status_code != 200:
            raise CameraControllerError(f"Move returned status {response.status_code}")

        # Return updated status after move
        return self.get_ptz_status()

    def move_relative(
        self,
        pan_delta: float = 0.0,
        tilt_delta: float = 0.0,
        zoom_delta: float = 0.0,
    ) -> PTZState:
        """Move camera relative to current position."""
        current = self._last_ptz_state or self.get_ptz_status()

        return self.move_absolute(
            pan=current.pan_raw + pan_delta,
            tilt=current.tilt_deg + tilt_delta,
            zoom=current.zoom + zoom_delta,
        )

    def _parse_ptz_status(self, xml_text: str) -> PTZState:
        """Parse PTZ status XML response."""
        try:
            root = ET.fromstring(xml_text)

            azimuth = root.find(".//h:azimuth", self._HIKVISION_NS)
            elevation = root.find(".//h:elevation", self._HIKVISION_NS)
            abs_zoom = root.find(".//h:absoluteZoom", self._HIKVISION_NS)

            if any(el is None or el.text is None for el in [azimuth, elevation, abs_zoom]):
                raise CameraControllerError("Missing PTZ values in response")

            # After the guard above, we know these are not None
            assert azimuth is not None and azimuth.text is not None
            assert elevation is not None and elevation.text is not None
            assert abs_zoom is not None and abs_zoom.text is not None

            return PTZState(
                pan_raw=float(azimuth.text) / 10,
                tilt_deg=float(elevation.text) / 10,
                zoom=float(abs_zoom.text) / 10,
            )
        except ET.ParseError as e:
            raise CameraControllerError(f"XML parse error: {e}") from e

    def _build_absolute_xml(
        self,
        pan: float,
        tilt: float,
        zoom: float,
    ) -> str:
        """Build XML for absolute positioning command."""
        # Hikvision ISAPI uses values * 10
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<PTZData xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <AbsoluteHigh>
        <azimuth>{int(pan * 10)}</azimuth>
        <elevation>{int(tilt * 10)}</elevation>
        <absoluteZoom>{int(zoom * 10)}</absoluteZoom>
    </AbsoluteHigh>
</PTZData>"""
