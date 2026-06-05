"""PTZ camera abstraction layer for the camera survey app.

Provides a slim webapp-facing wrapper (:class:`HikvisionPTZCamera`) around the
DDD :class:`HikvisionISAPIClient` adapter. All ISAPI protocol details (paths,
namespaces, raw x10 unit conversions) live inside the adapter; this module only
adapts the adapter's DDD value objects to the legacy webapp surface
(:class:`PTZPosition`, :class:`DeviceInfo`) the survey service expects.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from poc_homography.camera_config import get_tenant_credentials
from poc_homography.infrastructure.clients.hikvision import HikvisionISAPIClient
from poc_homography.infrastructure.clients.hikvision.isapi_errors import HikvisionError

from .models import CameraCapabilities, DeviceInfo, PTZPosition

if TYPE_CHECKING:
    from poc_homography.domain.vo.camera_preset import CameraPreset
    from poc_homography.domain.vo.device_health import DeviceHealth
    from poc_homography.domain.vo.image_optics import ImageOptics
    from poc_homography.domain.vo.ptz_state import PTZState
    from poc_homography.domain.vo.stream_profile import StreamProfile

logger = logging.getLogger(__name__)


class BasePTZCamera:
    """Minimal base marker for PTZ camera wrappers.

    Retained only so existing type annotations (e.g. in the diagnostic app and
    the survey service) that reference :class:`BasePTZCamera` keep resolving.
    Concrete wrappers are not required to declare every method abstractly; the
    shared method surface is documented on :class:`HikvisionPTZCamera`.
    """


def _to_position(state: PTZState | None) -> PTZPosition | None:
    """Map a DDD :class:`PTZState` onto the legacy :class:`PTZPosition`.

    Args:
        state: PTZ state from the adapter, or ``None``.

    Returns:
        Equivalent :class:`PTZPosition`, or ``None`` if ``state`` is ``None``.
    """
    if state is None:
        return None
    return PTZPosition(
        pan=float(state.pan_raw),
        tilt=float(state.tilt_deg),
        zoom=float(state.zoom),
    )


class HikvisionPTZCamera(BasePTZCamera):
    """Slim webapp wrapper around :class:`HikvisionISAPIClient`.

    Preserves the legacy webapp-facing method surface (``get_status``,
    ``move_to_position``, ``get_capabilities``, ``wait_for_stabilization``,
    ``get_device_info``, ``move_continuous``, ``stop_movement``) while
    delegating all camera I/O to the underlying DDD adapter, exposed as
    :attr:`client`. New survey pass-throughs (``get_optics``, ``get_health``,
    ``get_stream_profiles``, ``capture_snapshot``, ``list_presets``) forward
    directly to the adapter and return DDD value objects.
    """

    def __init__(
        self,
        camera_ip: str,
        camera_name: str = "Camera",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        """Initialize the wrapper and its underlying adapter.

        Args:
            camera_ip: IP address of the camera.
            camera_name: Display name for logging.
            username: Camera username. If ``None``, falls back to the
                ``CAMERA_USERNAME`` environment variable.
            password: Camera password. If ``None``, falls back to the
                ``CAMERA_PASSWORD`` environment variable.

        Raises:
            ValueError: If camera credentials are not set.
        """
        self.camera_ip = camera_ip
        self.camera_name = camera_name

        username = username or os.getenv("CAMERA_USERNAME")
        password = password or os.getenv("CAMERA_PASSWORD")

        if not username or not password:
            raise ValueError(
                "Camera credentials not set. Set CAMERA_USERNAME and CAMERA_PASSWORD "
                "environment variables."
            )

        self.client = HikvisionISAPIClient(camera_ip, username, password)

    # --- Legacy webapp-facing surface --------------------------------------

    def get_status(self) -> PTZPosition | None:
        """Get current PTZ position.

        Returns:
            Current :class:`PTZPosition`, or ``None`` on adapter error.
        """
        try:
            return _to_position(self.client.get_ptz_status())
        except HikvisionError as e:
            logger.error("Failed to get PTZ status for %s: %s", self.camera_name, e)
            return None

    def get_ptz_state(self) -> PTZState:
        """Get the raw adapter :class:`PTZState` (pan/tilt/zoom, plus focus).

        Unlike :meth:`get_status`, this preserves the full domain value object
        so callers can read fields (e.g. ``focus``) that :class:`PTZPosition`
        does not carry, without reaching into ``self.client``.

        Raises:
            HikvisionError: If the adapter call fails.
        """
        return self.client.get_ptz_status()

    def move_to_position(self, pan: float | None, tilt: float | None, zoom: float | None) -> bool:
        """Move the camera to an absolute position.

        Args:
            pan: Target pan angle in degrees, or ``None`` to keep current.
            tilt: Target tilt angle in degrees, or ``None`` to keep current.
            zoom: Target zoom level, or ``None`` to keep current.

        Returns:
            ``True`` if the move was accepted, ``False`` on adapter error.
        """
        try:
            self.client.move_absolute(pan, tilt, zoom)
            return True
        except HikvisionError as e:
            logger.error("Failed to move %s to position: %s", self.camera_name, e)
            return False

    def get_capabilities(self) -> CameraCapabilities:
        """Get the camera's real PTZ capability envelope.

        Returns:
            The DDD :class:`CameraCapabilities` reported by the camera.

        Raises:
            HikvisionError: If the capability query fails. No fabricated
                defaults are returned.
        """
        return self.client.get_capabilities()

    def wait_for_stabilization(self, max_wait: float = 5.0) -> PTZPosition | None:
        """Wait for the camera to stop moving.

        Args:
            max_wait: Maximum time to wait, in seconds.

        Returns:
            The settled :class:`PTZPosition`, or ``None`` on adapter error.
        """
        try:
            return _to_position(self.client.wait_for_stabilization(timeout_s=max_wait))
        except HikvisionError as e:
            logger.error("Failed waiting for stabilization on %s: %s", self.camera_name, e)
            return None

    def get_device_info(self) -> DeviceInfo | None:
        """Get device hardware information.

        Returns:
            A webapp :class:`DeviceInfo`, or ``None`` on adapter error.
        """
        try:
            info = self.client.get_device_info()
            return DeviceInfo(**info.to_dict())
        except HikvisionError as e:
            logger.error("Failed to get device info for %s: %s", self.camera_name, e)
            return None

    def move_continuous(self, pan: int = 0, tilt: int = 0, zoom: int = 0) -> bool:
        """Move the camera continuously at the given speeds.

        Args:
            pan: Pan speed (-100..100).
            tilt: Tilt speed (-100..100).
            zoom: Zoom speed (-100..100).

        Returns:
            ``True`` if the command was accepted, ``False`` on adapter error.
        """
        try:
            self.client.move_continuous(pan, tilt, zoom)
            return True
        except HikvisionError as e:
            logger.error("Failed move_continuous for %s: %s", self.camera_name, e)
            return False

    def stop_movement(self) -> bool:
        """Stop all PTZ movement.

        Returns:
            ``True`` if the command was accepted, ``False`` on adapter error.
        """
        try:
            self.client.stop()
            return True
        except HikvisionError as e:
            logger.error("Failed stop_movement for %s: %s", self.camera_name, e)
            return False

    # --- New survey pass-throughs (DDD value objects) ----------------------

    def get_optics(self) -> ImageOptics:
        """Read focus, iris, exposure, and white-balance state.

        Returns:
            The DDD :class:`ImageOptics` aggregate.
        """
        return self.client.get_optics()

    def get_health(self) -> DeviceHealth:
        """Read device health metrics and lens odometry.

        Returns:
            The DDD :class:`DeviceHealth` (with ``.odometry``).
        """
        return self.client.get_health()

    def get_stream_profiles(self) -> list[StreamProfile]:
        """Read the available streaming profiles.

        Returns:
            A list of DDD :class:`StreamProfile` objects.
        """
        return self.client.get_stream_profiles()

    def capture_snapshot(self) -> bytes:
        """Capture a JPEG still via the ISAPI picture endpoint.

        Returns:
            Raw JPEG bytes.
        """
        return self.client.capture_snapshot()

    def list_presets(self) -> list[CameraPreset]:
        """List stored PTZ presets.

        Returns:
            A list of DDD ``CameraPreset`` objects.
        """
        return self.client.list_presets()


def create_ptz_camera(
    camera_ip: str,
    camera_name: str,
    camera_model: str | None = None,
    tenant_id: str | None = None,
) -> HikvisionPTZCamera:
    """Create a PTZ camera wrapper for the given camera.

    Args:
        camera_ip: IP address of the camera.
        camera_name: Display name for logging.
        camera_model: Camera model string (reserved for future brand detection).
        tenant_id: Tenant ID used to look up credentials. If ``None``, the
            global ``CAMERA_USERNAME``/``CAMERA_PASSWORD`` fallback is used.

    Returns:
        A :class:`HikvisionPTZCamera` wrapping a :class:`HikvisionISAPIClient`.

    Raises:
        ValueError: If camera credentials are not set.
    """
    del camera_model  # Reserved for future brand detection.

    username, password = None, None
    if tenant_id:
        username, password = get_tenant_credentials(tenant_id)

    return HikvisionPTZCamera(
        camera_ip=camera_ip,
        camera_name=camera_name,
        username=username,
        password=password,
    )
