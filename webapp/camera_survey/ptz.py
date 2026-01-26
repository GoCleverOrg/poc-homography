"""PTZ camera abstraction layer.

Provides abstract base class for PTZ camera control to support
multi-brand camera support in the future.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ

from poc_homography.camera_config import get_tenant_credentials

from .models import CameraCapabilities, DeviceInfo, PTZPosition

logger = logging.getLogger(__name__)


class BasePTZCamera(ABC):
    """Abstract base class for PTZ camera control.

    Provides a brand-agnostic interface for PTZ operations used by
    the survey service. Concrete implementations wrap vendor-specific
    SDKs/APIs.
    """

    @abstractmethod
    def get_status(self) -> PTZPosition | None:
        """Get current PTZ position.

        Returns:
            PTZPosition with pan, tilt, zoom values, or None if failed.
        """

    @abstractmethod
    def move_to_position(self, pan: float | None, tilt: float | None, zoom: float | None) -> bool:
        """Move camera to absolute position.

        Args:
            pan: Pan angle in degrees (0-360), or None to keep current
            tilt: Tilt angle in degrees, or None to keep current
            zoom: Zoom level, or None to keep current

        Returns:
            True if movement command was accepted, False otherwise.
        """

    @abstractmethod
    def get_capabilities(self) -> CameraCapabilities:
        """Get camera PTZ capabilities (ranges).

        Returns:
            CameraCapabilities with valid ranges for pan, tilt, zoom.
        """

    @abstractmethod
    def wait_for_stabilization(self, max_wait: float = 5.0) -> PTZPosition | None:
        """Wait for camera to reach target position and stabilize.

        Args:
            max_wait: Maximum time to wait in seconds.

        Returns:
            Final PTZPosition after stabilization, or None if failed.
        """

    @abstractmethod
    def get_device_info(self) -> DeviceInfo | None:
        """Get device hardware information (model, serial, MAC, firmware).

        Returns:
            DeviceInfo with hardware details, or None if failed.
        """


class HikvisionPTZCamera(BasePTZCamera):
    """Hikvision PTZ camera implementation.

    Wraps the existing HikvisionPTZ class to provide BasePTZCamera interface.
    """

    # PTZ stabilization parameters
    POLL_INTERVAL = 0.1  # Seconds between status polls
    STABLE_THRESHOLD = 0.5  # Seconds of no change to consider stable
    POSITION_TOLERANCE = 0.1  # Degrees tolerance for position comparison

    def __init__(
        self,
        camera_ip: str,
        camera_name: str = "Camera",
        username: str | None = None,
        password: str | None = None,
    ):
        """Initialize Hikvision PTZ camera.

        Args:
            camera_ip: IP address of the camera.
            camera_name: Display name for logging.
            username: Camera username. If None, falls back to CAMERA_USERNAME env var.
            password: Camera password. If None, falls back to CAMERA_PASSWORD env var.

        Raises:
            ValueError: If camera credentials are not set.
        """
        self.camera_ip = camera_ip
        self.camera_name = camera_name

        # Use provided credentials or fall back to environment variables
        username = username or os.getenv("CAMERA_USERNAME")
        password = password or os.getenv("CAMERA_PASSWORD")

        if not username or not password:
            raise ValueError(
                "Camera credentials not set. Set CAMERA_USERNAME and CAMERA_PASSWORD environment variables."
            )

        # Create underlying HikvisionPTZ instance
        self._ptz = HikvisionPTZ(
            ip=camera_ip,
            username=username,
            password=password,
            name=camera_name,
        )

        # Default capabilities for Hikvision DS-2DF8425IX series
        # Using wide tilt range to support various camera models
        self._capabilities = CameraCapabilities(
            pan_min=0.0,
            pan_max=360.0,
            tilt_min=-90.0,
            tilt_max=90.0,
            zoom_min=1.0,
            zoom_max=25.0,
        )

    def get_status(self) -> PTZPosition | None:
        """Get current PTZ position from Hikvision camera."""
        try:
            status = self._ptz.get_status()
            if status:
                return PTZPosition(
                    pan=status.get("pan"),
                    tilt=status.get("tilt"),
                    zoom=status.get("zoom"),
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get PTZ status for {self.camera_name}: {e}")
            return None

    def move_to_position(self, pan: float | None, tilt: float | None, zoom: float | None) -> bool:
        """Move Hikvision camera to absolute position.

        Uses the send_ptz_return method which sends absolute positioning commands.
        """
        try:
            # Build status dict for send_ptz_return
            # If any value is None, get current position for that axis
            current = self.get_status()
            if current is None:
                logger.error(f"Cannot move {self.camera_name}: failed to get current position")
                return False

            target = {
                "pan": pan if pan is not None else current.pan,
                "tilt": tilt if tilt is not None else current.tilt,
                "zoom": zoom if zoom is not None else current.zoom,
            }

            return self._ptz.send_ptz_return(target)
        except Exception as e:
            logger.error(f"Failed to move {self.camera_name} to position: {e}")
            return False

    def get_capabilities(self) -> CameraCapabilities:
        """Get Hikvision camera capabilities from camera.

        Queries the camera's ISAPI endpoint for actual capabilities.
        Falls back to default values if query fails.
        """
        try:
            caps = self._ptz.get_capabilities()
            if caps:
                return CameraCapabilities(
                    pan_min=caps["pan"]["min"] if caps["pan"]["min"] is not None else 0.0,
                    pan_max=caps["pan"]["max"] if caps["pan"]["max"] is not None else 360.0,
                    tilt_min=caps["tilt"]["min"] if caps["tilt"]["min"] is not None else -90.0,
                    tilt_max=caps["tilt"]["max"] if caps["tilt"]["max"] is not None else 90.0,
                    zoom_min=caps["zoom"]["min"] if caps["zoom"]["min"] is not None else 1.0,
                    zoom_max=caps["zoom"]["max"] if caps["zoom"]["max"] is not None else 25.0,
                )
        except Exception as e:
            logger.warning(
                f"Failed to get capabilities for {self.camera_name}, using defaults: {e}"
            )

        # Return default capabilities if query failed
        return self._capabilities

    def get_device_info(self) -> DeviceInfo | None:
        """Get device hardware information from Hikvision camera.

        Queries /ISAPI/System/deviceInfo endpoint.
        """
        try:
            info = self._ptz.get_device_info()
            if info:
                return DeviceInfo(
                    model=info.get("model"),
                    serial_number=info.get("serial_number"),
                    mac_address=info.get("mac_address"),
                    firmware_version=info.get("firmware_version"),
                    device_name=info.get("device_name"),
                    device_type=info.get("device_type"),
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get device info for {self.camera_name}: {e}")
            return None

    def wait_for_stabilization(self, max_wait: float = 5.0) -> PTZPosition | None:
        """Wait for Hikvision camera to stabilize.

        Polls camera status until position stops changing for STABLE_THRESHOLD seconds.
        """
        start_time = time.time()
        last_position: PTZPosition | None = None
        stable_since: float | None = None

        while time.time() - start_time < max_wait:
            current = self.get_status()
            if current is None:
                time.sleep(self.POLL_INTERVAL)
                continue

            if last_position is not None:
                # Check if position has changed
                position_changed = False

                if current.pan is not None and last_position.pan is not None:
                    if abs(current.pan - last_position.pan) > self.POSITION_TOLERANCE:
                        position_changed = True

                if current.tilt is not None and last_position.tilt is not None:
                    if abs(current.tilt - last_position.tilt) > self.POSITION_TOLERANCE:
                        position_changed = True

                if current.zoom is not None and last_position.zoom is not None:
                    if abs(current.zoom - last_position.zoom) > self.POSITION_TOLERANCE:
                        position_changed = True

                if not position_changed:
                    if stable_since is None:
                        stable_since = time.time()
                    elif time.time() - stable_since >= self.STABLE_THRESHOLD:
                        # Position has been stable long enough
                        return current
                else:
                    stable_since = None

            last_position = current
            time.sleep(self.POLL_INTERVAL)

        # Return last known position even if not fully stabilized
        return self.get_status()


def create_ptz_camera(
    camera_ip: str,
    camera_name: str,
    camera_model: str | None = None,
    tenant_id: str | None = None,
) -> BasePTZCamera:
    """Factory function to create appropriate PTZ camera instance.

    Args:
        camera_ip: IP address of the camera.
        camera_name: Display name for logging.
        camera_model: Camera model string (used to detect brand).
        tenant_id: Tenant ID to look up credentials. If None, uses global credentials.

    Returns:
        BasePTZCamera instance appropriate for the camera brand.

    Currently only supports Hikvision cameras. Future enhancement:
    detect brand from model string and return appropriate implementation.
    """
    # Get tenant-specific credentials if tenant_id provided
    username, password = None, None
    if tenant_id:
        username, password = get_tenant_credentials(tenant_id)

    # For now, default to Hikvision
    # Future: detect brand from camera_model and return appropriate implementation
    return HikvisionPTZCamera(
        camera_ip=camera_ip,
        camera_name=camera_name,
        username=username,
        password=password,
    )
