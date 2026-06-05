"""Legacy ``HikvisionCameraController`` shim over :class:`HikvisionISAPIClient`.

This module preserves the historical constructor signature and export name
``HikvisionCameraController`` for existing :class:`CameraController` consumers.
All ISAPI logic (paths, namespace, unit math, XML building) now lives in the
:mod:`poc_homography.infrastructure.clients.hikvision` package; this class only
adapts the legacy ``__init__(camera_config, timeout)`` signature onto the new
adapter's standalone constructor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.infrastructure.clients.hikvision.isapi_client import (
    HikvisionISAPIClient,
)

if TYPE_CHECKING:
    from poc_homography.domain.entities.camera_config import CameraConfig


class HikvisionCameraController(HikvisionISAPIClient):
    """Hikvision PTZ camera controller (legacy-signature shim).

    Subclasses :class:`HikvisionISAPIClient`, so it satisfies both the
    :class:`CameraController` protocol and the broader ``CameraDevice`` protocol.
    The only behavior added here is the legacy constructor that builds the
    transport from a :class:`CameraConfig` entity.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        timeout: float = 5.0,
    ) -> None:
        """Initialize the controller from a camera configuration.

        Args:
            camera_config: Camera configuration entity with IP and credentials.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If ``camera_config`` has no IP address.
        """
        if not camera_config.ip_address:
            raise ValueError(f"Camera '{camera_config.name}' has no IP address")
        super().__init__(
            camera_config.ip_address,
            camera_config.credential.username,
            camera_config.credential.password,
            timeout=timeout,
        )
