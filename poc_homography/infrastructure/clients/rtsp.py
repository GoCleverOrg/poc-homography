"""RTSP URL building utilities for camera streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.domain.enums.camera_spec import CameraSpec

if TYPE_CHECKING:
    from poc_homography.domain.entities.camera_config import CameraConfig
    from poc_homography.domain.vo.credential import Credential


def hikvision_rtsp_url(ip: str, credential: Credential, stream_type: str = "main") -> str:
    """Build RTSP URL for Hikvision camera stream.

    Args:
        ip: Camera IP address.
        credential: Camera credentials.
        stream_type: ``"main"`` (channel 101) or ``"sub"`` (channel 102).

    Returns:
        Full RTSP URL for the camera stream.
    """
    channel = "101" if stream_type == "main" else "102"
    return (
        f"rtsp://{credential.username}:{credential.password}@{ip}:554/Streaming/Channels/{channel}"
    )


def build_rtsp_url(config: CameraConfig, stream_type: str = "main") -> str:
    """Build RTSP URL for a camera configuration.

    Dispatches to the appropriate vendor-specific RTSP URL builder
    based on the camera spec.

    Args:
        config: Camera configuration entity.
        stream_type: ``"main"`` for high quality or ``"sub"`` for low quality.

    Returns:
        Full RTSP URL for the camera stream.

    Raises:
        ValueError: If camera has no IP address configured.
        NotImplementedError: If camera spec has no known RTSP URL builder.
    """
    if not config.ip_address:
        raise ValueError(f"Camera '{config.name}' has no IP address configured")

    if config.spec == CameraSpec.HIKVISION_DS_2DF8425IX:
        return hikvision_rtsp_url(config.ip_address, config.credential, stream_type)

    raise NotImplementedError(f"No RTSP URL builder for camera spec: {config.spec}")
