"""Camera specification enum for known camera models."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

from poc_homography.domain.enums.tilt_convention import TiltConvention
from poc_homography.domain.vo.image_dimensions import ImageDimensions
from poc_homography.types import Millimeters, Pixels

if TYPE_CHECKING:
    from poc_homography.domain.vo.credential import Credential


def hikvision_rtsp_url(ip: str, credential: Credential, stream_type: str = "main") -> str:
    """Build RTSP URL for Hikvision camera stream.

    Args:
        ip: Camera IP address
        credential: Camera credentials
        stream_type: "main" (101) for high quality or "sub" (102) for low quality

    Returns:
        Full RTSP URL for the camera stream
    """
    channel = "101" if stream_type == "main" else "102"
    return (
        f"rtsp://{credential.username}:{credential.password}@{ip}:554/Streaming/Channels/{channel}"
    )


# Type alias for RTSP URL builder function
RtspUrlBuilder = Callable[["str", "Credential", "str"], str]


class CameraSpec(Enum):
    """Predefined specifications for known camera models.

    Each camera model has fixed hardware characteristics that determine
    its intrinsic parameters and PTZ behavior. This enum provides type-safe
    access to these specifications.

    Note: Distortion coefficients are NOT included here because they are
    calibrated per-camera (not per-model) and stored in CameraCalibration.

    Usage:
        spec = CameraSpec.HIKVISION_DS_2DF8425IX
        print(spec.sensor_width)  # Millimeters(6.78)
    """

    # Hikvision DS-2DF8425IX-AELW series (4MP PTZ, 25x optical zoom)
    # Source: Hikvision official datasheet
    # - Sensor: 1/1.8" progressive scan CMOS
    # - Resolution: 2560x1440 (4MP, 16:9)
    # - Focal Length: 5.9mm (wide) to 147.5mm (tele)
    # - Optical Zoom: 25x
    # - Horizontal FOV: 59.8° (wide) to 3.3° (tele)
    HIKVISION_DS_2DF8425IX = (
        "DS-2DF8425IX-AELW",  # model_name
        Millimeters(6.78),  # sensor_width (calculated from 59.8° FOV at 5.9mm)
        Millimeters(5.9),  # base_focal_length
        ImageDimensions.create(width=2560, height=1440),  # dimensions
        TiltConvention.POSITIVE_DOWN,  # tilt_convention
        25.0,  # max_zoom
        hikvision_rtsp_url,  # rtsp_url_builder
    )

    def __init__(
        self,
        model_name: str,
        sensor_width: Millimeters,
        base_focal_length: Millimeters,
        dimensions: ImageDimensions,
        tilt_convention: TiltConvention,
        max_zoom: float,
        rtsp_url_builder: RtspUrlBuilder,
    ) -> None:
        self._model_name = model_name
        self._sensor_width = sensor_width
        self._base_focal_length = base_focal_length
        self._dimensions = dimensions
        self._tilt_convention = tilt_convention
        self._max_zoom = max_zoom
        self._rtsp_url_builder = rtsp_url_builder

    @property
    def model_name(self) -> str:
        """Camera model name/identifier."""
        return self._model_name

    @property
    def sensor_width(self) -> Millimeters:
        """Sensor width in millimeters."""
        return self._sensor_width

    @property
    def base_focal_length(self) -> Millimeters:
        """Base focal length in millimeters (at 1x zoom)."""
        return self._base_focal_length

    @property
    def dimensions(self) -> ImageDimensions:
        """Image dimensions (width and height in pixels)."""
        return self._dimensions

    @property
    def image_width(self) -> Pixels:
        """Image width in pixels (backward-compatible property)."""
        return self._dimensions.width

    @property
    def image_height(self) -> Pixels:
        """Image height in pixels (backward-compatible property)."""
        return self._dimensions.height

    @property
    def tilt_convention(self) -> TiltConvention:
        """Tilt angle sign convention for this camera model."""
        return self._tilt_convention

    @property
    def max_zoom(self) -> float:
        """Maximum optical zoom factor."""
        return self._max_zoom

    def rtsp_url(self, ip: str, credential: Credential, stream_type: str = "main") -> str:
        """Build RTSP URL for this camera model.

        Args:
            ip: Camera IP address
            credential: Camera credentials
            stream_type: "main" for high quality or "sub" for low quality

        Returns:
            Full RTSP URL for the camera stream
        """
        return self._rtsp_url_builder(ip, credential, stream_type)

    def focal_length_at_zoom(self, zoom: float) -> Millimeters:
        """Calculate focal length at a given zoom level.

        Args:
            zoom: Zoom factor (1.0 = no zoom, max_zoom = full zoom)

        Returns:
            Focal length in millimeters at the specified zoom level.
        """
        clamped_zoom = max(1.0, min(zoom, self._max_zoom))
        return Millimeters(self._base_focal_length * clamped_zoom)
