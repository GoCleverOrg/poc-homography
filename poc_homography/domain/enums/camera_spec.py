"""Camera specification enum for known camera models."""

from enum import Enum

from poc_homography.domain.enums.tilt_convention import TiltConvention
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.types import Millimeters, Pixels, Unitless


class CameraSpec(Enum):
    """Predefined specifications for known camera models.

    Each camera model has fixed hardware characteristics that determine
    its intrinsic parameters, distortion, and PTZ behavior. This enum
    provides type-safe access to these specifications.

    Usage:
        spec = CameraSpec.HIKVISION_DS_2DF8425IX
        print(spec.sensor_width)  # Millimeters(6.78)
        print(spec.distortion.k1)  # Unitless(-0.341052)
    """

    # Hikvision DS-2DF8425IX-AELW series (4MP PTZ, 25x optical zoom)
    # Source: Hikvision official datasheet + calibration
    # - Sensor: 1/1.8" progressive scan CMOS
    # - Resolution: 2560x1440 (4MP, 16:9)
    # - Focal Length: 5.9mm (wide) to 147.5mm (tele)
    # - Optical Zoom: 25x
    # - Horizontal FOV: 59.8° (wide) to 3.3° (tele)
    # - Distortion: Calibrated 2025-12-11
    HIKVISION_DS_2DF8425IX = (
        "DS-2DF8425IX-AELW",  # model_name
        Millimeters(6.78),  # sensor_width (calculated from 59.8° FOV at 5.9mm)
        Millimeters(5.9),  # base_focal_length
        Pixels(2560),  # image_width
        Pixels(1440),  # image_height
        TiltConvention.POSITIVE_DOWN,  # tilt_convention
        25.0,  # max_zoom
        LensDistortion(  # distortion
            k1=Unitless(-0.341052),
            k2=Unitless(0.787571),
            p1=Unitless(0.0),
            p2=Unitless(0.0),
        ),
    )

    def __init__(
        self,
        model_name: str,
        sensor_width: Millimeters,
        base_focal_length: Millimeters,
        image_width: Pixels,
        image_height: Pixels,
        tilt_convention: TiltConvention,
        max_zoom: float,
        distortion: LensDistortion,
    ) -> None:
        self._model_name = model_name
        self._sensor_width = sensor_width
        self._base_focal_length = base_focal_length
        self._image_width = image_width
        self._image_height = image_height
        self._tilt_convention = tilt_convention
        self._max_zoom = max_zoom
        self._distortion = distortion

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
    def image_width(self) -> Pixels:
        """Image width in pixels."""
        return self._image_width

    @property
    def image_height(self) -> Pixels:
        """Image height in pixels."""
        return self._image_height

    @property
    def tilt_convention(self) -> TiltConvention:
        """Tilt angle sign convention for this camera model."""
        return self._tilt_convention

    @property
    def max_zoom(self) -> float:
        """Maximum optical zoom factor."""
        return self._max_zoom

    @property
    def distortion(self) -> LensDistortion:
        """Lens distortion coefficients."""
        return self._distortion

    def focal_length_at_zoom(self, zoom: float) -> Millimeters:
        """Calculate focal length at a given zoom level.

        Args:
            zoom: Zoom factor (1.0 = no zoom, max_zoom = full zoom)

        Returns:
            Focal length in millimeters at the specified zoom level.
        """
        clamped_zoom = max(1.0, min(zoom, self._max_zoom))
        return Millimeters(self._base_focal_length * clamped_zoom)
