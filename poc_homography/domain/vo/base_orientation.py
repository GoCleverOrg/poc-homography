"""Base orientation value object for camera installation orientation."""

from dataclasses import dataclass

from poc_homography.types import Degrees


@dataclass(frozen=True)
class BaseOrientation:
    """Camera orientation at installation (when PTZ reports 0,0).

    This represents the fixed orientation of the camera when installed,
    before any PTZ movement is applied. The PTZ state is relative to this
    base orientation.

    Coordinate convention:
        - Yaw: Azimuth from map-north, clockwise positive (0=North, 90=East)
        - Pitch: Elevation angle, convention depends on TiltConvention
        - Roll: Rotation around optical axis, clockwise when looking forward

    Attributes:
        yaw: Base azimuth angle in degrees (pan offset from north).
        pitch: Base elevation angle in degrees (tilt offset from horizontal).
        roll: Base roll angle in degrees (rotation around optical axis).
    """

    yaw: Degrees
    pitch: Degrees
    roll: Degrees = Degrees(0.0)  # noqa: RUF009
