"""Tilt convention enum for camera orientation."""

from enum import Enum


class TiltConvention(Enum):
    """Convention for tilt angle sign.

    Different PTZ camera manufacturers use different conventions for tilt angles.
    This enum allows per-camera configuration of the tilt sign convention.

    Attributes:
        POSITIVE_UP: Positive tilt = camera looks up (common convention)
        POSITIVE_DOWN: Positive tilt = camera looks down (Hikvision convention)
    """

    POSITIVE_UP = "positive_up"
    POSITIVE_DOWN = "positive_down"

    @property
    def sign(self) -> int:
        """Return the sign multiplier for tilt conversion.

        Returns:
            1 for POSITIVE_UP, -1 for POSITIVE_DOWN
        """
        return 1 if self == TiltConvention.POSITIVE_UP else -1
