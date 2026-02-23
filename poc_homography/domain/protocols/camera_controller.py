"""Protocol for PTZ camera controller."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from poc_homography.domain.vo.ptz_state import PTZState


class CameraControllerError(Exception):
    """Error from camera controller operations."""


class CameraController(Protocol):
    """Protocol for PTZ camera controllers.

    This protocol defines the interface for controlling PTZ cameras.
    Infrastructure implementations provide the actual hardware communication.

    The controller maintains in-memory state of the last known PTZ position.
    """

    @property
    def last_ptz_state(self) -> PTZState | None:
        """Get the last known PTZ state (in-memory cache).

        Returns:
            Last PTZ state if available, None if never fetched.
        """
        ...

    def get_ptz_status(self) -> PTZState:
        """Read current PTZ status from camera hardware.

        Updates the internal last_ptz_state cache.

        Returns:
            Current PTZ state with pan, tilt, and zoom values.

        Raises:
            CameraControllerError: If communication with camera fails.
        """
        ...

    def move_absolute(
        self,
        pan: float | None = None,
        tilt: float | None = None,
        zoom: float | None = None,
    ) -> PTZState:
        """Move camera to absolute PTZ position.

        Args:
            pan: Target pan angle in degrees (None = keep current).
            tilt: Target tilt angle in degrees (None = keep current).
            zoom: Target zoom level (None = keep current).

        Returns:
            PTZ state after move completes.

        Raises:
            CameraControllerError: If move command fails.
        """
        ...

    def move_relative(
        self,
        pan_delta: float = 0.0,
        tilt_delta: float = 0.0,
        zoom_delta: float = 0.0,
    ) -> PTZState:
        """Move camera relative to current position.

        Args:
            pan_delta: Pan offset in degrees.
            tilt_delta: Tilt offset in degrees.
            zoom_delta: Zoom offset.

        Returns:
            PTZ state after move completes.

        Raises:
            CameraControllerError: If move command fails.
        """
        ...
