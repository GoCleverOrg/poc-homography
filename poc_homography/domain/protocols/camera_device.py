"""Protocol for a full-featured PTZ camera device.

``CameraDevice`` is a structural superset of :class:`CameraController`: it keeps
the same method names and parameter names for the overlapping PTZ-read/PTZ-write
operations (so any ``CameraDevice`` also satisfies ``CameraController``) while
adding identity/health, optics, presets, and capture operations.

Infrastructure implementations (e.g. the Hikvision ISAPI adapter) provide the
concrete hardware communication. Value-object types are imported only under
``TYPE_CHECKING`` to avoid runtime import cost and import cycles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from poc_homography.domain.vo.camera_capabilities import CameraCapabilities
    from poc_homography.domain.vo.camera_preset import CameraPreset
    from poc_homography.domain.vo.device_health import DeviceHealth
    from poc_homography.domain.vo.device_info import DeviceInfo
    from poc_homography.domain.vo.image_optics import ImageOptics
    from poc_homography.domain.vo.ptz_state import PTZState
    from poc_homography.domain.vo.stream_profile import StreamProfile


class CameraDevice(Protocol):
    """Protocol for a full-featured PTZ camera device.

    Extends the PTZ-control surface of :class:`CameraController` with identity,
    health, optics, preset, and capture operations. The device maintains an
    in-memory cache of the last known PTZ position.

    All methods raise :class:`CameraControllerError` (or a subclass) on
    communication failure.
    """

    # --- Identity / health -------------------------------------------------

    def get_device_info(self) -> DeviceInfo:
        """Read device identity and firmware metadata.

        Returns:
            Device identity and firmware information.

        Raises:
            CameraControllerError: If communication with the camera fails.
        """
        ...

    def get_capabilities(self) -> CameraCapabilities:
        """Read the camera's PTZ position and speed limits (degrees-based).

        Returns:
            The real hardware capability envelope.

        Raises:
            CameraControllerError: If communication with the camera fails.
        """
        ...

    def get_health(self) -> DeviceHealth:
        """Read runtime health metrics and lens odometry.

        Returns:
            Device health, including lens and dome wear counters.

        Raises:
            CameraControllerError: If communication with the camera fails.
        """
        ...

    def get_stream_profiles(self) -> list[StreamProfile]:
        """Read the available video streaming profiles.

        Returns:
            One profile per readable streaming channel (at least the primary).

        Raises:
            CameraControllerError: If communication with the camera fails.
        """
        ...

    # --- PTZ read ----------------------------------------------------------

    @property
    def last_ptz_state(self) -> PTZState | None:
        """Get the last known PTZ state (in-memory cache).

        Returns:
            Last PTZ state if available, ``None`` if never fetched.
        """
        ...

    def get_ptz_status(self) -> PTZState:
        """Read current PTZ status from camera hardware.

        Updates the internal ``last_ptz_state`` cache.

        Returns:
            Current PTZ state with pan, tilt, and zoom values.

        Raises:
            CameraControllerError: If communication with the camera fails.
        """
        ...

    def wait_for_stabilization(
        self,
        timeout_s: float = 5.0,
        threshold: float = 0.1,
    ) -> PTZState:
        """Poll PTZ status until the position stops changing.

        The position is considered stable once consecutive readings differ by
        less than ``threshold`` for a short stable window.

        Args:
            timeout_s: Maximum time to wait in seconds.
            threshold: Per-axis change tolerance (degrees / zoom factor).

        Returns:
            The final PTZ state, or the last reading at timeout.

        Raises:
            CameraControllerError: If communication with the camera fails.
        """
        ...

    # --- PTZ write ---------------------------------------------------------

    def move_absolute(
        self,
        pan: float | None = None,
        tilt: float | None = None,
        zoom: float | None = None,
    ) -> PTZState:
        """Move camera to an absolute PTZ position.

        Args:
            pan: Target pan angle in degrees (``None`` = keep current).
            tilt: Target tilt angle in degrees (``None`` = keep current).
            zoom: Target zoom level (``None`` = keep current).

        Returns:
            PTZ state after the move completes.

        Raises:
            CameraControllerError: If the move command fails.
        """
        ...

    def move_relative(
        self,
        pan_delta: float = 0.0,
        tilt_delta: float = 0.0,
        zoom_delta: float = 0.0,
    ) -> PTZState:
        """Move camera relative to its current position.

        Args:
            pan_delta: Pan offset in degrees.
            tilt_delta: Tilt offset in degrees.
            zoom_delta: Zoom offset.

        Returns:
            PTZ state after the move completes.

        Raises:
            CameraControllerError: If the move command fails.
        """
        ...

    def move_continuous(
        self,
        pan_speed: int = 0,
        tilt_speed: int = 0,
        zoom_speed: int = 0,
    ) -> None:
        """Start continuous PTZ movement at the given speeds.

        Args:
            pan_speed: Pan speed in the range -100..100 (0 = stop).
            tilt_speed: Tilt speed in the range -100..100 (0 = stop).
            zoom_speed: Zoom speed in the range -100..100 (0 = stop).

        Raises:
            CameraControllerError: If the command fails.
        """
        ...

    def stop(self) -> None:
        """Stop all PTZ movement.

        Raises:
            CameraControllerError: If the command fails.
        """
        ...

    def goto_preset(self, preset_id: int) -> None:
        """Move the camera to a stored preset.

        Args:
            preset_id: Identifier of the preset to recall.

        Raises:
            CameraControllerError: If the command fails.
        """
        ...

    def position3d(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> None:
        """Issue a 3D drag-zoom positioning command.

        Args:
            start_x: Drag-box start X coordinate.
            start_y: Drag-box start Y coordinate.
            end_x: Drag-box end X coordinate.
            end_y: Drag-box end Y coordinate.

        Raises:
            CameraControllerError: If the command fails.
        """
        ...

    # --- Optics read / write ----------------------------------------------

    def get_optics(self) -> ImageOptics:
        """Read the current focus, iris, exposure, and white-balance state.

        Returns:
            Aggregated optics state.

        Raises:
            CameraControllerError: If communication with the camera fails.
        """
        ...

    def set_focus(self, focus: int) -> None:
        """Set the focus position.

        Args:
            focus: Target focus position in raw lens steps.

        Raises:
            CameraControllerError: If the command fails or is unsupported.
        """
        ...

    def set_iris(self, level: int) -> None:
        """Set the iris level.

        Args:
            level: Target iris level.

        Raises:
            CameraControllerError: If the command fails or is unsupported.
        """
        ...

    def set_exposure(self, exposure_type: str) -> None:
        """Set the exposure mode.

        Args:
            exposure_type: Target exposure type (e.g. ``"auto"``).

        Raises:
            CameraControllerError: If the command fails or is unsupported.
        """
        ...

    # --- Presets -----------------------------------------------------------

    def list_presets(self) -> list[CameraPreset]:
        """List the stored PTZ presets.

        Returns:
            All configured presets with their target PTZ states.

        Raises:
            CameraControllerError: If communication with the camera fails.
        """
        ...

    # --- Capture -----------------------------------------------------------

    def capture_snapshot(self) -> bytes:
        """Capture a still image from the primary channel.

        Returns:
            JPEG image bytes.

        Raises:
            CameraControllerError: If communication with the camera fails.
        """
        ...
