"""Camera snapshot value object for point-in-time camera state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_homography.domain.entities.camera_calibration import CameraCalibration
    from poc_homography.domain.entities.camera_config import CameraConfig
    from poc_homography.domain.vo.ptz_state import PTZState


@dataclass(frozen=True)
class CameraSnapshot:
    """Immutable point-in-time combination of all camera data.

    This value object combines the three separate pieces of camera data
    (config, calibration, ptz_state) into a single immutable snapshot.
    It is useful for passing to domain services that need all three pieces.

    Each piece has a different lifecycle:
    - config: Set once during registration, rarely changes
    - calibration: Refined during calibration workflows
    - ptz_state: Transient, from hardware API, changes constantly

    Attributes:
        config: Camera configuration (registration data).
        calibration: Camera calibration (refined during calibration).
        ptz_state: Current PTZ state (transient, from hardware).
    """

    config: CameraConfig
    calibration: CameraCalibration
    ptz_state: PTZState
