"""Pose value objects for the multi-phase survey planner.

A :class:`Pose` is a single PTZ target the planner emits. Poses are frozen and
serialise via ``to_dict`` / ``from_dict`` following the project pattern in
``poc_homography/domain/vo/ptz_state.py`` (enums via ``.value``, floats via
``float(...)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from poc_homography.types import Degrees, Unitless


def canonical_pose_key(pan: float, tilt: float, zoom: float) -> str:
    """Return a stable, deterministic pose id for a physical PTZ pose.

    Pan and tilt are quantised to 0.1 degrees and zoom to 0.01 before
    formatting, so two physically-equal poses yield the SAME id on every run
    regardless of insertion order, randomness, or time. The result is a
    human-readable fixed-width string, e.g. ``"p+0120.1_t-0015.0_z004.00"``.

    Args:
        pan: Pan angle in degrees.
        tilt: Tilt angle in degrees.
        zoom: Zoom factor (dimensionless).

    Returns:
        A stable string id derived purely from the quantised geometry.
    """
    pan_q = round(float(pan), 1)
    tilt_q = round(float(tilt), 1)
    zoom_q = round(float(zoom), 2)
    # ``+ 0.0`` normalises a possible ``-0.0`` to ``0.0`` for a stable sign.
    return f"p{pan_q + 0.0:+07.1f}_t{tilt_q + 0.0:+07.1f}_z{zoom_q + 0.0:06.2f}"


class ApproachDirection(str, Enum):
    """Direction from which a pose is approached along a sweep axis.

    Members subclass :class:`str` so their ``.value`` is a stable, plain
    string safe for YAML/JSON serialisation.

    Members:
        ASCENDING: Approached from lower toward higher axis values.
        DESCENDING: Approached from higher toward lower axis values.
    """

    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass(frozen=True)
class Pose:
    """A single PTZ target pose emitted by the planner.

    Attributes:
        pan: Pan angle in degrees.
        tilt: Tilt angle in degrees.
        zoom: Zoom factor (dimensionless; 1.0 = wide).
        approach_direction: Direction the pose is approached from, if tagged.
        is_holdout: Whether the pose belongs to a held-out validation set.
        region_id: Stable per-region identifier for cross-zoom grouping.
    """

    pan: Degrees
    tilt: Degrees
    zoom: Unitless
    approach_direction: ApproachDirection | None = None
    is_holdout: bool = False
    region_id: str | None = None
    pose_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pan": float(self.pan),
            "tilt": float(self.tilt),
            "zoom": float(self.zoom),
            "approach_direction": (
                self.approach_direction.value if self.approach_direction is not None else None
            ),
            "is_holdout": self.is_holdout,
            "region_id": self.region_id,
            "pose_id": self.pose_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pose:
        """Create :class:`Pose` from a dictionary."""
        approach = data.get("approach_direction")
        region_id = data.get("region_id")
        pose_id = data.get("pose_id")
        return cls(
            pan=Degrees(float(data["pan"])),
            tilt=Degrees(float(data["tilt"])),
            zoom=Unitless(float(data["zoom"])),
            approach_direction=(ApproachDirection(approach) if approach is not None else None),
            is_holdout=bool(data["is_holdout"]),
            region_id=str(region_id) if region_id is not None else None,
            pose_id=str(pose_id) if pose_id is not None else None,
        )
