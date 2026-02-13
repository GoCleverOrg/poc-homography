"""CalibrationLineTraceSet entity — a set of N-point line traces for a camera frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from poc_homography.domain.vo import LineTrace, PTZState


@dataclass(frozen=True)
class CalibrationLineTraceSet:
    """A set of N-point line traces from a single camera frame.

    Attributes:
        name: Identifier (also serves as entity id).
        image: Image filename.
        camera_pose: PTZ state at capture time.
        line_traces: N-point line traces.
    """

    name: str
    image: str
    camera_pose: PTZState
    line_traces: tuple[LineTrace, ...]

    @property
    def id(self) -> str:
        """Entity identifier."""
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "image": self.image,
            "camera_pose": self.camera_pose.to_dict(),
            "line_traces": [lt.to_dict() for lt in self.line_traces],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationLineTraceSet:
        """Create CalibrationLineTraceSet from dictionary."""
        from poc_homography.domain.vo import LineTrace, PTZState

        return cls(
            name=data["name"],
            image=data["image"],
            camera_pose=PTZState.from_dict(data["camera_pose"]),
            line_traces=tuple(LineTrace.from_dict(lt) for lt in data["line_traces"]),
        )
