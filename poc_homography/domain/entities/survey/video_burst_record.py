"""Video burst record for Phase 8 (static jitter) RTSP segments.

Preserves the original encoded RTSP segment while making each contained frame
addressable for offline processing. ``FrameRef`` is a lightweight pointer; the
full :class:`~poc_homography.domain.entities.survey.frame_record.FrameRecord`
for each frame is stored separately under the frame layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from poc_homography.domain.entities.survey.frame_record import CommandedState
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.types import FPS, Seconds


@dataclass(frozen=True)
class FrameRef:
    """A lightweight pointer to one frame within a video burst."""

    capture_id: str
    frame_index: int
    timestamp_at_capture: datetime
    image_path: Path

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "capture_id": self.capture_id,
            "frame_index": self.frame_index,
            "timestamp_at_capture": self.timestamp_at_capture.isoformat(),
            "image_path": str(self.image_path),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameRef:
        """Create :class:`FrameRef` from a dictionary."""
        return cls(
            capture_id=str(data["capture_id"]),
            frame_index=int(data["frame_index"]),
            timestamp_at_capture=datetime.fromisoformat(data["timestamp_at_capture"]),
            image_path=Path(data["image_path"]),
        )


@dataclass(frozen=True, eq=False)
class VideoBurstRecord:
    """An encoded RTSP segment plus addressable per-frame references.

    The ``id`` property is the ``burst_id``, satisfying the ``Entity``
    protocol. ``phase`` is typed generally but is always
    :attr:`SurveyPhase.STATIC_JITTER` in practice (Phase 8).
    """

    burst_id: str
    run_id: str
    camera_id: str
    phase: SurveyPhase
    segment_path: Path
    duration_s: Seconds
    fps: FPS
    codec: str
    commanded_state: CommandedState
    frame_refs: tuple[FrameRef, ...]

    @property
    def id(self) -> str:
        """Unique identifier — the burst id."""
        return self.burst_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def frame_by_index(self, frame_index: int) -> FrameRef | None:
        """Return the :class:`FrameRef` with ``frame_index``, or ``None``."""
        for ref in self.frame_refs:
            if ref.frame_index == frame_index:
                return ref
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "burst_id": self.burst_id,
            "run_id": self.run_id,
            "camera_id": self.camera_id,
            "phase": self.phase.value,
            "segment_path": str(self.segment_path),
            "duration_s": float(self.duration_s),
            "fps": float(self.fps),
            "codec": self.codec,
            "commanded_state": self.commanded_state.to_dict(),
            "frame_refs": [ref.to_dict() for ref in self.frame_refs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoBurstRecord:
        """Create :class:`VideoBurstRecord` from a dictionary."""
        return cls(
            burst_id=str(data["burst_id"]),
            run_id=str(data["run_id"]),
            camera_id=str(data["camera_id"]),
            phase=SurveyPhase(data["phase"]),
            segment_path=Path(data["segment_path"]),
            duration_s=Seconds(float(data["duration_s"])),
            fps=FPS(float(data["fps"])),
            codec=str(data["codec"]),
            commanded_state=CommandedState.from_dict(data["commanded_state"]),
            frame_refs=tuple(FrameRef.from_dict(ref) for ref in data["frame_refs"]),
        )
