"""Phase plan descriptor for the multi-phase survey planner.

A :class:`PhasePlan` binds one :class:`SurveyPhase` identity to its ordered
poses plus the parameters that produced them. It is the planner-side
"abstraction descriptor" for a phase (the issue's "SurveyPhase abstraction",
renamed to avoid colliding with the existing ``SurveyPhase`` enum).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.survey.planner.poses import ApproachDirection, Pose


@dataclass(frozen=True)
class PhasePlan:
    """An ordered set of poses for a single survey phase.

    Attributes:
        phase: The phase identity (existing :class:`SurveyPhase` enum).
        poses: The ordered poses to capture for this phase.
        zoom_levels: Zoom levels relevant to this phase, if any.
        repeat_count: How many times the phase should be repeated.
        approach_directions: Approach directions exercised by this phase.
        is_holdout: Whether this whole phase plan is a held-out set.
    """

    phase: SurveyPhase
    poses: tuple[Pose, ...]
    zoom_levels: tuple[float, ...] = ()
    repeat_count: int = 1
    approach_directions: tuple[ApproachDirection, ...] = ()
    is_holdout: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase.value,
            "poses": [pose.to_dict() for pose in self.poses],
            "zoom_levels": [float(z) for z in self.zoom_levels],
            "repeat_count": self.repeat_count,
            "approach_directions": [d.value for d in self.approach_directions],
            "is_holdout": self.is_holdout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhasePlan:
        """Create :class:`PhasePlan` from a dictionary."""
        return cls(
            phase=SurveyPhase(data["phase"]),
            poses=tuple(Pose.from_dict(p) for p in data["poses"]),
            zoom_levels=tuple(float(z) for z in data.get("zoom_levels", [])),
            repeat_count=int(data.get("repeat_count", 1)),
            approach_directions=tuple(
                ApproachDirection(d) for d in data.get("approach_directions", [])
            ),
            is_holdout=bool(data.get("is_holdout", False)),
        )
