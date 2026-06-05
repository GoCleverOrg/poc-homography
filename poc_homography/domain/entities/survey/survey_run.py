"""SurveyRun aggregate entity for the multi-phase, multi-camera workflow.

``SurveyRun`` is a new entity that extends the thin ``SurveySession`` concept
without altering it; there is no migration. The ``status`` field enables a
planner to resume interrupted runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from poc_homography.domain.entities.survey import (
    SURVEY_SCHEMA_VERSION,
    check_schema_version,
)
from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus


@dataclass(frozen=True, eq=False)
class SurveyRun:
    """A single survey run for one camera across a set of phases.

    The ``id`` property is the ``run_id``, satisfying the ``Entity`` protocol.
    """

    run_id: str
    camera_id: str
    phases: frozenset[SurveyPhase]
    started_at: datetime
    finished_at: datetime | None = None
    status: SurveyRunStatus = SurveyRunStatus.PENDING
    schema_version: str = field(default=SURVEY_SCHEMA_VERSION)

    @property
    def id(self) -> str:
        """Unique identifier — the run id."""
        return self.run_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Phases are serialised as a sorted list of stable string values for
        deterministic output.
        """
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "camera_id": self.camera_id,
            "phases": sorted(phase.value for phase in self.phases),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurveyRun:
        """Create :class:`SurveyRun` from a dictionary.

        Raises:
            ValueError: If ``schema_version`` is unrecognised.
        """
        version = check_schema_version(str(data["schema_version"]))
        finished_raw = data.get("finished_at")
        return cls(
            run_id=str(data["run_id"]),
            camera_id=str(data["camera_id"]),
            phases=frozenset(SurveyPhase(value) for value in data["phases"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=datetime.fromisoformat(finished_raw) if finished_raw else None,
            status=SurveyRunStatus(data["status"]),
            schema_version=version,
        )
