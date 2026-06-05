"""Planner run aggregate with resume cursor and persistence bridge.

``PlannedSurveyRun`` is the planner-side run aggregate: an ordered tuple of
:class:`PhasePlan` plus a :class:`ResumeCursor` for crash-safe resume. It
produces a C1 :class:`SurveyRun` "header" via :meth:`PlannedSurveyRun.header`
for persistence compatibility with the existing thin entity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from poc_homography.domain.entities.survey import (
    SURVEY_SCHEMA_VERSION,
    check_schema_version,
)
from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus
from poc_homography.survey.planner.phase_plan import PhasePlan

if TYPE_CHECKING:
    from collections.abc import Iterator

    from poc_homography.survey.planner.poses import Pose


@dataclass(frozen=True)
class ResumeCursor:
    """Checkpoint marking the last fully-completed (phase, pose).

    A value of ``-1`` means nothing has completed yet.

    Attributes:
        last_completed_phase_index: Index of the phase whose ``pose`` was last
            completed.
        last_completed_pose_index: Index of the last completed pose within that
            phase.
    """

    last_completed_phase_index: int = -1
    last_completed_pose_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "last_completed_phase_index": self.last_completed_phase_index,
            "last_completed_pose_index": self.last_completed_pose_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResumeCursor:
        """Create :class:`ResumeCursor` from a dictionary."""
        return cls(
            last_completed_phase_index=int(data["last_completed_phase_index"]),
            last_completed_pose_index=int(data["last_completed_pose_index"]),
        )


@dataclass(frozen=True, eq=False)
class PlannedSurveyRun:
    """Ordered multi-phase plan for one camera with resume support.

    The ``id`` property is the ``run_id``, satisfying the ``Entity`` protocol.
    Equality/hash are by id, mirroring :class:`SurveyRun`.

    Attributes:
        run_id: Unique run identifier.
        camera_id: The camera this run targets.
        phases: Ordered phase plans.
        started_at: Run start timestamp.
        finished_at: Run finish timestamp, if finished.
        status: Lifecycle status.
        cursor: Resume checkpoint.
        schema_version: Survey schema version, stamped and validated on load.
    """

    run_id: str
    camera_id: str
    phases: tuple[PhasePlan, ...]
    started_at: datetime
    finished_at: datetime | None = None
    status: SurveyRunStatus = SurveyRunStatus.PENDING
    cursor: ResumeCursor = field(default_factory=ResumeCursor)
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

    def remaining_poses(self) -> Iterator[tuple[int, int, Pose]]:
        """Yield ``(phase_index, pose_index, pose)`` not yet completed.

        Skips phases before the cursor's phase entirely; within the cursor's
        phase, skips poses at or before the cursor's pose; later phases yield
        all their poses. When the cursor is ``-1/-1`` nothing is skipped.

        Yields:
            ``(phase_index, pose_index, pose)`` for each pending pose, in
            execution order.
        """
        phase_cursor = self.cursor.last_completed_phase_index
        pose_cursor = self.cursor.last_completed_pose_index
        for phase_index, plan in enumerate(self.phases):
            if phase_index < phase_cursor:
                continue
            for pose_index, pose in enumerate(plan.poses):
                if phase_index == phase_cursor and pose_index <= pose_cursor:
                    continue
                yield phase_index, pose_index, pose

    def advance(self, phase_index: int, pose_index: int) -> PlannedSurveyRun:
        """Return a copy whose cursor advances to ``(phase_index, pose_index)``."""
        return replace(
            self,
            cursor=ResumeCursor(
                last_completed_phase_index=phase_index,
                last_completed_pose_index=pose_index,
            ),
        )

    def with_status(self, status: SurveyRunStatus) -> PlannedSurveyRun:
        """Return a copy with ``status`` replaced."""
        return replace(self, status=status)

    def header(self) -> SurveyRun:
        """Build the C1 :class:`SurveyRun` persistence header for this run."""
        return SurveyRun(
            run_id=self.run_id,
            camera_id=self.camera_id,
            phases=frozenset(plan.phase for plan in self.phases),
            started_at=self.started_at,
            finished_at=self.finished_at,
            status=self.status,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (manifest checkpoint)."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "camera_id": self.camera_id,
            "phases": [plan.to_dict() for plan in self.phases],
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "status": self.status.value,
            "cursor": self.cursor.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannedSurveyRun:
        """Create :class:`PlannedSurveyRun` from a dictionary.

        Raises:
            ValueError: If ``schema_version`` is unrecognised.
        """
        version = check_schema_version(str(data["schema_version"]))
        finished_raw = data.get("finished_at")
        return cls(
            run_id=str(data["run_id"]),
            camera_id=str(data["camera_id"]),
            phases=tuple(PhasePlan.from_dict(p) for p in data["phases"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=datetime.fromisoformat(finished_raw) if finished_raw else None,
            status=SurveyRunStatus(data["status"]),
            cursor=ResumeCursor.from_dict(data["cursor"]),
            schema_version=version,
        )
