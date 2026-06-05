"""Protocol + DTOs for the concurrent multi-camera survey orchestrator (C3 boundary).

C5 (issue #262) ships only the operator-facing surface — CLI, Django views,
FastAPI endpoints, and persistence of the reproducible plan config. The real
concurrent orchestrator lands with C3. This module defines the boundary so the
operator surface can be built, tested (against a stub), and wired today.

The DTOs are deliberately plain, JSON-friendly value objects so the CLI and both
HTTP layers can serialise them identically without reaching into domain
entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig


@dataclass(frozen=True)
class RunHandle:
    """Identifiers returned when a multi-camera run starts.

    Attributes:
        run_id: The run-level identifier shared by every per-camera session.
        session_ids: Mapping of ``camera_id`` to its per-camera ``session_id``.
    """

    run_id: str
    session_ids: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraStatus:
    """Per-camera status snapshot within a run.

    Attributes:
        session_id: The per-camera session identifier.
        phase: Current ``SurveyPhase`` value, or a terminal marker.
        frame_count: Frames captured by this camera so far.
        status: Current ``SurveyRunStatus`` value for this camera.
    """

    session_id: str
    phase: str
    frame_count: int
    status: str


@dataclass(frozen=True)
class ProgressEvent:
    """A single streamed progress update for one camera entering/finishing a phase.

    Attributes:
        camera_id: The camera this event concerns.
        phase: The ``SurveyPhase`` value just reached or completed.
        frame_count: Cumulative frame count for the camera at this event.
        status: Current ``SurveyRunStatus`` value for the camera.
    """

    camera_id: str
    phase: str
    frame_count: int
    status: str


@dataclass(frozen=True)
class RunSummary:
    """One row in a run listing.

    Attributes:
        run_id: The run identifier.
        start_time: ISO-8601 start timestamp.
        camera_count: Number of cameras in the run.
        total_frame_count: Total frames captured across all cameras.
        status: Current ``SurveyRunStatus`` value for the run.
    """

    run_id: str
    start_time: str
    camera_count: int
    total_frame_count: int
    status: str


class ConcurrentSurveyOrchestrator(Protocol):
    """Boundary to the C3 concurrent multi-camera survey orchestrator.

    Implementations launch one per-camera session under a shared ``run_id`` and
    expose status / abort / progress streaming. C5 ships an in-memory stub
    (:class:`~poc_homography.survey.orchestrator_memory.InMemorySurveyOrchestrator`)
    so the operator surface is exercisable without a live camera; C3 replaces it
    with the real implementation behind this same Protocol.
    """

    def start(self, plan_config: SurveyPlanConfig, camera_ids: Sequence[str]) -> RunHandle:
        """Launch a concurrent run across ``camera_ids`` from ``plan_config``."""
        ...

    def status(self, run_id: str) -> dict[str, CameraStatus] | None:
        """Return per-camera status keyed by ``camera_id``, or ``None`` if unknown."""
        ...

    def abort(self, run_id: str) -> bool:
        """Request graceful abort; return ``True`` if the run existed."""
        ...

    def iter_progress(self, run_id: str) -> Iterator[ProgressEvent]:
        """Yield progress events until every camera reaches a terminal state."""
        ...

    def list_runs(self, limit: int = 20) -> list[RunSummary]:
        """Return up to ``limit`` run summaries, newest first."""
        ...
