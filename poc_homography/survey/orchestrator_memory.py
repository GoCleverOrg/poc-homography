"""In-memory placeholder orchestrator (camera-free) for the C5 operator surface.

This is the C5 stand-in for the real C3 concurrent multi-camera orchestrator.
It satisfies the ``ConcurrentSurveyOrchestrator`` protocol (see
``poc_homography.domain.protocols.survey_orchestrator``)
without touching a live camera: runs and per-camera progress are simulated
deterministically in memory. The CLI, Django, and FastAPI layers wire this by
default until C3 lands; tests inject deterministic ``id_factory`` / ``clock``
callables for reproducible assertions.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.domain.enums.survey_run_status import SurveyRunStatus
from poc_homography.domain.protocols.survey_orchestrator import (
    CameraStatus,
    ProgressEvent,
    RunHandle,
    RunSummary,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig

# Phase numbers (1..9) map to the nine-member SurveyPhase enum in declaration order.
_PHASE_BY_NUMBER: dict[int, SurveyPhase] = dict(enumerate(SurveyPhase, start=1))

DEFAULT_FRAMES_PER_PHASE = 2


@dataclass
class _CameraState:
    """Mutable per-camera progress within a simulated run."""

    session_id: str
    phase: str
    frame_count: int = 0
    status: str = SurveyRunStatus.RUNNING.value


@dataclass
class _RunState:
    """Mutable state for one simulated run."""

    run_id: str
    started_at: datetime
    phases: tuple[SurveyPhase, ...]
    cameras: dict[str, _CameraState] = field(default_factory=dict)
    status: str = SurveyRunStatus.RUNNING.value


class InMemorySurveyOrchestrator:
    """Deterministic, camera-free orchestrator stub for the operator surface."""

    def __init__(
        self,
        *,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
        frames_per_phase: int = DEFAULT_FRAMES_PER_PHASE,
    ) -> None:
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))
        self._frames_per_phase = frames_per_phase
        self._runs: dict[str, _RunState] = {}
        self._lock = threading.Lock()

    def start(self, plan_config: SurveyPlanConfig, camera_ids: Sequence[str]) -> RunHandle:
        """Register a run with one session per camera; return its handle."""
        phases = tuple(
            _PHASE_BY_NUMBER[n] for n in sorted(plan_config.enabled_phases) if n in _PHASE_BY_NUMBER
        )
        with self._lock:
            run_id = self._id_factory()
            first_phase = phases[0].value if phases else SurveyPhase.CAMERA_INVENTORY.value
            cameras = {
                camera_id: _CameraState(session_id=self._id_factory(), phase=first_phase)
                for camera_id in camera_ids
            }
            self._runs[run_id] = _RunState(
                run_id=run_id,
                started_at=self._clock(),
                phases=phases,
                cameras=cameras,
            )
            session_ids = {cid: state.session_id for cid, state in cameras.items()}
        return RunHandle(run_id=run_id, session_ids=session_ids)

    def status(self, run_id: str) -> dict[str, CameraStatus] | None:
        """Return per-camera status keyed by ``camera_id``, or ``None`` if unknown."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return {
                cid: CameraStatus(
                    session_id=cam.session_id,
                    phase=cam.phase,
                    frame_count=cam.frame_count,
                    status=cam.status,
                )
                for cid, cam in run.cameras.items()
            }

    def abort(self, run_id: str) -> bool:
        """Mark the run and all its cameras aborted; return whether it existed."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            run.status = SurveyRunStatus.ABORTED.value
            for cam in run.cameras.values():
                cam.status = SurveyRunStatus.ABORTED.value
            return True

    def iter_progress(self, run_id: str) -> Iterator[ProgressEvent]:
        """Advance every camera through each enabled phase, yielding one event each."""
        run = self._runs.get(run_id)
        if run is None:
            return
        for phase in run.phases:
            for camera_id, cam in run.cameras.items():
                with self._lock:
                    if run.status == SurveyRunStatus.ABORTED.value:
                        return
                    cam.phase = phase.value
                    cam.frame_count += self._frames_per_phase
                    event = ProgressEvent(
                        camera_id=camera_id,
                        phase=cam.phase,
                        frame_count=cam.frame_count,
                        status=cam.status,
                    )
                yield event
        with self._lock:
            if run.status != SurveyRunStatus.ABORTED.value:
                run.status = SurveyRunStatus.COMPLETED.value
                for cam in run.cameras.values():
                    cam.status = SurveyRunStatus.COMPLETED.value

    def list_runs(self, limit: int = 20) -> list[RunSummary]:
        """Return up to ``limit`` run summaries, newest first."""
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)
            return [
                RunSummary(
                    run_id=run.run_id,
                    start_time=run.started_at.isoformat(),
                    camera_count=len(run.cameras),
                    total_frame_count=sum(c.frame_count for c in run.cameras.values()),
                    status=run.status,
                )
                for run in runs[:limit]
            ]
