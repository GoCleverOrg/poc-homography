"""Service facade for the multi-camera survey operator surface (C5, issue #262).

A thin, transport-agnostic layer that the CLI (``hom survey ...``), the Django
views, and the FastAPI endpoints all delegate to. It composes a
:class:`~poc_homography.domain.protocols.survey_orchestrator.ConcurrentSurveyOrchestrator`
(run lifecycle) with an optional
:class:`~poc_homography.domain.protocols.survey_run_repository.SurveyRunRepository`
(dataset browsing) and returns plain JSON-friendly dicts so every transport
serialises results identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.domain.enums.survey_phase import SurveyPhase
from poc_homography.survey.orchestrator_memory import InMemorySurveyOrchestrator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from poc_homography.domain.protocols.survey_orchestrator import (
        ConcurrentSurveyOrchestrator,
    )
    from poc_homography.domain.protocols.survey_run_repository import SurveyRunRepository
    from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig

# Phase numbers (1..9) map to the nine-member SurveyPhase enum in declaration order.
_PHASE_BY_NUMBER: dict[int, SurveyPhase] = dict(enumerate(SurveyPhase, start=1))


class SurveyRunService:
    """Transport-agnostic facade over the orchestrator and run repository."""

    def __init__(
        self,
        orchestrator: ConcurrentSurveyOrchestrator,
        run_repo: SurveyRunRepository | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._run_repo = run_repo

    def start_run(self, plan_config: SurveyPlanConfig, camera_ids: list[str]) -> dict[str, object]:
        """Launch a run; return ``{"run_id", "session_ids"}``."""
        handle = self._orchestrator.start(plan_config, camera_ids)
        return {"run_id": handle.run_id, "session_ids": dict(handle.session_ids)}

    def get_status(self, run_id: str) -> dict[str, object] | None:
        """Return ``{"run_id", "cameras": {...}}`` or ``None`` if the run is unknown."""
        statuses = self._orchestrator.status(run_id)
        if statuses is None:
            return None
        return {
            "run_id": run_id,
            "cameras": {
                camera_id: {
                    "session_id": cam.session_id,
                    "phase": cam.phase,
                    "frame_count": cam.frame_count,
                    "status": cam.status,
                }
                for camera_id, cam in statuses.items()
            },
        }

    def abort_run(self, run_id: str) -> dict[str, str] | None:
        """Request graceful abort; return a message dict or ``None`` if unknown."""
        if not self._orchestrator.abort(run_id):
            return None
        return {"run_id": run_id, "message": "Run abort requested"}

    def list_runs(self, limit: int = 20) -> list[dict[str, object]]:
        """Return run summaries (newest first) as JSON-friendly dicts."""
        return [
            {
                "run_id": summary.run_id,
                "start_time": summary.start_time,
                "camera_count": summary.camera_count,
                "total_frame_count": summary.total_frame_count,
                "status": summary.status,
            }
            for summary in self._orchestrator.list_runs(limit)
        ]

    def iter_progress(self, run_id: str) -> Iterator[dict[str, object]]:
        """Yield per-camera, per-phase progress dicts until the run terminates."""
        for event in self._orchestrator.iter_progress(run_id):
            yield {
                "camera_id": event.camera_id,
                "phase": event.phase,
                "frame_count": event.frame_count,
                "status": event.status,
            }

    def browse_groups(
        self,
        run_id: str,
        *,
        phase: int | None = None,
        camera: str | None = None,
        zoom: float | None = None,
    ) -> list[dict[str, object]]:
        """Return dataset grouping keys with frame counts for ``run_id``.

        Frames are grouped by ``(phase, camera, reported_zoom)`` and counted.
        Optional filters narrow the result: ``phase`` is a 1..9 phase number,
        ``camera`` a camera id, and ``zoom`` a reported zoom factor (matched to
        one decimal place). Returns an empty list when no run repository is wired.
        """
        if self._run_repo is None:
            return []
        phase_value = _PHASE_BY_NUMBER[phase].value if phase is not None else None
        counts: dict[tuple[str, str, float], int] = {}
        for frame in self._run_repo.get_frames_by_run(run_id):
            frame_phase = frame.capture.phase.value
            frame_camera = frame.camera.camera_id
            frame_zoom = round(float(frame.reported.reported_zoom), 1)
            if phase_value is not None and frame_phase != phase_value:
                continue
            if camera is not None and frame_camera != camera:
                continue
            if zoom is not None and frame_zoom != round(float(zoom), 1):
                continue
            key = (frame_phase, frame_camera, frame_zoom)
            counts[key] = counts.get(key, 0) + 1
        return [
            {"phase": key[0], "camera": key[1], "zoom": key[2], "frame_count": count}
            for key, count in sorted(counts.items())
        ]


def build_survey_run_service(run_repo: SurveyRunRepository | None = None) -> SurveyRunService:
    """Build the default operator-surface service.

    Wires the C5 in-memory orchestrator stub (camera-free) pending the real C3
    orchestrator. ``run_repo`` enables dataset browsing when supplied.
    """
    return SurveyRunService(orchestrator=InMemorySurveyOrchestrator(), run_repo=run_repo)
