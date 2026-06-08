"""Protocol for the survey-run persistence boundary used by the operator surface.

C5 (issue #262) adds the ``plan_config.yaml`` sidecar contract on top of the C1
``SurveyRun`` repositories. The concrete YAML and Postgres repos implement these
methods; the service layer depends only on this Protocol so it can be tested
with an in-memory fake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from poc_homography.domain.entities.survey.frame_record import FrameRecord
    from poc_homography.domain.entities.survey.pose_catalog import PoseCatalog
    from poc_homography.domain.entities.survey.survey_run import SurveyRun
    from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig


class SurveyRunRepository(Protocol):
    """Persistence operations the survey operator surface relies on.

    A subset of the full C1 repository API — only the methods C5 consumes for
    reproducibility (``plan_config``) and dataset browsing (frame grouping).
    """

    def save_plan_config(self, run_id: str, config: SurveyPlanConfig) -> bool:
        """Persist ``config`` for ``run_id``; return ``True`` on success."""
        ...

    def load_plan_config(self, run_id: str) -> SurveyPlanConfig:
        """Load the persisted plan config for ``run_id``.

        Raises:
            KeyError: If no plan config is stored for ``run_id``.
        """
        ...

    def get_frames_by_run(self, run_id: str) -> list[FrameRecord]:
        """Return every frame captured in ``run_id``."""
        ...

    def save_pose_catalog(self, run_id: str, catalog: PoseCatalog) -> bool:
        """Persist ``catalog`` for ``run_id``; return ``True`` on success."""
        ...

    def load_pose_catalog(self, run_id: str) -> PoseCatalog:
        """Load the persisted pose catalog for ``run_id``.

        Raises:
            KeyError: If no pose catalog is stored for ``run_id``.
        """
        ...

    def get_runs_by_camera_and_pose(self, camera_id: str, pose_id: str) -> list[SurveyRun]:
        """Return every run for ``camera_id`` whose pose catalog holds ``pose_id``.

        Because pose ids are deterministic across runs, multiple multi-visit
        runs for one camera that visited the same physical pose are returned
        together — the ``(camera_id, pose_id)`` grouping linkage.
        """
        ...
