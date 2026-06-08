"""YAML-backed SurveyRun repository with per-frame grouping queries.

Run manifests live at ``data/survey_runs/{run_id}.yaml`` (one file per run).
Individual :class:`FrameRecord` files are partitioned under
``data/survey/{run_id}/{camera_id}/frames/*.yaml``; the grouping queries scan
that layout and filter in Python, mirroring ``RepoYaml._filter_by``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from poc_homography.domain.entities.survey.frame_record import FrameRecord
from poc_homography.domain.entities.survey.pose_catalog import PoseCatalog
from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig
from poc_homography.infrastructure.repositories.base.repo_yaml import RepoYaml

if TYPE_CHECKING:
    from poc_homography.domain.enums.survey_phase import SurveyPhase
    from poc_homography.types import Unitless

logger = logging.getLogger(__name__)

_FRAME_GLOB = "*/*/frames/*.yaml"


class RepoYamlSurveyRun(RepoYaml[SurveyRun]):
    """Repository for :class:`SurveyRun` manifests plus frame grouping queries."""

    def __init__(
        self,
        data_dir: Path,
        frames_dir: Path | None = None,
        *,
        create_dir: bool = True,
    ) -> None:
        super().__init__(data_dir, SurveyRun, create_dir=create_dir)
        self._frames_dir = (
            Path(frames_dir) if frames_dir is not None else self._data_dir.parent / "survey"
        )

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _all_frames(self) -> list[FrameRecord]:
        """Load every :class:`FrameRecord` under the frame layout.

        Malformed or unreadable frame files are skipped with a warning so a
        single bad file does not break a grouping query.
        """
        frames: list[FrameRecord] = []
        if not self._frames_dir.exists():
            return frames
        for path in sorted(self._frames_dir.glob(_FRAME_GLOB)):
            data = self._read_yaml(path)
            if data is None:
                continue
            try:
                frames.append(FrameRecord.from_dict(data))
            except (KeyError, ValueError):
                logger.warning("Skipping malformed frame record at %s", path, exc_info=True)
        return frames

    # ------------------------------------------------------------------
    # Grouping queries
    # ------------------------------------------------------------------

    def get_frames_by_run(self, run_id: str) -> list[FrameRecord]:
        """Return all frames captured in ``run_id``."""
        return [f for f in self._all_frames() if f.capture.run_id == run_id]

    def get_frames_by_phase(self, phase: SurveyPhase) -> list[FrameRecord]:
        """Return all frames captured during ``phase``."""
        return [f for f in self._all_frames() if f.capture.phase == phase]

    def get_frames_by_camera(self, camera_id: str) -> list[FrameRecord]:
        """Return all frames captured by ``camera_id``."""
        return [f for f in self._all_frames() if f.camera.camera_id == camera_id]

    def get_frames_by_zoom_range(self, min_zoom: Unitless, max_zoom: Unitless) -> list[FrameRecord]:
        """Return all frames whose reported zoom is within ``[min, max]``."""
        return [f for f in self._all_frames() if min_zoom <= f.reported.reported_zoom <= max_zoom]

    def get_frames_by_burst(self, burst_id: str) -> list[FrameRecord]:
        """Return all frames belonging to ``burst_id``."""
        return [f for f in self._all_frames() if f.capture.burst_id == burst_id]

    # ------------------------------------------------------------------
    # Plan-config sidecar
    # ------------------------------------------------------------------

    def _plan_config_path(self, run_id: str) -> Path:
        return self._frames_dir / run_id / "plan_config.yaml"

    def save_plan_config(self, run_id: str, config: SurveyPlanConfig) -> bool:
        """Persist ``config`` as a ``plan_config.yaml`` sidecar for ``run_id``.

        Returns ``True`` on success, ``False`` (logged) on any failure.
        """
        path = self._plan_config_path(run_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
        except Exception:
            logger.exception("Failed to save plan config for run %s", run_id)
            return False
        return True

    def load_plan_config(self, run_id: str) -> SurveyPlanConfig:
        """Load the ``plan_config.yaml`` sidecar for ``run_id``.

        Raises:
            KeyError: If no sidecar exists for ``run_id``.
        """
        path = self._plan_config_path(run_id)
        if not path.exists():
            raise KeyError(run_id)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return SurveyPlanConfig.from_dict(data)

    # ------------------------------------------------------------------
    # Pose-catalog sidecar
    # ------------------------------------------------------------------

    def _pose_catalog_path(self, run_id: str) -> Path:
        return self._frames_dir / run_id / "pose_catalog.yaml"

    def save_pose_catalog(self, run_id: str, catalog: PoseCatalog) -> bool:
        """Persist ``catalog`` as a ``pose_catalog.yaml`` sidecar for ``run_id``.

        Returns ``True`` on success, ``False`` (logged) on any failure.
        """
        path = self._pose_catalog_path(run_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(catalog.to_dict(), f, default_flow_style=False, sort_keys=False)
        except Exception:
            logger.exception("Failed to save pose catalog for run %s", run_id)
            return False
        return True

    def load_pose_catalog(self, run_id: str) -> PoseCatalog:
        """Load the ``pose_catalog.yaml`` sidecar for ``run_id``.

        Raises:
            KeyError: If no sidecar exists for ``run_id``.
        """
        path = self._pose_catalog_path(run_id)
        if not path.exists():
            raise KeyError(run_id)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return PoseCatalog.from_dict(data)

    # ------------------------------------------------------------------
    # Multi-visit grouping
    # ------------------------------------------------------------------

    def get_runs_by_camera_and_pose(self, camera_id: str, pose_id: str) -> list[SurveyRun]:
        """Return every run for ``camera_id`` whose pose catalog holds ``pose_id``.

        Runs are read from the manifest store; a run is included only when its
        ``pose_catalog.yaml`` sidecar exists and its ``entries`` contain
        ``pose_id``. Multiple runs for the same camera that visited the same
        physical pose are returned together.
        """
        matches: list[SurveyRun] = []
        for run in self.get_all():
            if run.camera_id != camera_id:
                continue
            try:
                catalog = self.load_pose_catalog(run.id)
            except KeyError:
                continue
            if pose_id in catalog.entries:
                matches.append(run)
        return matches
