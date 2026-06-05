"""YAML-backed :class:`PhaseSink` writing C1 records into the survey layout.

Records are written one file per record into the partitioned layout that
:class:`~poc_homography.infrastructure.repositories.repo_yaml_survey_run.RepoYamlSurveyRun`
reads back: frames at ``{root}/{run_id}/{camera_id}/frames/{capture_id}.yaml``,
bursts alongside them under ``bursts/``, and the per-run inventory under
``inventory/``. This keeps the phase layer persistence-agnostic (it depends only
on the :class:`PhaseSink` port) while letting a full multi-phase run produce a
real, queryable dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path

    from poc_homography.domain.entities.survey.frame_record import FrameRecord
    from poc_homography.domain.entities.survey.inventory_record import (
        CameraInventoryRecord,
    )
    from poc_homography.domain.entities.survey.video_burst_record import (
        VideoBurstRecord,
    )


class YamlPhaseSink:
    """Persist survey-phase C1 records as one YAML file per record.

    Implements the :class:`~poc_homography.survey.phases.common.PhaseSink`
    protocol structurally.
    """

    def __init__(self, root: Path) -> None:
        """Initialise the sink.

        Args:
            root: The survey data root (e.g. ``data/survey``); per-run and
                per-camera sub-directories are created on demand.
        """
        self._root = root

    def save_inventory(self, record: CameraInventoryRecord) -> None:
        """Write the Phase 1 inventory record under ``inventory/``."""
        path = self._dir(record.run_id, record.camera_id, "inventory") / f"{record.id}.yaml"
        _write_yaml(path, record.to_dict())

    def save_frame(self, record: FrameRecord) -> None:
        """Write a per-frame C1 record under ``frames/``."""
        run_id = record.capture.run_id
        camera_id = record.camera.camera_id
        path = self._dir(run_id, camera_id, "frames") / f"{record.id}.yaml"
        _write_yaml(path, record.to_dict())

    def save_burst(self, record: VideoBurstRecord) -> None:
        """Write a Phase 8 video-burst record under ``bursts/``."""
        path = self._dir(record.run_id, record.camera_id, "bursts") / f"{record.id}.yaml"
        _write_yaml(path, record.to_dict())

    def _dir(self, run_id: str, camera_id: str, leaf: str) -> Path:
        """Return (creating) the ``{root}/{run_id}/{camera_id}/{leaf}`` dir."""
        directory = self._root / run_id / camera_id / leaf
        directory.mkdir(parents=True, exist_ok=True)
        return directory


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` matching the project's YAML conventions."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(data, handle, default_flow_style=False, sort_keys=False)
