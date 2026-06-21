"""YAML-based PtzRegistration repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.domain.entities.ptz_registration import PtzRegistration
from poc_homography.infrastructure.repositories.base import RepoYaml

if TYPE_CHECKING:
    from pathlib import Path


class RepoYamlPtzRegistration(RepoYaml[PtzRegistration]):
    """Repository for :class:`PtzRegistration` records stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, PtzRegistration)

    def get_by_camera(self, camera_id: str) -> list[PtzRegistration]:
        """Return every registration record stored for ``camera_id``."""
        return list(self._filter_by("camera_id", camera_id).values())
