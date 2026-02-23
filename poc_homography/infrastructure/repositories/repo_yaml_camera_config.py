"""YAML-based CameraConfig repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poc_homography.domain.entities.camera_config import CameraConfig
from poc_homography.infrastructure.repositories.base import (
    MixinRepoMapFilter,
    MixinRepoTenantFilter,
    RepoYaml,
)

if TYPE_CHECKING:
    from pathlib import Path


class RepoYamlCameraConfig(RepoYaml[CameraConfig], MixinRepoMapFilter, MixinRepoTenantFilter):
    """Repository for CameraConfig entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, CameraConfig)

    def _entity_to_dict(self, entity: CameraConfig) -> dict[str, Any]:
        data = entity.to_dict()
        # Persist actual credential for storage (to_dict masks by default)
        data["credential"] = entity.credential.to_dict(include_secret=True)
        return data
