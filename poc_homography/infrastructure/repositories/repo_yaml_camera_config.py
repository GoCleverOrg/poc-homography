"""YAML-based CameraConfig repository."""

from pathlib import Path

from poc_homography.domain.entities.camera_config import CameraConfig
from poc_homography.infrastructure.repositories.base import MixinRepoMapFilter, RepoYaml


class RepoYamlCameraConfig(RepoYaml[CameraConfig], MixinRepoMapFilter):
    """Repository for CameraConfig entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, CameraConfig)
