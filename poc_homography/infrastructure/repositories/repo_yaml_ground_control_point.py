"""YAML-based GroundControlPoint repository."""

from pathlib import Path

from poc_homography.domain.entities.ground_control_point import GroundControlPoint
from poc_homography.infrastructure.repositories.base import MixinRepoMapFilter, RepoYaml


class RepoYamlGroundControlPoint(RepoYaml[GroundControlPoint], MixinRepoMapFilter):
    """Repository for GroundControlPoint entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, GroundControlPoint)
