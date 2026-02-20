"""YAML-based Map repository."""

from pathlib import Path

from poc_homography.domain.entities.map import Map
from poc_homography.infrastructure.repositories.base import MixinRepoTenantFilter, RepoYaml


class RepoYamlMap(RepoYaml[Map], MixinRepoTenantFilter):
    """Repository for Map entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, Map, create_dir=False)
