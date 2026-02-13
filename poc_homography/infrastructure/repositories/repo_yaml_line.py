"""YAML-based Line repository."""

from pathlib import Path

from poc_homography.domain.entities.line import Line
from poc_homography.infrastructure.repositories.base import MixinRepoMapFilter, RepoYaml


class RepoYamlLine(RepoYaml[Line], MixinRepoMapFilter):
    """Repository for Line entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, Line)
