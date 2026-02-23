"""YAML-based LineAnnotation repository."""

from pathlib import Path

from poc_homography.domain.entities.line_annotation import LineAnnotation
from poc_homography.infrastructure.repositories.base import RepoYaml


class RepoYamlLineAnnotation(RepoYaml[LineAnnotation]):
    """Repository for LineAnnotation entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, LineAnnotation)
