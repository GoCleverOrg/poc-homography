"""YAML-based Annotation repository."""

from pathlib import Path

from poc_homography.domain.entities.annotation import Annotation
from poc_homography.infrastructure.repositories.base import RepoYaml


class RepoYamlAnnotation(RepoYaml[Annotation]):
    """Repository for Annotation entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, Annotation)
