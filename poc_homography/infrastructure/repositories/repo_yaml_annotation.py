"""YAML-based Annotation repository."""

from pathlib import Path

from poc_homography.domain.entities.annotation import Annotation
from poc_homography.infrastructure.repositories.base import RepoYaml


class RepoYamlAnnotation(RepoYaml[Annotation]):
    """Repository for Annotation entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, Annotation)

    def get_by_frame_id(self, frame_id: str) -> list[Annotation]:
        """Return annotations for a specific frame using prefix-based glob."""
        prefix = frame_id.replace(self._id_sep, self._filename_sep)
        results: list[Annotation] = []
        for path in self._data_dir.glob(f"{prefix}{self._filename_sep}*.yaml"):
            entity = self.get(self._path_to_id(path))
            if entity:
                results.append(entity)
        return results
