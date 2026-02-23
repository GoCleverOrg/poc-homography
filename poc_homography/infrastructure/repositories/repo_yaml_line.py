"""YAML-based Line repository."""

from pathlib import Path

from poc_homography.domain.entities.line import Line
from poc_homography.infrastructure.repositories.base import MixinRepoMapFilter, RepoYaml


class RepoYamlLine(RepoYaml[Line], MixinRepoMapFilter):
    """Repository for Line entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, Line)

    def get_by_map_id(self, map_id: str) -> list[Line]:
        """Return lines for a specific map using prefix-based glob."""
        prefix = map_id.replace(self._id_sep, self._filename_sep)
        results: list[Line] = []
        for path in self._data_dir.glob(f"{prefix}{self._filename_sep}*.yaml"):
            entity = self.get(self._path_to_id(path))
            if entity:
                results.append(entity)
        return results
