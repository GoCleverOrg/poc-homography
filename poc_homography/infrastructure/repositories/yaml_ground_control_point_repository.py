"""YAML-based GroundControlPoint repository."""

from pathlib import Path

from poc_homography.domain.entities.ground_control_point import GroundControlPoint
from poc_homography.infrastructure.repositories.base import MapFilterMixin, YamlRepositoryBase


class YamlGroundControlPointRepository(YamlRepositoryBase[GroundControlPoint], MapFilterMixin):
    """Repository for GroundControlPoint entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, GroundControlPoint)
