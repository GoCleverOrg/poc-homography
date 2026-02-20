"""YAML-based Tenant repository."""

from pathlib import Path

from poc_homography.domain.entities.tenant import Tenant
from poc_homography.infrastructure.repositories.base import RepoYaml


class RepoYamlTenant(RepoYaml[Tenant]):
    """Repository for Tenant entities stored as YAML files."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, Tenant)
