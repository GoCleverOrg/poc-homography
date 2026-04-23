"""PostgreSQL-backed CameraConfig repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poc_homography.domain.entities.camera_config import CameraConfig
from poc_homography.infrastructure.models.camera_config import CameraConfigModel
from poc_homography.infrastructure.repositories.base import (
    MixinRepoMapFilter,
    MixinRepoTenantFilter,
    RepoPostgres,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class RepoPostgresCameraConfig(
    RepoPostgres[CameraConfig],
    MixinRepoTenantFilter,
    MixinRepoMapFilter,
):
    """Repository for CameraConfig entities stored in PostgreSQL.

    The ``credential`` value object is stored as a JSONB column.  The default
    ``CameraConfig.to_dict()`` masks the password (``'***'``), so we override
    ``_entity_to_row`` to persist the plaintext credential — matching the
    behaviour of the YAML repository.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, CameraConfigModel, CameraConfig)

    def _entity_to_row(self, entity: CameraConfig) -> dict[str, Any]:
        data = entity.to_dict()
        # Persist actual credential (to_dict masks the password by default).
        data["credential"] = entity.credential.to_dict(include_secret=True)
        # to_dict() omits ip_address when None/falsy; always include it so
        # UPDATE statements can clear a previously-set value.
        data["ip_address"] = entity.ip_address
        return data
