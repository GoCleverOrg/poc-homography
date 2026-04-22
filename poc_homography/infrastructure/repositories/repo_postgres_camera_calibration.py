"""PostgreSQL-backed CameraCalibration repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.domain.entities.camera_calibration import CameraCalibration
from poc_homography.infrastructure.models.camera_calibration import CameraCalibrationModel
from poc_homography.infrastructure.repositories.base import RepoPostgres

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class RepoPostgresCameraCalibration(RepoPostgres[CameraCalibration]):
    """Repository for CameraCalibration entities stored in PostgreSQL.

    Nested value objects (``position``, ``base_orientation``, ``distortion``)
    are stored as JSONB columns.  ``CameraCalibration.to_dict()`` already
    produces the correct shape so no serialisation override is needed.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, CameraCalibrationModel, CameraCalibration)
