"""PostgreSQL-backed CalibrationLineTraceSet repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poc_homography.domain.entities.calibration_line_trace_set import (
    CalibrationLineTraceSet,
)
from poc_homography.infrastructure.models.calibration_line_trace_set import (
    CalibrationLineTraceSetModel,
)
from poc_homography.infrastructure.repositories.base import RepoPostgres

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.infrastructure.database import Base


class RepoPostgresCalibrationLineTraceSet(RepoPostgres[CalibrationLineTraceSet]):
    """Repository for CalibrationLineTraceSet entities stored in PostgreSQL.

    ``camera_pose`` is stored as JSONB (PTZState dict) and ``line_traces``
    as a JSONB array of LineTrace dicts.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, CalibrationLineTraceSetModel, CalibrationLineTraceSet)

    # -- serialisation overrides -------------------------------------------

    def _entity_to_row(self, entity: CalibrationLineTraceSet) -> dict[str, Any]:
        return {
            "id": entity.id,
            "name": entity.name,
            "image": entity.image,
            "camera_pose": entity.camera_pose.to_dict(),
            "line_traces": [lt.to_dict() for lt in entity.line_traces],
        }

    def _row_to_entity(self, row: Base) -> CalibrationLineTraceSet:
        return CalibrationLineTraceSet.from_dict(
            {
                "name": row.name,  # type: ignore[attr-defined]
                "image": row.image,  # type: ignore[attr-defined]
                "camera_pose": row.camera_pose,  # type: ignore[attr-defined]
                "line_traces": row.line_traces,  # type: ignore[attr-defined]
            }
        )
