"""PostgreSQL-backed LensCalibrationTable repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.infrastructure.models.lens_calibration_table import (
    LensCalibrationTableModel,
)
from poc_homography.infrastructure.repositories.base import RepoPostgres

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.infrastructure.database import Base


class RepoPostgresLensCalibrationTable(RepoPostgres[LensCalibrationTable]):
    """Repository for LensCalibrationTable entities stored in PostgreSQL.

    Entries are persisted as a JSONB array of ``ZoomCalibrationEntry`` dicts.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, LensCalibrationTableModel, LensCalibrationTable)

    # -- serialisation overrides -------------------------------------------

    def _entity_to_row(self, entity: LensCalibrationTable) -> dict[str, Any]:
        return {
            "id": entity.id,
            "entries": [e.to_dict() for e in entity.entries],
            "created_date": entity.created_date,
            "last_modified": entity.last_modified,
        }

    def _row_to_entity(self, row: Base) -> LensCalibrationTable:
        return LensCalibrationTable.from_dict(
            {
                "id": row.id,  # type: ignore[attr-defined]
                "entries": row.entries,  # type: ignore[attr-defined]
                "created_date": row.created_date,  # type: ignore[attr-defined]
                "last_modified": row.last_modified,  # type: ignore[attr-defined]
            }
        )
