"""PostgreSQL-backed PtzRegistration repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poc_homography.domain.entities.ptz_registration import (
    PtzRegistration,
    metrics_of,
    result_from_dict,
    result_to_dict,
    stats_from_dict,
    stats_to_dict,
)
from poc_homography.infrastructure.models.ptz_registration import PtzRegistrationModel
from poc_homography.infrastructure.repositories.base import RepoPostgres

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.infrastructure.database import Base


class RepoPostgresPtzRegistration(RepoPostgres[PtzRegistration]):
    """Repository for :class:`PtzRegistration` records stored in PostgreSQL.

    The solver matrices and metrics are split across dedicated JSONB columns
    (``homography_matrix`` / ``inverse_matrix`` / ``metrics``) plus an optional
    ``reprojection_stats`` JSONB. The inherited ``save`` / ``get`` / ``delete`` /
    ``exists`` operate by the synthetic primary key; :meth:`get_by_camera` adds a
    camera-scoped query over the indexed ``camera_id`` column.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, PtzRegistrationModel, PtzRegistration)

    # -- serialisation overrides -------------------------------------------

    def _entity_to_row(self, entity: PtzRegistration) -> dict[str, Any]:
        reg_dict = result_to_dict(entity.registration)
        return {
            "id": entity.id,
            "camera_id": entity.camera_id,
            "run_id": entity.run_id,
            "commanded_zoom": entity.commanded_zoom,
            "commanded_tilt": entity.commanded_tilt,
            "homography_matrix": reg_dict["homography_matrix"],
            "inverse_matrix": reg_dict["inverse_matrix"],
            "metrics": metrics_of(entity.registration),
            "reprojection_stats": (
                stats_to_dict(entity.reprojection_stats)
                if entity.reprojection_stats is not None
                else None
            ),
            "created_date": entity.created_date,
            "last_modified": entity.last_modified,
        }

    def _row_to_entity(self, row: Base) -> PtzRegistration:
        registration = result_from_dict(
            {
                "homography_matrix": row.homography_matrix,  # type: ignore[attr-defined]
                "inverse_matrix": row.inverse_matrix,  # type: ignore[attr-defined]
                **row.metrics,  # type: ignore[attr-defined]
                "run_id": row.run_id,  # type: ignore[attr-defined]
                "camera_id": row.camera_id,  # type: ignore[attr-defined]
                "commanded_zoom": row.commanded_zoom,  # type: ignore[attr-defined]
                "commanded_tilt": row.commanded_tilt,  # type: ignore[attr-defined]
            }
        )
        stats = row.reprojection_stats  # type: ignore[attr-defined]
        return PtzRegistration(
            camera_id=row.camera_id,  # type: ignore[attr-defined]
            run_id=row.run_id,  # type: ignore[attr-defined]
            commanded_zoom=row.commanded_zoom,  # type: ignore[attr-defined]
            commanded_tilt=row.commanded_tilt,  # type: ignore[attr-defined]
            registration=registration,
            reprojection_stats=stats_from_dict(stats) if stats is not None else None,
            created_date=row.created_date,  # type: ignore[attr-defined]
            last_modified=row.last_modified,  # type: ignore[attr-defined]
        )

    # -- camera-scoped query ------------------------------------------------

    def get_by_camera(self, camera_id: str) -> list[PtzRegistration]:
        """Return every registration record stored for ``camera_id``."""
        return list(self._filter_by("camera_id", camera_id).values())
