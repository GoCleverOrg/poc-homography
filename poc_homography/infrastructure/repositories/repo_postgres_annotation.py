"""PostgreSQL-backed Annotation repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from poc_homography.domain.entities.annotation import Annotation
from poc_homography.infrastructure.models.annotation import AnnotationModel
from poc_homography.infrastructure.repositories.base import RepoPostgres

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class RepoPostgresAnnotation(RepoPostgres[Annotation]):
    """Repository for Annotation entities stored in PostgreSQL.

    Composite identity: ``{frame_id}/{gcp_id}`` stored in the ``id`` column.
    The ``camera_pose`` (PTZState) and ``pixel`` (PixelPoint) value objects
    are persisted as JSONB.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, AnnotationModel, Annotation)

    def _entity_to_row(self, entity: Annotation) -> dict[str, Any]:
        return {
            "id": entity.id,
            "gcp_id": entity.gcp_id,
            "frame_id": entity.frame_id,
            "camera_pose": entity.camera_pose.to_dict(),
            "pixel": entity.pixel.to_dict(),
        }

    def get_by_frame_id(self, frame_id: str) -> list[Annotation]:
        """Return all annotations belonging to *frame_id*."""
        stmt = select(self._model_cls).where(AnnotationModel.frame_id == frame_id)
        rows = self._session.execute(stmt).scalars().all()
        return [self._row_to_entity(row) for row in rows]
