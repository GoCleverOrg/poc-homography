"""PostgreSQL-backed LineAnnotation repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from poc_homography.domain.entities.line_annotation import LineAnnotation
from poc_homography.infrastructure.models.line_annotation import LineAnnotationModel
from poc_homography.infrastructure.repositories.base import RepoPostgres

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class RepoPostgresLineAnnotation(RepoPostgres[LineAnnotation]):
    """Repository for LineAnnotation entities stored in PostgreSQL.

    Composite identity: ``{frame_id}/{line_id}`` stored in the ``id`` column.
    The ``camera_pose`` (PTZState), ``start_pixel`` / ``end_pixel`` (PixelPoint),
    and ``points`` (list of coordinate pairs) are persisted as JSONB.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, LineAnnotationModel, LineAnnotation)

    def _entity_to_row(self, entity: LineAnnotation) -> dict[str, Any]:
        return {
            "id": entity.id,
            "line_id": entity.line_id,
            "frame_id": entity.frame_id,
            "camera_pose": entity.camera_pose.to_dict(),
            "start_pixel": entity.start_pixel.to_dict(),
            "end_pixel": entity.end_pixel.to_dict(),
            "points": (
                [[float(p.x), float(p.y)] for p in entity.points]
                if entity.points is not None
                else None
            ),
        }

    def get_by_frame_id(self, frame_id: str) -> list[LineAnnotation]:
        """Return all line annotations belonging to *frame_id*."""
        stmt = select(self._model_cls).where(LineAnnotationModel.frame_id == frame_id)
        rows = self._session.execute(stmt).scalars().all()
        return [self._row_to_entity(row) for row in rows]
