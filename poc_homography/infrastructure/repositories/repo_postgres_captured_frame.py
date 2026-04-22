"""PostgreSQL-backed CapturedFrame repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select

from poc_homography.domain.entities.annotation import Annotation
from poc_homography.domain.entities.captured_frame import CapturedFrame
from poc_homography.infrastructure.models.annotation import AnnotationModel
from poc_homography.infrastructure.models.captured_frame import CapturedFrameModel
from poc_homography.infrastructure.repositories.base import RepoPostgres

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.orm import Session

    from poc_homography.infrastructure.database import Base


class RepoPostgresCapturedFrame(RepoPostgres[CapturedFrame]):
    """Repository for CapturedFrame entities stored in PostgreSQL.

    The domain entity uses a 3-part composite ID ``map_id/camera_name/timestamp``
    which is stored as the PK string.  The ORM model also has separate
    ``map_id``, ``camera_name``, and ``timestamp`` columns used for indexed
    queries.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, CapturedFrameModel, CapturedFrame)

    # -- serialisation overrides -------------------------------------------

    def _entity_to_row(self, entity: CapturedFrame) -> dict[str, Any]:
        return {
            "id": entity.id,
            "map_id": entity.map_id,
            "camera_name": entity.camera_name,
            "timestamp": entity.timestamp,
            "ptz_state": entity.ptz_state.to_dict(),
            "image_path": str(entity.image_path),
        }

    def _row_to_entity(self, row: Base) -> CapturedFrame:
        return CapturedFrame.from_dict(
            {
                "id": row.id,  # type: ignore[attr-defined]
                "map_id": row.map_id,  # type: ignore[attr-defined]
                "camera_name": row.camera_name,  # type: ignore[attr-defined]
                "timestamp": row.timestamp.isoformat(),  # type: ignore[attr-defined]
                "ptz_state": row.ptz_state,  # type: ignore[attr-defined]
                "image_path": row.image_path,  # type: ignore[attr-defined]
            }
        )

    # -- custom query methods ----------------------------------------------

    def get_by_map(self, map_id: str) -> list[CapturedFrame]:
        """Return all frames belonging to *map_id*."""
        stmt = select(CapturedFrameModel).where(CapturedFrameModel.map_id == map_id)
        rows = self._session.execute(stmt).scalars().all()
        return [self._row_to_entity(row) for row in rows]

    def get_by_camera(self, map_id: str, camera_name: str) -> list[CapturedFrame]:
        """Return all frames for a specific camera within a map."""
        stmt = (
            select(CapturedFrameModel)
            .where(CapturedFrameModel.map_id == map_id)
            .where(CapturedFrameModel.camera_name == camera_name)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [self._row_to_entity(row) for row in rows]

    # -- annotation helpers ------------------------------------------------

    def get_annotations(self, frame_id: str) -> list[Annotation]:
        """Return all annotations for *frame_id*."""
        stmt = select(AnnotationModel).where(AnnotationModel.frame_id == frame_id)
        rows = self._session.execute(stmt).scalars().all()
        return [Annotation.from_dict(self._annotation_row_to_dict(row)) for row in rows]

    def save_annotations(self, frame_id: str, annotations: list[Annotation]) -> None:
        """Replace all annotations for *frame_id*.

        Raises:
            ValueError: If the frame does not exist.
        """
        if not self.exists(frame_id):
            raise ValueError(f"Cannot save annotations: frame '{frame_id}' not found")

        # Delete existing annotations for this frame, then insert new ones.
        self._session.execute(delete(AnnotationModel).where(AnnotationModel.frame_id == frame_id))

        for ann in annotations:
            row = AnnotationModel(
                id=ann.id,
                gcp_id=ann.gcp_id,
                frame_id=ann.frame_id,
                camera_pose=ann.camera_pose.to_dict(),
                pixel=ann.pixel.to_dict(),
            )
            self._session.add(row)

        self._session.flush()

    def delete_annotations(self, frame_id: str) -> bool:
        """Delete all annotations for *frame_id*.

        Returns:
            True if any annotations were deleted, False otherwise.
        """
        result = self._session.execute(
            delete(AnnotationModel).where(AnnotationModel.frame_id == frame_id)
        )
        self._session.flush()
        return (getattr(result, "rowcount", 0) or 0) > 0

    # -- private helpers ---------------------------------------------------

    @staticmethod
    def _annotation_row_to_dict(row: AnnotationModel) -> dict[str, Any]:
        return {
            "gcp_id": row.gcp_id,
            "frame_id": row.frame_id,
            "camera_pose": row.camera_pose,
            "pixel": row.pixel,
        }

    # -- filesystem helpers (thin wrappers) --------------------------------

    @staticmethod
    def get_image_path(entity: CapturedFrame, frames_dir: Path) -> Path:
        """Return the absolute path to a frame's image file.

        Args:
            entity: The captured frame entity.
            frames_dir: Root directory where frame images are stored
                        (typically ``data/frames/``).

        Returns:
            Absolute path to the image file.
        """
        return frames_dir / entity.map_id / entity.camera_name / str(entity.image_path)
