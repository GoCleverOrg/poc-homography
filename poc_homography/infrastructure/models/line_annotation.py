"""SQLAlchemy ORM model for the LineAnnotation entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.captured_frame import CapturedFrameModel


class LineAnnotationModel(Base):
    """Line annotations table — a camera observation of a map line.

    Domain entity: ``poc_homography.domain.entities.line_annotation.LineAnnotation``

    Composite identity is (frame_id, line_id).  The ``id`` column stores the
    composite string ``frame_id/line_id`` as a surrogate PK.

    Value objects stored as JSONB:
      - camera_pose: ``{pan_raw, tilt_deg, zoom}``  (PTZState)
      - start_pixel: ``{x, y}``  (PixelPoint)
      - end_pixel: ``{x, y}``  (PixelPoint)
      - points: ``[[x, y], ...]`` or null  (list of PixelPoint coords)
    """

    __tablename__ = "line_annotations"
    __table_args__ = (
        UniqueConstraint("frame_id", "line_id", name="uq_line_annotation_frame_line"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    line_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    frame_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("captured_frames.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    camera_pose: Mapped[dict] = mapped_column(JSONB, nullable=False)
    start_pixel: Mapped[dict] = mapped_column(JSONB, nullable=False)
    end_pixel: Mapped[dict] = mapped_column(JSONB, nullable=False)
    points: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # -- relationships --
    frame: Mapped[CapturedFrameModel] = relationship(
        "CapturedFrameModel", back_populates="line_annotations"
    )


__all__ = ["LineAnnotationModel"]
