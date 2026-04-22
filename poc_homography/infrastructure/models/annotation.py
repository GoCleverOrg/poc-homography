"""SQLAlchemy ORM model for the Annotation entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.captured_frame import CapturedFrameModel


class AnnotationModel(Base):
    """Annotations table — links a GCP to a pixel observation in a frame.

    Domain entity: ``poc_homography.domain.entities.annotation.Annotation``

    Composite identity is (frame_id, gcp_id).  The ``id`` column stores the
    composite string ``frame_id/gcp_id`` as a surrogate PK.

    Value objects stored as JSONB:
      - camera_pose: ``{pan_raw, tilt_deg, zoom}``  (PTZState)
      - pixel: ``{x, y}``  (PixelPoint)
    """

    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("frame_id", "gcp_id", name="uq_annotation_frame_gcp"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    gcp_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    frame_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("captured_frames.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    camera_pose: Mapped[dict] = mapped_column(JSONB, nullable=False)
    pixel: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # -- relationships --
    frame: Mapped[CapturedFrameModel] = relationship(
        "CapturedFrameModel", back_populates="annotations"
    )


__all__ = ["AnnotationModel"]
