"""SQLAlchemy ORM model for the CapturedFrame entity."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — required at runtime for Mapped[] resolution
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.annotation import AnnotationModel
    from poc_homography.infrastructure.models.line_annotation import LineAnnotationModel
    from poc_homography.infrastructure.models.map import MapModel


class CapturedFrameModel(Base):
    """Captured frames table — a photo with camera PTZ state.

    Domain entity: ``poc_homography.domain.entities.captured_frame.CapturedFrame``

    ID format: ``{map_id}/{camera_name}/{timestamp}`` (3-part composite string).
    A unique constraint on (map_id, camera_name, timestamp) enforces uniqueness.

    Value objects stored as JSONB:
      - ptz_state: ``{pan_raw, tilt_deg, zoom}``
    """

    __tablename__ = "captured_frames"
    __table_args__ = (
        UniqueConstraint("map_id", "camera_name", "timestamp", name="uq_captured_frame_composite"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    map_id: Mapped[str] = mapped_column(
        String, ForeignKey("maps.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ptz_state: Mapped[dict] = mapped_column(JSONB, nullable=False)
    image_path: Mapped[str] = mapped_column(String, nullable=False)

    # -- relationships --
    map: Mapped[MapModel] = relationship("MapModel", back_populates="captured_frames")
    annotations: Mapped[list[AnnotationModel]] = relationship(
        "AnnotationModel", back_populates="frame"
    )
    line_annotations: Mapped[list[LineAnnotationModel]] = relationship(
        "LineAnnotationModel", back_populates="frame"
    )


__all__ = ["CapturedFrameModel"]
