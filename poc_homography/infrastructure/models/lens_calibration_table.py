"""SQLAlchemy ORM model for the LensCalibrationTable entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.camera_config import CameraConfigModel


class LensCalibrationTableModel(Base):
    """Lens calibration tables — zoom-indexed distortion data per camera.

    Domain entity: ``poc_homography.domain.entities.lens_calibration_table.LensCalibrationTable``

    The ``id`` references ``camera_configs.id``.

    Value objects stored as JSONB:
      - entries: list of ZoomCalibrationEntry dicts
        ``[{zoom_factor, distortion: {k1,k2,p1,p2,k3}, calibration_date, ...}, ...]``
    """

    __tablename__ = "lens_calibration_tables"

    id: Mapped[str] = mapped_column(
        String,
        ForeignKey("camera_configs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    entries: Mapped[list] = mapped_column(JSONB, nullable=False)
    created_date: Mapped[str] = mapped_column(String, nullable=False, default="")
    last_modified: Mapped[str] = mapped_column(String, nullable=False, default="")

    # -- relationships --
    camera_config: Mapped[CameraConfigModel] = relationship(
        "CameraConfigModel", back_populates="lens_calibration_table"
    )


__all__ = ["LensCalibrationTableModel"]
