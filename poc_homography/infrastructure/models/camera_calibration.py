"""SQLAlchemy ORM model for the CameraCalibration entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.camera_config import CameraConfigModel


class CameraCalibrationModel(Base):
    """Camera calibrations table — calibration data refined during calibration.

    Domain entity: ``poc_homography.domain.entities.camera_calibration.CameraCalibration``

    The ``id`` references ``camera_configs.id`` (format: ``map_id/camera_name``).

    Value objects stored as JSONB:
      - position: ``{x, y}``  (PixelPoint)
      - base_orientation: ``{yaw, pitch, roll}``  (Orientation)
      - distortion: ``{k1, k2, p1, p2, k3}``  (LensDistortion)
    """

    __tablename__ = "camera_calibrations"

    id: Mapped[str] = mapped_column(
        String,
        ForeignKey("camera_configs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    position: Mapped[dict] = mapped_column(JSONB, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    base_orientation: Mapped[dict] = mapped_column(JSONB, nullable=False)
    distortion: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # -- relationships --
    camera_config: Mapped[CameraConfigModel] = relationship(
        "CameraConfigModel", back_populates="camera_calibration"
    )


__all__ = ["CameraCalibrationModel"]
