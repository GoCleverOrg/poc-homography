"""SQLAlchemy ORM model for the CameraConfig entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.camera_calibration import CameraCalibrationModel
    from poc_homography.infrastructure.models.lens_calibration_table import (
        LensCalibrationTableModel,
    )
    from poc_homography.infrastructure.models.map import MapModel
    from poc_homography.infrastructure.models.tenant import TenantModel


class CameraConfigModel(Base):
    """Camera configs table — static camera registration data.

    Domain entity: ``poc_homography.domain.entities.camera_config.CameraConfig``

    Value objects stored as JSONB:
      - credential: ``{username, password}``
    """

    __tablename__ = "camera_configs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    map_id: Mapped[str] = mapped_column(
        String, ForeignKey("maps.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    spec: Mapped[str] = mapped_column(String, nullable=False)
    credential: Mapped[dict] = mapped_column(JSONB, nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String, nullable=True)

    # -- relationships --
    tenant: Mapped[TenantModel] = relationship("TenantModel", back_populates="camera_configs")
    map: Mapped[MapModel] = relationship("MapModel", back_populates="camera_configs")
    camera_calibration: Mapped[CameraCalibrationModel | None] = relationship(
        "CameraCalibrationModel", back_populates="camera_config", uselist=False
    )
    lens_calibration_table: Mapped[LensCalibrationTableModel | None] = relationship(
        "LensCalibrationTableModel", back_populates="camera_config", uselist=False
    )


__all__ = ["CameraConfigModel"]
