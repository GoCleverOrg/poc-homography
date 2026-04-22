"""SQLAlchemy ORM model for the Map entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.camera_config import CameraConfigModel
    from poc_homography.infrastructure.models.captured_frame import CapturedFrameModel
    from poc_homography.infrastructure.models.ground_control_point import (
        GroundControlPointModel,
    )
    from poc_homography.infrastructure.models.line import LineModel
    from poc_homography.infrastructure.models.tenant import TenantModel


class MapModel(Base):
    """Maps table — a georeferenced map image.

    Domain entity: ``poc_homography.domain.entities.map.Map``

    Value objects stored as JSONB:
      - photo: ``{path, width, height}``
      - geotiff: ``{geotransform: {origin_easting, pixel_width, ...}, crs}``
    """

    __tablename__ = "maps"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Value objects serialised as JSONB
    photo: Mapped[dict] = mapped_column(JSONB, nullable=False)
    geotiff: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # -- relationships --
    tenant: Mapped[TenantModel] = relationship("TenantModel", back_populates="maps")
    camera_configs: Mapped[list[CameraConfigModel]] = relationship(
        "CameraConfigModel", back_populates="map"
    )
    captured_frames: Mapped[list[CapturedFrameModel]] = relationship(
        "CapturedFrameModel", back_populates="map"
    )
    lines: Mapped[list[LineModel]] = relationship("LineModel", back_populates="map")
    ground_control_points: Mapped[list[GroundControlPointModel]] = relationship(
        "GroundControlPointModel", back_populates="map"
    )


__all__ = ["MapModel"]
