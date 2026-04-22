"""SQLAlchemy ORM model for the Tenant entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.camera_config import CameraConfigModel
    from poc_homography.infrastructure.models.map import MapModel
    from poc_homography.infrastructure.models.user import UserModel


class TenantModel(Base):
    """Tenant table — a deployment site (e.g. a port terminal).

    Domain entity: ``poc_homography.domain.entities.tenant.Tenant``
    """

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False, default="")
    location_lat: Mapped[str] = mapped_column(String, nullable=False, default="")
    location_lon: Mapped[str] = mapped_column(String, nullable=False, default="")

    # -- relationships (back-populated by child models) --
    maps: Mapped[list[MapModel]] = relationship("MapModel", back_populates="tenant")
    camera_configs: Mapped[list[CameraConfigModel]] = relationship(
        "CameraConfigModel", back_populates="tenant"
    )
    users: Mapped[list[UserModel]] = relationship("UserModel", back_populates="tenant")


__all__ = ["TenantModel"]
