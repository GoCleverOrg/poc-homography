"""SQLAlchemy ORM model for the GroundControlPoint entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.map import MapModel


class GroundControlPointModel(Base):
    """Ground control points table — reference points on a georeferenced map.

    Domain entity: ``poc_homography.domain.entities.ground_control_point.GroundControlPoint``

    Composite identity is (map_id, name).  The ``id`` column stores the
    composite string ``{map_id}/{name}`` as a surrogate PK.

    Value objects stored as JSONB:
      - map_point: ``{map_id, pixel_point: {x, y}}``  (MapPoint)
    """

    __tablename__ = "ground_control_points"
    __table_args__ = (UniqueConstraint("map_id", "name", name="uq_gcp_map_name"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    map_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("maps.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    map_point: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # -- relationships --
    map: Mapped[MapModel] = relationship("MapModel", back_populates="ground_control_points")


__all__ = ["GroundControlPointModel"]
