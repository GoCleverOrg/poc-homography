"""SQLAlchemy ORM model for the Line entity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.map import MapModel


class LineModel(Base):
    """Lines table — a line on a map image defined by two pixel endpoints.

    Domain entity: ``poc_homography.domain.entities.line.Line``

    Composite identity is (map_id, name).  The ``id`` column stores the
    composite string ``{map_id}/{name}`` as a surrogate PK.

    Value objects stored as JSONB:
      - start: ``{x, y}``  (PixelPoint)
      - end: ``{x, y}``  (PixelPoint)
    """

    __tablename__ = "lines"
    __table_args__ = (UniqueConstraint("map_id", "name", name="uq_line_map_name"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    map_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("maps.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    start: Mapped[dict] = mapped_column(JSONB, nullable=False)
    end: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # -- relationships --
    map: Mapped[MapModel] = relationship("MapModel", back_populates="lines")


__all__ = ["LineModel"]
