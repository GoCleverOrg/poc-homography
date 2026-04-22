"""SQLAlchemy ORM model for the CalibrationLineTraceSet entity."""

from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from poc_homography.infrastructure.database import Base


class CalibrationLineTraceSetModel(Base):
    """Calibration line trace sets — N-point line traces from a camera frame.

    Domain entity:
        ``poc_homography.domain.entities.calibration_line_trace_set.CalibrationLineTraceSet``

    Value objects stored as JSONB:
      - camera_pose: ``{pan_raw, tilt_deg, zoom}``  (PTZState)
      - line_traces: ``[{line_id, points: [[x,y], ...]}, ...]``  (list of LineTrace dicts)
    """

    __tablename__ = "calibration_line_trace_sets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    image: Mapped[str] = mapped_column(String, nullable=False)
    camera_pose: Mapped[dict] = mapped_column(JSONB, nullable=False)
    line_traces: Mapped[list] = mapped_column(JSONB, nullable=False)


__all__ = ["CalibrationLineTraceSetModel"]
