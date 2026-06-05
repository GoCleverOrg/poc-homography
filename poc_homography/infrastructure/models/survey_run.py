"""SQLAlchemy ORM model for :class:`SurveyRun`."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — required at runtime for Mapped[] resolution

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from poc_homography.infrastructure.database import Base


class SurveyRunModel(Base):
    """Survey runs table.

    Stores the full :meth:`SurveyRun.to_dict` blob as JSONB with indexed
    columns for ``camera_id``, ``status``, and ``created_date`` to support
    filtering and pagination. Added additively alongside the untouched
    ``survey_sessions`` table.
    """

    __tablename__ = "survey_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)


__all__ = ["SurveyRunModel"]
