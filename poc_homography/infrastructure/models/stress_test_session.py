"""SQLAlchemy ORM model for StressTestSession (webapp-layer entity)."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — required at runtime for Mapped[] resolution

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from poc_homography.infrastructure.database import Base


class StressTestSessionModel(Base):
    """Stress test sessions table.

    Stores the full session manifest as JSONB with indexed columns for
    tenant_id, camera_id, and created_date to support filtering/pagination.

    The session entity lives in the webapp layer (not the domain layer),
    so this model stores the raw dict and relies on the repository layer
    for domain conversion.
    """

    __tablename__ = "stress_test_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)


__all__ = ["StressTestSessionModel"]
