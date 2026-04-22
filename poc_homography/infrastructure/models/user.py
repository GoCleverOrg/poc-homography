"""SQLAlchemy ORM model for the User table (authentication)."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — required at runtime for Mapped[] resolution
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poc_homography.infrastructure.database import Base

if TYPE_CHECKING:
    from poc_homography.infrastructure.models.tenant import TenantModel


class UserModel(Base):
    """Users table — application users with tenant association.

    This is a pure infrastructure concern (authentication/authorisation)
    and does not correspond to a domain entity.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    tenant_id: Mapped[str] = mapped_column(
        String, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # -- relationships --
    tenant: Mapped[TenantModel] = relationship("TenantModel", back_populates="users")


__all__ = ["UserModel"]
