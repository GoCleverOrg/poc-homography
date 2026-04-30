"""Shared FastAPI dependencies (auth, tenant, DB session)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import bcrypt
from fastapi import Depends, HTTPException, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session

from poc_homography.infrastructure.database import get_sessionmaker
from poc_homography.infrastructure.models.user import UserModel
from poc_homography.infrastructure.repositories import RepoPostgresMap, RepoPostgresTenant

if TYPE_CHECKING:
    from collections.abc import Generator

    from poc_homography.domain.entities.map import Map

# ---------------------------------------------------------------------------
# DB session
# ---------------------------------------------------------------------------

_http_basic = HTTPBasic()


def get_db_session() -> Generator[Session, None, None]:
    """Yield an SQLAlchemy session, committing on success or rolling back on error."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def get_current_user(
    credentials: HTTPBasicCredentials = Depends(_http_basic),
    session: Session = Depends(get_db_session),
) -> UserModel:
    """Authenticate via HTTP Basic and return the active user."""
    user = (
        session.query(UserModel)
        .filter(UserModel.username == credentials.username, UserModel.is_active.is_(True))
        .first()
    )

    if user is None or not bcrypt.checkpw(
        credentials.password.encode("utf-8"),
        user.hashed_password.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return user


# ---------------------------------------------------------------------------
# Tenant resolution
# ---------------------------------------------------------------------------


def get_tenant_id(
    tenant_id: str = Query(...),
    session: Session = Depends(get_db_session),
) -> str:
    """Extract and validate the ``tenant_id`` query parameter."""
    repo = RepoPostgresTenant(session)
    if not repo.exists(tenant_id):
        raise HTTPException(status_code=400, detail=f"Unknown tenant_id: {tenant_id}")
    return tenant_id


def get_map_for_tenant(
    tenant_id: str = Depends(get_tenant_id),
    session: Session = Depends(get_db_session),
) -> Map:
    """Resolve the :class:`Map` entity associated with *tenant_id*."""
    repo = RepoPostgresMap(session)
    maps = repo.get_by_tenant(tenant_id)
    if not maps:
        raise HTTPException(status_code=404, detail=f"No map found for tenant: {tenant_id}")
    # Return the first (and typically only) map for this tenant.
    return next(iter(maps.values()))
