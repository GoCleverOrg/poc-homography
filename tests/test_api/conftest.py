"""Shared fixtures for FastAPI integration tests."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.deps import get_current_user, get_db_session
from api.main import app

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Mock user
# ---------------------------------------------------------------------------


def _mock_user() -> MagicMock:
    """Return a lightweight stand-in for :class:`UserModel`."""
    user = MagicMock()
    user.id = "test-user-id"
    user.username = "testuser"
    user.tenant_id = "test-tenant"
    user.is_active = True
    return user


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """FastAPI ``TestClient`` with auth and DB dependencies overridden."""
    app.dependency_overrides[get_current_user] = _mock_user
    app.dependency_overrides[get_db_session] = lambda: MagicMock()

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture()
def client_no_auth_override() -> Iterator[TestClient]:
    """``TestClient`` *without* auth override — real dependency is used.

    The mock DB session is configured so that user queries return ``None``,
    ensuring ``get_current_user`` raises a 401 instead of hitting bcrypt
    with a MagicMock.
    """

    def _empty_session() -> MagicMock:
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = None
        return session

    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides[get_db_session] = _empty_session

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_basic_auth_header(username: str, password: str) -> dict[str, str]:
    """Build an ``Authorization: Basic …`` header dict."""
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}
