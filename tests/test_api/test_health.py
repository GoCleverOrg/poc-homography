"""Integration tests for the ``GET /health`` endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    """Health endpoint should return ``{"status": "ok"}``."""
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_no_auth_required(client_no_auth_override: TestClient) -> None:
    """Health endpoint must be accessible without authentication."""
    resp = client_no_auth_override.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
