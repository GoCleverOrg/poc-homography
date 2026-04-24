"""Integration tests verifying CORS headers on responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


_ALLOWED_ORIGIN = "http://localhost:3000"


def test_cors_allows_configured_origin(client: TestClient) -> None:
    """Responses to requests from an allowed origin must include CORS headers."""
    resp = client.get("/health", headers={"Origin": _ALLOWED_ORIGIN})

    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == _ALLOWED_ORIGIN


def test_cors_preflight(client: TestClient) -> None:
    """An OPTIONS preflight request must succeed with CORS headers."""
    resp = client.options(
        "/health",
        headers={
            "Origin": _ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == _ALLOWED_ORIGIN
    assert "GET" in resp.headers.get("access-control-allow-methods", "")


def test_cors_disallows_unknown_origin(client: TestClient) -> None:
    """Requests from an unlisted origin must not receive a permissive CORS header."""
    resp = client.get("/health", headers={"Origin": "http://evil.example.com"})

    assert resp.status_code == 200
    # The origin should NOT be reflected back.
    assert resp.headers.get("access-control-allow-origin") != "http://evil.example.com"
