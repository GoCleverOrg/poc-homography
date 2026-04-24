"""Integration tests for authentication dependency."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.test_api.conftest import make_basic_auth_header

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


# We pick an auth-protected route for these tests.
_PROTECTED_URL = "/gcp/api/tenants/"


class TestAuthRequired:
    """Verify that protected endpoints enforce authentication."""

    def test_returns_401_without_credentials(
        self, client_no_auth_override: TestClient
    ) -> None:
        """Request without ``Authorization`` header must be rejected."""
        resp = client_no_auth_override.get(_PROTECTED_URL)
        assert resp.status_code == 401

    def test_returns_401_with_wrong_credentials(
        self, client_no_auth_override: TestClient
    ) -> None:
        """Request with invalid credentials must be rejected."""
        headers = make_basic_auth_header("wrong", "creds")
        resp = client_no_auth_override.get(_PROTECTED_URL, headers=headers)
        assert resp.status_code == 401


class TestAuthOverridden:
    """Verify that dependency override lets requests through."""

    def test_protected_endpoint_with_override(self, client: TestClient) -> None:
        """With the mock user override, protected endpoints should not 401."""
        resp = client.get(_PROTECTED_URL)
        # We don't assert 200 because the repo may not find data, but it
        # must NOT be 401 — the auth layer is bypassed.
        assert resp.status_code != 401
