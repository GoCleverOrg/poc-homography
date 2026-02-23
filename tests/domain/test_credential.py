"""Tests for Credential value object — repr masking and frozen (Security #8)."""

from __future__ import annotations

import pytest

from poc_homography.domain.vo.credential import Credential


@pytest.fixture
def cred() -> Credential:
    return Credential(username="admin", password="s3cret!")


class TestCredentialReprMasking:
    def test_repr_hides_password(self, cred: Credential) -> None:
        assert "s3cret!" not in repr(cred)

    def test_repr_shows_mask(self, cred: Credential) -> None:
        assert "***" in repr(cred)

    def test_repr_shows_username(self, cred: Credential) -> None:
        assert "admin" in repr(cred)

    def test_str_hides_password(self, cred: Credential) -> None:
        # str() falls back to __repr__ for dataclasses with custom __repr__
        assert "s3cret!" not in str(cred)

    def test_password_field_still_accessible(self, cred: Credential) -> None:
        assert cred.password == "s3cret!"


class TestCredentialFrozen:
    def test_mutate_password_raises(self, cred: Credential) -> None:
        with pytest.raises(AttributeError):
            cred.password = "new"  # type: ignore[misc]
