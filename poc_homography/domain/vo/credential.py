"""Credential value object for camera authentication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Credential:
    """Authentication credentials for camera access.

    Immutable value object containing username and password
    for authenticating with camera hardware APIs.

    Attributes:
        username: Authentication username.
        password: Authentication password.
    """

    username: str
    password: str

    def __repr__(self) -> str:
        return f"Credential(username={self.username!r}, password='***')"

    def to_dict(self, *, include_secret: bool = False) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Args:
            include_secret: If True, include the plaintext password.
                Defaults to False (password redacted as ``'***'``).
        """
        return {
            "username": self.username,
            "password": self.password if include_secret else "***",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Credential:
        """Create Credential from dictionary."""
        return cls(
            username=data["username"],
            password=data["password"],
        )
