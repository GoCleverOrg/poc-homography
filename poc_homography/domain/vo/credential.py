"""Credential value object for camera authentication."""

from dataclasses import dataclass


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
