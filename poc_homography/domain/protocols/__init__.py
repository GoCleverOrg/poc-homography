"""Protocols for external service integrations.

This module defines interfaces (protocols) for external services that the
domain and application layers depend on. Infrastructure implementations
provide concrete implementations of these protocols.
"""

from poc_homography.domain.protocols.sam3_client import (
    Sam3ApiError,
    Sam3Client,
    Sam3Detection,
)

__all__ = [
    "Sam3ApiError",
    "Sam3Client",
    "Sam3Detection",
]
