"""Shared Pydantic response schemas."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class StatusResponse(BaseModel, Generic[T]):
    """Envelope for successful responses: ``{"status": "success", "data": ...}``."""

    status: str = "success"
    data: T


class ErrorResponse(BaseModel):
    """Envelope for error responses: ``{"status": "error", "error": "..."}``."""

    status: str = "error"
    error: str
