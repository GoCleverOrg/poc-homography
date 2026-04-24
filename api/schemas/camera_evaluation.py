"""Pydantic request/response schemas for camera-evaluation endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class TenantOut(BaseModel):
    """Tenant summary returned by the camera-evaluation API."""

    id: str
    name: str
    description: str = ""


class CameraOut(BaseModel):
    """Camera summary returned by the camera-evaluation API."""

    id: str
    name: str
    ip: str


# ---------------------------------------------------------------------------
# Survey request schemas
# ---------------------------------------------------------------------------


class SurveyStartRequest(BaseModel):
    """Body for ``POST /api/survey/start/``."""

    tenant_id: str
    camera_id: str
    axis: str
    start: float
    end: float
    step: float
    restore_ptz: bool = True
    retry_timeout: int = 60
    session_tags: list[str] | str = Field(default_factory=list)
    fixed_pan: float | None = None
    fixed_tilt: float | None = None
    fixed_zoom: float | None = None


# ---------------------------------------------------------------------------
# Survey response data schemas
# ---------------------------------------------------------------------------


class SurveyStartData(BaseModel):
    """``data`` field for survey start response."""

    session_id: str
    message: str


class SurveyStatusData(BaseModel):
    """``data`` field for survey status response."""

    session_id: str
    status: str
    step_count: int
    total_steps: int
    current_ptz: dict[str, float | None] | None = None
    last_capture_path: str | None = None
    error_message: str | None = None


class SurveyAbortData(BaseModel):
    """``data`` field for survey abort response."""

    session_id: str
    message: str


class SurveySessionListData(BaseModel):
    """``data`` field for survey session list response."""

    sessions: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class SurveyDeleteData(BaseModel):
    """``data`` field for survey delete response."""

    session_id: str
    message: str


# ---------------------------------------------------------------------------
# Camera PTZ response data schemas
# ---------------------------------------------------------------------------


class AxisRange(BaseModel):
    """Min/max range for a PTZ axis."""

    min: float
    max: float


class CameraCapabilitiesData(BaseModel):
    """``data.capabilities`` for camera capabilities response."""

    pan: AxisRange
    tilt: AxisRange
    zoom: AxisRange
    min_step: float = 0.1


class CameraPositionData(BaseModel):
    """``data`` field for camera position response."""

    pan: float
    tilt: float
    zoom: float
