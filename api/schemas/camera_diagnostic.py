"""Pydantic request/response schemas for camera-diagnostic endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class TenantOut(BaseModel):
    """Tenant summary returned by the diagnostic API."""

    id: str
    name: str
    description: str | None = None


class CameraOut(BaseModel):
    """Camera summary returned by the diagnostic API."""

    id: str
    name: str
    ip: str | None = None
    tenant_id: str | None = None


# ---------------------------------------------------------------------------
# Diagnostic test result schemas
# ---------------------------------------------------------------------------


class DiagnosticTestResultIn(BaseModel):
    """A single diagnostic test result submitted by the client."""

    status: str = "pending"
    response_time_ms: float | None = None
    error_message: str | None = None
    error_category: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class CameraResultIn(BaseModel):
    """Camera diagnostic result submitted by the client."""

    camera_id: str
    camera_name: str
    camera_ip: str = ""
    device_info: dict[str, Any] | None = None
    rtsp_test: DiagnosticTestResultIn | None = None
    webui_test: DiagnosticTestResultIn | None = None
    ptz_test: DiagnosticTestResultIn | None = None


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class RunDiagnosticRequest(BaseModel):
    """Body for ``POST /api/diagnostic/run/``."""

    tenant_id: str
    camera_ids: list[str] | None = None


class SaveDiagnosticRequest(BaseModel):
    """Body for ``POST /api/diagnostic/save/``."""

    tenant_id: str
    camera_results: list[CameraResultIn]


class StressTestStartRequest(BaseModel):
    """Body for ``POST /api/stress-test/start/``."""

    tenant_id: str
    camera_id: str
    preset_name: str | None = None
    test_type: str | None = None
    pan_config: dict[str, Any] | None = None
    tilt_config: dict[str, Any] | None = None
    zoom_config: dict[str, Any] | None = None
    repetitions: int = 1
    max_speed: bool = False


class StressTestEvaluateRequest(BaseModel):
    """Body for ``POST /api/stress-test/sessions/{session_id}/evaluate/``."""

    evaluation: str
    notes: str = ""
