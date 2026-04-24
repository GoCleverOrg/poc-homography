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


class ResolutionOut(BaseModel):
    """Video resolution."""

    width: int
    height: int


class RtspMetrics(BaseModel):
    """RTSP connection metrics."""

    fps: float | None = None
    resolution: ResolutionOut
    latency_ms: float


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


# ---------------------------------------------------------------------------
# Response data schemas (nested inside the status envelope)
# ---------------------------------------------------------------------------


class TenantListData(BaseModel):
    """``data`` field for tenant list response."""

    tenants: list[TenantOut]


class CameraListData(BaseModel):
    """``data`` field for camera list response."""

    cameras: list[CameraOut]


class RtspTestData(BaseModel):
    """``data`` field for RTSP test response."""

    message: str
    metrics: RtspMetrics


class WebUiTestData(BaseModel):
    """``data`` field for WebUI test response."""

    message: str
    login_success: bool
    login_attempted: bool
    ptz_controls_found: Any  # bool or list depending on context
    screenshot_path: str | None = None
    login_error: str | None = None
    ptz_test_result: dict[str, Any] | None = None


class PtzTestData(BaseModel):
    """``data`` field for PTZ test response."""

    camera_name: str
    camera_ip: str
    initial_position: dict[str, Any] | None = None
    position_restored: bool = False
    tests: dict[str, Any] = Field(default_factory=dict)
    restore_error: str | None = None


class DiagnosticRunData(BaseModel):
    """``data`` field for run-diagnostic response."""

    session_id: str
    status: str
    summary: dict[str, int]
    camera_results: list[dict[str, Any]]


class DiagnosticSaveData(BaseModel):
    """``data`` field for save-diagnostic response."""

    session_id: str
    status: str
    summary: dict[str, int]


class SessionSummaryOut(BaseModel):
    """Single session in a list response."""

    id: str
    created_at: str
    completed_at: str | None = None
    status: str
    tenant_id: str
    tenant_name: str
    summary: dict[str, int]


class DiagnosticSessionListData(BaseModel):
    """``data`` field for diagnostic session list response."""

    sessions: list[SessionSummaryOut]
    total: int
    limit: int
    offset: int


class StressTestSessionSummaryOut(BaseModel):
    """Single stress test session in a list response."""

    id: str
    created_at: str
    completed_at: str | None = None
    status: str
    tenant_id: str
    camera_id: str
    camera_name: str
    test_type: str | None = None
    user_evaluation: str
    result_success: bool | None = None


class StressTestSessionListData(BaseModel):
    """``data`` field for stress test session list response."""

    sessions: list[StressTestSessionSummaryOut]
    total: int
    limit: int
    offset: int


class StressTestStartData(BaseModel):
    """``data`` field for stress test start response."""

    session_id: str
    message: str


class StressTestStatusData(BaseModel):
    """``data`` field for stress test status response."""

    session_id: str
    status: str
    current_repetition: int = 0
    total_repetitions: int = 0
    current_movement: int | None = None
    total_movements: int | None = None
    current_position: dict[str, float] | None = None
    message: str = ""


class MessageData(BaseModel):
    """Generic ``data`` field with a single message."""

    message: str


class EvaluationSavedData(BaseModel):
    """``data`` field for evaluate response."""

    message: str
    session_id: str
    evaluation: str


# ---------------------------------------------------------------------------
# Top-level envelope schemas
# ---------------------------------------------------------------------------


class SuccessEnvelope(BaseModel):
    """Standard ``{"status": "success", "data": ...}`` response envelope."""

    status: str = "success"
    data: Any


class ErrorEnvelope(BaseModel):
    """Standard ``{"status": "error", ...}`` response envelope."""

    status: str = "error"
    error_category: str
    message: str
