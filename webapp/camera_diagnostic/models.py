"""Data models for the Camera Diagnostic Tool.

This module contains dataclasses for diagnostic session storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from camera_survey.models import DeviceInfo


class DiagnosticTestStatus(str, Enum):
    """Status of a diagnostic test."""

    PENDING = "pending"
    RUNNING = "running"
    PASS = "pass"
    FAIL = "fail"


class DiagnosticSessionStatus(str, Enum):
    """Status of a diagnostic session."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class DiagnosticTestResult:
    """Result of a single diagnostic test (RTSP, WebUI, or PTZ)."""

    test_type: str  # "rtsp", "webui", "ptz"
    status: DiagnosticTestStatus = DiagnosticTestStatus.PENDING
    response_time_ms: float | None = None
    error_message: str | None = None
    error_category: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_type": self.test_type,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "error_category": self.error_category,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticTestResult:
        """Create from dictionary."""
        return cls(
            test_type=data["test_type"],
            status=DiagnosticTestStatus(data["status"]),
            response_time_ms=data.get("response_time_ms"),
            error_message=data.get("error_message"),
            error_category=data.get("error_category"),
            details=data.get("details", {}),
        )


@dataclass
class CameraDiagnosticResult:
    """Diagnostic results for a single camera."""

    camera_id: str
    camera_name: str
    camera_ip: str
    device_info: DeviceInfo | None = None
    rtsp_test: DiagnosticTestResult | None = None
    webui_test: DiagnosticTestResult | None = None
    ptz_test: DiagnosticTestResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "camera_ip": self.camera_ip,
            "device_info": self.device_info.to_dict() if self.device_info else None,
            "rtsp_test": self.rtsp_test.to_dict() if self.rtsp_test else None,
            "webui_test": self.webui_test.to_dict() if self.webui_test else None,
            "ptz_test": self.ptz_test.to_dict() if self.ptz_test else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraDiagnosticResult:
        """Create from dictionary."""
        return cls(
            camera_id=data["camera_id"],
            camera_name=data["camera_name"],
            camera_ip=data["camera_ip"],
            device_info=DeviceInfo.from_dict(data.get("device_info"))
            if data.get("device_info")
            else None,
            rtsp_test=DiagnosticTestResult.from_dict(data["rtsp_test"])
            if data.get("rtsp_test")
            else None,
            webui_test=DiagnosticTestResult.from_dict(data["webui_test"])
            if data.get("webui_test")
            else None,
            ptz_test=DiagnosticTestResult.from_dict(data["ptz_test"])
            if data.get("ptz_test")
            else None,
        )

    def get_overall_status(self) -> DiagnosticTestStatus:
        """Get overall status based on individual test results."""
        tests = [self.rtsp_test, self.webui_test, self.ptz_test]
        tests = [t for t in tests if t is not None]

        if not tests:
            return DiagnosticTestStatus.PENDING

        if any(t.status == DiagnosticTestStatus.RUNNING for t in tests):
            return DiagnosticTestStatus.RUNNING

        if any(t.status == DiagnosticTestStatus.FAIL for t in tests):
            return DiagnosticTestStatus.FAIL

        if all(t.status == DiagnosticTestStatus.PASS for t in tests):
            return DiagnosticTestStatus.PASS

        return DiagnosticTestStatus.PENDING


@dataclass
class DiagnosticSession:
    """A complete diagnostic session with all camera test results."""

    id: str  # UUID
    created_at: datetime
    completed_at: datetime | None = None
    status: DiagnosticSessionStatus = DiagnosticSessionStatus.PENDING
    tenant_id: str = ""
    tenant_name: str = ""
    camera_results: list[CameraDiagnosticResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML manifest."""
        return {
            "session": {
                "id": self.id,
                "created_at": self.created_at.isoformat(),
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "status": self.status.value,
            },
            "tenant": {
                "id": self.tenant_id,
                "name": self.tenant_name,
            },
            "summary": self.get_summary(),
            "camera_results": [r.to_dict() for r in self.camera_results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticSession:
        """Create from dictionary."""
        session_data = data.get("session", {})
        tenant_data = data.get("tenant", {})

        return cls(
            id=session_data.get("id", ""),
            created_at=datetime.fromisoformat(session_data["created_at"])
            if session_data.get("created_at")
            else datetime.now(),
            completed_at=datetime.fromisoformat(session_data["completed_at"])
            if session_data.get("completed_at")
            else None,
            status=DiagnosticSessionStatus(session_data.get("status", "completed")),
            tenant_id=tenant_data.get("id", ""),
            tenant_name=tenant_data.get("name", ""),
            camera_results=[
                CameraDiagnosticResult.from_dict(r) for r in data.get("camera_results", [])
            ],
        )

    def get_summary(self) -> dict[str, int]:
        """Get summary statistics."""
        passed = 0
        failed = 0
        pending = 0

        for result in self.camera_results:
            for test in [result.rtsp_test, result.webui_test, result.ptz_test]:
                if test is None:
                    continue
                if test.status == DiagnosticTestStatus.PASS:
                    passed += 1
                elif test.status == DiagnosticTestStatus.FAIL:
                    failed += 1
                else:
                    pending += 1

        return {
            "total_cameras": len(self.camera_results),
            "tests_passed": passed,
            "tests_failed": failed,
            "tests_pending": pending,
        }
