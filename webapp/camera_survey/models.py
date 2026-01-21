"""Data models for camera survey operations.

These are dataclasses for type safety and validation, not Django ORM models
since data persists to filesystem, not database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SurveyAxis(str, Enum):
    """Axis for survey sweep."""

    PAN = "pan"
    TILT = "tilt"
    ZOOM = "zoom"


class SurveyStatus(str, Enum):
    """Survey execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class PTZPosition:
    """PTZ position values."""

    pan: float | None = None
    tilt: float | None = None
    zoom: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert to dictionary."""
        return {"pan": self.pan, "tilt": self.tilt, "zoom": self.zoom}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PTZPosition:
        """Create from dictionary."""
        return cls(
            pan=data.get("pan"),
            tilt=data.get("tilt"),
            zoom=data.get("zoom"),
        )


@dataclass
class CameraCapabilities:
    """Camera PTZ capabilities for validation."""

    pan_min: float = 0.0
    pan_max: float = 360.0
    tilt_min: float = -15.0
    tilt_max: float = 90.0
    zoom_min: float = 1.0
    zoom_max: float = 25.0

    def validate_range(self, axis: SurveyAxis, start: float, end: float) -> tuple[bool, str | None]:
        """Validate that start/end values are within valid range for axis.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if axis == SurveyAxis.PAN:
            min_val, max_val = self.pan_min, self.pan_max
        elif axis == SurveyAxis.TILT:
            min_val, max_val = self.tilt_min, self.tilt_max
        else:  # ZOOM
            min_val, max_val = self.zoom_min, self.zoom_max

        # For pan, values can wrap around (e.g., 350 to 10)
        if axis == SurveyAxis.PAN:
            if not (min_val <= start <= max_val):
                return False, f"Start value {start} outside valid pan range [{min_val}, {max_val}]"
            if not (min_val <= end <= max_val):
                return False, f"End value {end} outside valid pan range [{min_val}, {max_val}]"
        else:
            if not (min_val <= start <= max_val):
                return False, f"Start value {start} outside valid {axis.value} range [{min_val}, {max_val}]"
            if not (min_val <= end <= max_val):
                return False, f"End value {end} outside valid {axis.value} range [{min_val}, {max_val}]"

        return True, None

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Convert to dictionary."""
        return {
            "pan": {"min": self.pan_min, "max": self.pan_max},
            "tilt": {"min": self.tilt_min, "max": self.tilt_max},
            "zoom": {"min": self.zoom_min, "max": self.zoom_max},
        }


@dataclass
class SurveyConfig:
    """Configuration for a survey sweep."""

    tenant_id: str
    camera_id: str
    axis: SurveyAxis
    start: float
    end: float
    step: float
    restore_ptz: bool = True
    retry_timeout: int = 60
    session_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "camera_id": self.camera_id,
            "axis": self.axis.value,
            "start": self.start,
            "end": self.end,
            "step": self.step,
            "restore_ptz": self.restore_ptz,
            "retry_timeout": self.retry_timeout,
            "session_tags": self.session_tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurveyConfig:
        """Create from dictionary."""
        return cls(
            tenant_id=data["tenant_id"],
            camera_id=data["camera_id"],
            axis=SurveyAxis(data["axis"]),
            start=float(data["start"]),
            end=float(data["end"]),
            step=float(data["step"]),
            restore_ptz=data.get("restore_ptz", True),
            retry_timeout=int(data.get("retry_timeout", 60)),
            session_tags=data.get("session_tags", []),
        )

    def generate_steps(self) -> list[float]:
        """Generate list of axis values for the survey.

        Handles wraparound for pan axis (e.g., 350 to 10 degrees).
        """
        steps = []
        current = self.start

        if self.axis == SurveyAxis.PAN and self.end < self.start:
            # Wraparound case: e.g., 350 to 10 degrees
            # First go from start to 360
            while current <= 360.0:
                steps.append(round(current, 1))
                current += self.step
            # Then continue from 0 to end
            current = 0.0
            while current <= self.end:
                steps.append(round(current, 1))
                current += self.step
        else:
            # Normal case: start to end
            while current <= self.end:
                steps.append(round(current, 1))
                current += self.step

        return steps


@dataclass
class CaptureRecord:
    """Record of a single capture during survey."""

    filename: str
    timestamp: str  # ISO 8601 format
    ptz: PTZPosition
    step_index: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "filename": self.filename,
            "timestamp": self.timestamp,
            "ptz": self.ptz.to_dict(),
            "step_index": self.step_index,
        }
        if self.errors:
            result["errors"] = self.errors
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureRecord:
        """Create from dictionary."""
        return cls(
            filename=data["filename"],
            timestamp=data["timestamp"],
            ptz=PTZPosition.from_dict(data["ptz"]),
            step_index=data["step_index"],
            errors=data.get("errors", []),
        )


@dataclass
class TenantInfo:
    """Tenant information for manifest."""

    id: str
    name: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {"id": self.id, "name": self.name}


@dataclass
class CameraInfo:
    """Camera information for manifest."""

    id: str
    name: str
    ip: str
    model: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary."""
        return {"id": self.id, "name": self.name, "ip": self.ip, "model": self.model}


@dataclass
class CaptureMetadata:
    """Metadata about capture settings."""

    resolution: str  # e.g., "2560x1440"
    codec: str = "h264"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {"resolution": self.resolution, "codec": self.codec}


@dataclass
class SurveySession:
    """Complete survey session data."""

    id: str  # UUID string
    start_time: datetime
    end_time: datetime | None = None
    status: SurveyStatus = SurveyStatus.PENDING
    abort_reason: str | None = None

    # Context info
    tenant: TenantInfo | None = None
    camera: CameraInfo | None = None
    capture_metadata: CaptureMetadata | None = None

    # Survey configuration
    survey_parameters: SurveyConfig | None = None

    # PTZ positions
    initial_ptz: PTZPosition | None = None
    final_ptz: PTZPosition | None = None

    # Tags and captures
    session_tags: list[str] = field(default_factory=list)
    captures: list[CaptureRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML manifest."""
        result: dict[str, Any] = {
            "session": {
                "id": self.id,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "status": self.status.value,
            }
        }

        if self.abort_reason:
            result["session"]["abort_reason"] = self.abort_reason

        if self.tenant:
            result["tenant"] = self.tenant.to_dict()

        if self.camera:
            result["camera"] = self.camera.to_dict()

        if self.capture_metadata:
            result["capture_metadata"] = self.capture_metadata.to_dict()

        if self.survey_parameters:
            result["survey_parameters"] = {
                "axis": self.survey_parameters.axis.value,
                "start": self.survey_parameters.start,
                "end": self.survey_parameters.end,
                "step": self.survey_parameters.step,
                "restore_ptz": self.survey_parameters.restore_ptz,
                "retry_timeout": self.survey_parameters.retry_timeout,
            }

        if self.initial_ptz:
            result["initial_ptz"] = self.initial_ptz.to_dict()

        if self.final_ptz:
            result["final_ptz"] = self.final_ptz.to_dict()

        result["session_tags"] = self.session_tags
        result["captures"] = [c.to_dict() for c in self.captures]

        return result


@dataclass
class SurveyProgress:
    """Current survey progress for status endpoint."""

    session_id: str
    status: SurveyStatus
    step_count: int
    total_steps: int
    current_ptz: PTZPosition | None = None
    last_capture_path: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "step_count": self.step_count,
            "total_steps": self.total_steps,
            "current_ptz": self.current_ptz.to_dict() if self.current_ptz else None,
            "last_capture_path": self.last_capture_path,
            "error_message": self.error_message,
        }


@dataclass
class SurveyPreset:
    """Preset survey configuration."""

    name: str
    axis: SurveyAxis
    start: float
    end: float
    step: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "axis": self.axis.value,
            "start": self.start,
            "end": self.end,
            "step": self.step,
        }


# Default survey presets
SURVEY_PRESETS = [
    SurveyPreset(name="360° Pan (10° step)", axis=SurveyAxis.PAN, start=0, end=360, step=10),
    SurveyPreset(name="360° Pan (5° step)", axis=SurveyAxis.PAN, start=0, end=360, step=5),
    SurveyPreset(name="180° Pan Sweep", axis=SurveyAxis.PAN, start=90, end=270, step=10),
    SurveyPreset(name="Tilt Sweep (full)", axis=SurveyAxis.TILT, start=-15, end=90, step=5),
    SurveyPreset(name="Tilt Sweep (±45°)", axis=SurveyAxis.TILT, start=-45, end=45, step=5),
    SurveyPreset(name="Full Zoom Range", axis=SurveyAxis.ZOOM, start=1, end=25, step=1),
    SurveyPreset(name="Zoom Range (1-10x)", axis=SurveyAxis.ZOOM, start=1, end=10, step=1),
]
