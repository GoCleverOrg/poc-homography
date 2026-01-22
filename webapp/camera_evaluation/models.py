"""Data models for the Camera Evaluation Tool.

This module contains dataclasses and enums for stress testing functionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StressTestType(Enum):
    """Types of stress tests available."""

    OSCILLATION = "oscillation"
    RANDOM_STEP_ACCURACY = "random_step_accuracy"
    FULL_RANGE_SWEEP = "full_range_sweep"
    TILT_STRESS = "tilt_stress"
    COMBINED_AXIS_LOAD = "combined_axis_load"
    POSITION_REPEATABILITY = "position_repeatability"
    SPEED_TEST = "speed_test"


class StressTestStatus(Enum):
    """Status of a stress test session."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class UserEvaluation(Enum):
    """User evaluation of stress test results."""

    GOOD = "good"  # Green - Camera performed well
    NEEDS_IMPROVEMENT = "needs_improvement"  # Yellow - Some issues observed
    BAD = "bad"  # Red - Camera has significant problems
    NOT_EVALUATED = "not_evaluated"  # Not yet evaluated by user


@dataclass
class AxisMovementConfig:
    """Configuration for a single axis movement in a stress test."""

    axis: str  # "pan", "tilt", or "zoom"
    start: float  # Starting position in degrees
    end: float  # Ending position in degrees
    step: float = 10.0  # Fixed step size in degrees
    step_min: float = 5.0  # Minimum random step size
    step_max: float = 15.0  # Maximum random step size
    use_random_steps: bool = False  # Whether to use random step sizes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "axis": self.axis,
            "start": self.start,
            "end": self.end,
            "step": self.step,
            "step_min": self.step_min,
            "step_max": self.step_max,
            "use_random_steps": self.use_random_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AxisMovementConfig:
        """Create from dictionary."""
        return cls(
            axis=data["axis"],
            start=data["start"],
            end=data["end"],
            step=data.get("step", 10.0),
            step_min=data.get("step_min", 5.0),
            step_max=data.get("step_max", 15.0),
            use_random_steps=data.get("use_random_steps", False),
        )


@dataclass
class MovementTiming:
    """Timing data for a single movement operation."""

    command_sent: datetime
    stabilized: datetime
    duration_ms: float
    start_position: dict[str, float]  # {"pan": x, "tilt": y, "zoom": z}
    end_position: dict[str, float]
    target_position: dict[str, float]
    position_error: dict[str, float]  # Difference between target and actual

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "command_sent": self.command_sent.isoformat(),
            "stabilized": self.stabilized.isoformat(),
            "duration_ms": self.duration_ms,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "target_position": self.target_position,
            "position_error": self.position_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MovementTiming:
        """Create from dictionary."""
        return cls(
            command_sent=datetime.fromisoformat(data["command_sent"]),
            stabilized=datetime.fromisoformat(data["stabilized"]),
            duration_ms=data["duration_ms"],
            start_position=data["start_position"],
            end_position=data["end_position"],
            target_position=data["target_position"],
            position_error=data["position_error"],
        )


@dataclass
class StressTestConfig:
    """Configuration for a stress test session."""

    tenant_id: str
    camera_id: str
    test_type: StressTestType
    pan_config: AxisMovementConfig | None = None
    tilt_config: AxisMovementConfig | None = None
    zoom_config: AxisMovementConfig | None = None
    repetitions: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tenant_id": self.tenant_id,
            "camera_id": self.camera_id,
            "test_type": self.test_type.value,
            "pan_config": self.pan_config.to_dict() if self.pan_config else None,
            "tilt_config": self.tilt_config.to_dict() if self.tilt_config else None,
            "zoom_config": self.zoom_config.to_dict() if self.zoom_config else None,
            "repetitions": self.repetitions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StressTestConfig:
        """Create from dictionary."""
        return cls(
            tenant_id=data["tenant_id"],
            camera_id=data["camera_id"],
            test_type=StressTestType(data["test_type"]),
            pan_config=AxisMovementConfig.from_dict(data["pan_config"]) if data.get("pan_config") else None,
            tilt_config=AxisMovementConfig.from_dict(data["tilt_config"]) if data.get("tilt_config") else None,
            zoom_config=AxisMovementConfig.from_dict(data["zoom_config"]) if data.get("zoom_config") else None,
            repetitions=data.get("repetitions", 1),
        )


@dataclass
class StressTestResult:
    """Results of a completed stress test."""

    success: bool
    position_match: bool  # Did camera return to expected position?
    position_error: dict[str, float]  # Final position error
    total_duration_ms: float
    movements: list[MovementTiming] = field(default_factory=list)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "position_match": self.position_match,
            "position_error": self.position_error,
            "total_duration_ms": self.total_duration_ms,
            "movements": [m.to_dict() for m in self.movements],
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StressTestResult:
        """Create from dictionary."""
        return cls(
            success=data["success"],
            position_match=data["position_match"],
            position_error=data["position_error"],
            total_duration_ms=data["total_duration_ms"],
            movements=[MovementTiming.from_dict(m) for m in data.get("movements", [])],
            error_message=data.get("error_message"),
        )


@dataclass
class StressTestSession:
    """A complete stress test session with configuration, results, and evaluation."""

    id: str  # UUID
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: StressTestStatus = StressTestStatus.PENDING
    tenant_id: str = ""
    camera_id: str = ""
    camera_name: str = ""
    config: StressTestConfig | None = None
    result: StressTestResult | None = None
    user_evaluation: UserEvaluation = UserEvaluation.NOT_EVALUATED
    user_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "tenant_id": self.tenant_id,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "config": self.config.to_dict() if self.config else None,
            "result": self.result.to_dict() if self.result else None,
            "user_evaluation": self.user_evaluation.value,
            "user_notes": self.user_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StressTestSession:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            status=StressTestStatus(data["status"]),
            tenant_id=data.get("tenant_id", ""),
            camera_id=data.get("camera_id", ""),
            camera_name=data.get("camera_name", ""),
            config=StressTestConfig.from_dict(data["config"]) if data.get("config") else None,
            result=StressTestResult.from_dict(data["result"]) if data.get("result") else None,
            user_evaluation=UserEvaluation(data.get("user_evaluation", "not_evaluated")),
            user_notes=data.get("user_notes", ""),
        )


@dataclass
class StressTestProgress:
    """Real-time progress information for a running stress test."""

    session_id: str
    status: StressTestStatus
    current_repetition: int = 0
    total_repetitions: int = 1
    current_movement: int = 0
    total_movements: int = 0
    current_position: dict[str, float] | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "current_repetition": self.current_repetition,
            "total_repetitions": self.total_repetitions,
            "current_movement": self.current_movement,
            "total_movements": self.total_movements,
            "current_position": self.current_position,
            "message": self.message,
        }


@dataclass
class StressTestPreset:
    """A pre-configured stress test preset."""

    name: str
    description: str
    test_type: StressTestType
    pan_config: AxisMovementConfig | None = None
    tilt_config: AxisMovementConfig | None = None
    zoom_config: AxisMovementConfig | None = None
    repetitions: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "pan_config": self.pan_config.to_dict() if self.pan_config else None,
            "tilt_config": self.tilt_config.to_dict() if self.tilt_config else None,
            "zoom_config": self.zoom_config.to_dict() if self.zoom_config else None,
            "repetitions": self.repetitions,
        }


# =============================================================================
# Built-in Stress Test Presets
# =============================================================================

STRESS_TEST_PRESETS: list[StressTestPreset] = [
    # 1. Oscillation Test (Pan 10 degrees)
    StressTestPreset(
        name="Oscillation Test (Pan 10°)",
        description="Back-and-forth pan movement of 10 degrees, 20 repetitions",
        test_type=StressTestType.OSCILLATION,
        pan_config=AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=10.0,
            step=10.0,
        ),
        repetitions=20,
    ),
    # 2. Oscillation Test (Tilt 5 degrees)
    StressTestPreset(
        name="Oscillation Test (Tilt 5°)",
        description="Back-and-forth tilt movement of 5 degrees, 20 repetitions",
        test_type=StressTestType.OSCILLATION,
        tilt_config=AxisMovementConfig(
            axis="tilt",
            start=0.0,
            end=5.0,
            step=5.0,
        ),
        repetitions=20,
    ),
    # 3. Random Step Accuracy (Pan)
    StressTestPreset(
        name="Random Step Accuracy (Pan)",
        description="Pan 90 degrees forward and backward with random 5-15 degree steps",
        test_type=StressTestType.RANDOM_STEP_ACCURACY,
        pan_config=AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=90.0,
            step_min=5.0,
            step_max=15.0,
            use_random_steps=True,
        ),
        repetitions=1,
    ),
    # 4. Full Range Sweep (Pan 360 degrees)
    StressTestPreset(
        name="Full Range Sweep (Pan 360°)",
        description="Single continuous pan movement across full 360 degree range, 3 repetitions",
        test_type=StressTestType.FULL_RANGE_SWEEP,
        pan_config=AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=360.0,
            step=360.0,  # Single movement
        ),
        repetitions=3,
    ),
    # 5. Tilt Stress (Full Range)
    StressTestPreset(
        name="Tilt Stress (Full Range)",
        description="Rapid tilt movement from -15 to 90 degrees, 10 repetitions",
        test_type=StressTestType.TILT_STRESS,
        tilt_config=AxisMovementConfig(
            axis="tilt",
            start=-15.0,
            end=90.0,
            step=105.0,  # Full range in one movement
        ),
        repetitions=10,
    ),
    # 6. Combined Axis Load
    StressTestPreset(
        name="Combined Axis Load",
        description="Simultaneous pan and tilt movements in diagonal pattern",
        test_type=StressTestType.COMBINED_AXIS_LOAD,
        pan_config=AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=45.0,
            step=15.0,
        ),
        tilt_config=AxisMovementConfig(
            axis="tilt",
            start=0.0,
            end=30.0,
            step=10.0,
        ),
        repetitions=5,
    ),
    # 7. Position Repeatability
    StressTestPreset(
        name="Position Repeatability",
        description="Move to same position 10 times and measure variance",
        test_type=StressTestType.POSITION_REPEATABILITY,
        pan_config=AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=45.0,
            step=45.0,
        ),
        tilt_config=AxisMovementConfig(
            axis="tilt",
            start=0.0,
            end=20.0,
            step=20.0,
        ),
        repetitions=10,
    ),
]


# =============================================================================
# Survey Models (from camera_survey)
# =============================================================================


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
        """Generate list of axis values for the survey."""
        steps = []
        current = self.start

        if self.axis == SurveyAxis.PAN and self.end < self.start:
            # Wraparound case
            while current <= 360.0:
                steps.append(round(current, 1))
                current += self.step
            current = 0.0
            while current <= self.end:
                steps.append(round(current, 1))
                current += self.step
        else:
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
