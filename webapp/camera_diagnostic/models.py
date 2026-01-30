"""Data models for the Camera Diagnostic Tool.

This module contains dataclasses for diagnostic session storage and stress testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from camera_survey.models import DeviceInfo

# =============================================================================
# Diagnostic Test Models
# =============================================================================


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
    origin: str = "server_executed"  # "server_executed" or "client_orchestrated"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML manifest."""
        return {
            "session": {
                "id": self.id,
                "created_at": self.created_at.isoformat(),
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "status": self.status.value,
                "origin": self.origin,
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
            origin=session_data.get("origin", "server_executed"),
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


# =============================================================================
# Stress Test Models
# =============================================================================


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
    max_speed: bool = False  # Use continuous movement at max speed (±100)

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
            "max_speed": self.max_speed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StressTestConfig:
        """Create from dictionary."""
        return cls(
            tenant_id=data["tenant_id"],
            camera_id=data["camera_id"],
            test_type=StressTestType(data["test_type"]),
            pan_config=AxisMovementConfig.from_dict(data["pan_config"])
            if data.get("pan_config")
            else None,
            tilt_config=AxisMovementConfig.from_dict(data["tilt_config"])
            if data.get("tilt_config")
            else None,
            zoom_config=AxisMovementConfig.from_dict(data["zoom_config"])
            if data.get("zoom_config")
            else None,
            repetitions=data.get("repetitions", 1),
            max_speed=data.get("max_speed", False),
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
    device_info: DeviceInfo | None = None  # Hardware info from camera API
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
            "device_info": self.device_info.to_dict() if self.device_info else None,
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
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            status=StressTestStatus(data["status"]),
            tenant_id=data.get("tenant_id", ""),
            camera_id=data.get("camera_id", ""),
            camera_name=data.get("camera_name", ""),
            device_info=DeviceInfo.from_dict(data.get("device_info"))
            if data.get("device_info")
            else None,
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
    max_speed: bool = False  # Use continuous movement at max speed (±100)

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
            "max_speed": self.max_speed,
        }


# =============================================================================
# Built-in Stress Test Presets
# =============================================================================

STRESS_TEST_PRESETS: list[StressTestPreset] = [
    # 1. Oscillation Test (Pan 10 degrees)
    StressTestPreset(
        name="Oscillation Test (Pan 10°)",
        description="Back-and-forth pan movement of 10 degrees, 10 repetitions",
        test_type=StressTestType.OSCILLATION,
        pan_config=AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=10.0,
            step=10.0,
        ),
        repetitions=10,
    ),
    # 2. Oscillation Test (Tilt 5 degrees)
    StressTestPreset(
        name="Oscillation Test (Tilt 5°)",
        description="Back-and-forth tilt movement of 5 degrees, 10 repetitions",
        test_type=StressTestType.OSCILLATION,
        tilt_config=AxisMovementConfig(
            axis="tilt",
            start=0.0,
            end=5.0,
            step=5.0,
        ),
        repetitions=10,
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
        description="Single pan movement across full 360 degree range, 2 repetitions",
        test_type=StressTestType.FULL_RANGE_SWEEP,
        pan_config=AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=360.0,
            step=360.0,  # Single movement
        ),
        repetitions=2,
    ),
    # 5. Tilt Stress (Full Range)
    StressTestPreset(
        name="Tilt Stress (Full Range)",
        description="Rapid tilt movement from -15 to 90 degrees, 5 repetitions",
        test_type=StressTestType.TILT_STRESS,
        tilt_config=AxisMovementConfig(
            axis="tilt",
            start=-15.0,
            end=90.0,
            step=105.0,  # Full range in one movement
        ),
        repetitions=5,
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
        repetitions=3,
    ),
    # 7. Position Repeatability
    StressTestPreset(
        name="Position Repeatability",
        description="Move to same position 5 times and measure variance",
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
        repetitions=5,
    ),
    # === MAX SPEED PRESETS (for installation stress testing) ===
    # 8. Max Speed Oscillation (Pan)
    StressTestPreset(
        name="MAX SPEED: Oscillation (Pan 30°)",
        description="Aggressive back-and-forth pan at maximum motor speed, 10 reps",
        test_type=StressTestType.OSCILLATION,
        pan_config=AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=30.0,
            step=30.0,
        ),
        repetitions=10,
        max_speed=True,
    ),
    # 9. Max Speed Oscillation (Tilt)
    StressTestPreset(
        name="MAX SPEED: Oscillation (Tilt 20°)",
        description="Aggressive back-and-forth tilt at maximum motor speed, 10 reps",
        test_type=StressTestType.OSCILLATION,
        tilt_config=AxisMovementConfig(
            axis="tilt",
            start=0.0,
            end=20.0,
            step=20.0,
        ),
        repetitions=10,
        max_speed=True,
    ),
    # 10. Max Speed Combined Axis
    StressTestPreset(
        name="MAX SPEED: Combined Axis",
        description="Diagonal movement at maximum speed - maximum torque stress",
        test_type=StressTestType.COMBINED_AXIS_LOAD,
        pan_config=AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=60.0,
            step=60.0,
        ),
        tilt_config=AxisMovementConfig(
            axis="tilt",
            start=0.0,
            end=40.0,
            step=40.0,
        ),
        repetitions=5,
        max_speed=True,
    ),
    # 11. Max Speed Full Range
    StressTestPreset(
        name="MAX SPEED: Full Range Sweep",
        description="Full 360° pan sweep at maximum speed, 3 reps",
        test_type=StressTestType.FULL_RANGE_SWEEP,
        pan_config=AxisMovementConfig(
            axis="pan",
            start=0.0,
            end=360.0,
            step=360.0,
        ),
        repetitions=3,
        max_speed=True,
    ),
]
