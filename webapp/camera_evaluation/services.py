"""Business logic services for the Camera Evaluation Tool.

This module contains service classes for stress testing PTZ cameras.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml
from django.conf import settings

from poc_homography.camera_config import get_camera_by_id, get_rtsp_url

# Use the PTZ abstraction layer for multi-brand camera support
from camera_survey.ptz import BasePTZCamera, create_ptz_camera
from camera_survey.models import PTZPosition

from .models import (
    AxisMovementConfig,
    MovementTiming,
    StressTestConfig,
    StressTestPreset,
    StressTestProgress,
    StressTestResult,
    StressTestSession,
    StressTestStatus,
    StressTestType,
    UserEvaluation,
)

logger = logging.getLogger(__name__)

# Constants for stress testing
STRESS_TEST_STABILIZATION_TIMEOUT = 10.0  # Max seconds to wait for position stabilization
STRESS_TEST_STABILIZATION_THRESHOLD = 0.5  # Seconds of no change to consider stabilized
STRESS_TEST_POLL_INTERVAL = 0.1  # Polling interval for position checks
POSITION_TOLERANCE = 0.5  # Position tolerance in degrees for matching


def _ptz_position_to_dict(position: PTZPosition | None) -> dict[str, float]:
    """Convert PTZPosition to dict for backward compatibility."""
    if position is None:
        return {"pan": 0.0, "tilt": 0.0, "zoom": 0.0}
    return {
        "pan": position.pan or 0.0,
        "tilt": position.tilt or 0.0,
        "zoom": position.zoom or 0.0,
    }


def _is_valid_session_id(session_id: str) -> bool:
    """Validate that session_id is a valid UUID to prevent path traversal.

    Args:
        session_id: String to validate

    Returns:
        True if valid UUID format, False otherwise
    """
    try:
        # Parse as UUID to validate format
        uuid.UUID(session_id)
        return True
    except (ValueError, TypeError):
        return False


def get_stress_test_storage_dir() -> Path:
    """Get the storage directory for stress test results."""
    base_dir = getattr(settings, "BASE_DIR", Path.cwd())
    return Path(base_dir).parent / "data" / "stress_test"


# =============================================================================
# MJPEG Streaming Service (reused from camera_diagnostic)
# =============================================================================

RTSP_CONNECTION_TIMEOUT_SEC = 10


def create_rtsp_capture(rtsp_url: str) -> cv2.VideoCapture:
    """Create an OpenCV VideoCapture with RTSP-optimized settings."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_CONNECTION_TIMEOUT_SEC * 1000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_CONNECTION_TIMEOUT_SEC * 1000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def generate_mjpeg_frames(camera_id: str) -> Generator[bytes, None, None]:
    """Generate MJPEG frames from an RTSP stream."""
    try:
        rtsp_url = get_rtsp_url(camera_id)
    except ValueError as e:
        logger.error(f"Failed to get RTSP URL for {camera_id}: {e}")
        return

    if not rtsp_url:
        logger.error(f"Camera not found: {camera_id}")
        return

    cap = create_rtsp_capture(rtsp_url)

    try:
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream for {camera_id}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from {camera_id}, ending stream")
                break

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            success, buffer = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    finally:
        cap.release()


# =============================================================================
# Camera Stress Test Service
# =============================================================================


class CameraStressTestService:
    """Service for executing and managing PTZ stress tests."""

    # Class-level storage for active sessions (session_id -> session data)
    _active_sessions: dict[str, StressTestSession] = {}
    _session_progress: dict[str, StressTestProgress] = {}
    _session_threads: dict[str, threading.Thread] = {}
    _abort_flags: dict[str, bool] = {}
    _lock = threading.Lock()

    @classmethod
    def start_stress_test(cls, config: StressTestConfig) -> tuple[str | None, str | None]:
        """Start a new stress test session.

        Args:
            config: Stress test configuration

        Returns:
            Tuple of (session_id, error_message)
        """
        # Get camera info
        camera = get_camera_by_id(config.camera_id)
        if not camera:
            return None, f"Camera not found: {config.camera_id}"

        camera_ip = camera.get("ip")
        if not camera_ip:
            return None, f"No IP address for camera: {config.camera_id}"

        username = os.getenv("CAMERA_USERNAME")
        password = os.getenv("CAMERA_PASSWORD")
        if not username or not password:
            return None, "Camera credentials not set (CAMERA_USERNAME, CAMERA_PASSWORD)"

        # Create session
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        session = StressTestSession(
            id=session_id,
            created_at=now,
            status=StressTestStatus.PENDING,
            tenant_id=config.tenant_id,
            camera_id=config.camera_id,
            camera_name=camera.get("name", config.camera_id),
            config=config,
        )

        progress = StressTestProgress(
            session_id=session_id,
            status=StressTestStatus.PENDING,
            total_repetitions=config.repetitions,
            message="Initializing...",
        )

        with cls._lock:
            cls._active_sessions[session_id] = session
            cls._session_progress[session_id] = progress
            cls._abort_flags[session_id] = False

        # Start execution in background thread
        thread = threading.Thread(
            target=cls._execute_stress_test,
            args=(session_id, camera_ip, username, password),
            daemon=True,
        )
        cls._session_threads[session_id] = thread
        thread.start()

        return session_id, None

    @classmethod
    def _execute_stress_test(
        cls,
        session_id: str,
        camera_ip: str,
        username: str,
        password: str,
    ) -> None:
        """Execute stress test in background thread."""
        session = cls._active_sessions.get(session_id)
        if not session or not session.config:
            return

        config = session.config

        # Update session status
        with cls._lock:
            session.status = StressTestStatus.RUNNING
            session.started_at = datetime.now(timezone.utc)
            cls._session_progress[session_id].status = StressTestStatus.RUNNING
            cls._session_progress[session_id].message = "Connecting to camera..."

        # Initialize PTZ controller using abstraction layer
        ptz = create_ptz_camera(camera_ip, session.camera_name)

        movements: list[MovementTiming] = []
        test_start_time = time.time()
        initial_position: dict[str, float] | None = None

        try:
            # Get initial position
            initial_status = ptz.get_status()
            if not initial_status:
                raise RuntimeError("Failed to get initial camera position")

            initial_position = _ptz_position_to_dict(initial_status)

            with cls._lock:
                cls._session_progress[session_id].current_position = initial_position
                cls._session_progress[session_id].message = "Starting test..."

            # Execute test based on type
            if config.test_type == StressTestType.OSCILLATION:
                movements = cls._execute_oscillation_test(session_id, ptz, config)
            elif config.test_type == StressTestType.RANDOM_STEP_ACCURACY:
                movements = cls._execute_random_step_accuracy_test(session_id, ptz, config)
            elif config.test_type == StressTestType.FULL_RANGE_SWEEP:
                movements = cls._execute_full_range_sweep(session_id, ptz, config)
            elif config.test_type == StressTestType.TILT_STRESS:
                movements = cls._execute_tilt_stress_test(session_id, ptz, config)
            elif config.test_type == StressTestType.COMBINED_AXIS_LOAD:
                movements = cls._execute_combined_axis_load(session_id, ptz, config)
            elif config.test_type == StressTestType.POSITION_REPEATABILITY:
                movements = cls._execute_position_repeatability(session_id, ptz, config)
            elif config.test_type == StressTestType.SPEED_TEST:
                movements = cls._execute_speed_test(session_id, ptz, config)

            # Check if aborted
            if cls._abort_flags.get(session_id):
                with cls._lock:
                    session.status = StressTestStatus.ABORTED
                    cls._session_progress[session_id].status = StressTestStatus.ABORTED
                    cls._session_progress[session_id].message = "Test aborted by user"
                # Position will be restored in finally block
                return

            # Get final position and calculate error
            final_position = _ptz_position_to_dict(ptz.get_status())

            position_error = {
                "pan": abs(final_position["pan"] - initial_position["pan"]),
                "tilt": abs(final_position["tilt"] - initial_position["tilt"]),
                "zoom": abs(final_position["zoom"] - initial_position["zoom"]),
            }

            position_match = cls._positions_match(initial_position, final_position)
            total_duration_ms = (time.time() - test_start_time) * 1000

            # Create result - success depends on position_match
            result = StressTestResult(
                success=position_match,
                position_match=position_match,
                position_error=position_error,
                total_duration_ms=total_duration_ms,
                movements=movements,
            )

            # Update session
            with cls._lock:
                session.status = StressTestStatus.COMPLETED
                session.completed_at = datetime.now(timezone.utc)
                session.result = result
                cls._session_progress[session_id].status = StressTestStatus.COMPLETED
                cls._session_progress[session_id].message = "Test completed successfully"
                cls._session_progress[session_id].current_position = final_position

            # Save session to disk
            cls._save_session(session)

        except Exception as e:
            logger.exception(f"Stress test {session_id} failed: {e}")
            total_duration_ms = (time.time() - test_start_time) * 1000

            result = StressTestResult(
                success=False,
                position_match=False,
                position_error={"pan": 0, "tilt": 0, "zoom": 0},
                total_duration_ms=total_duration_ms,
                movements=movements,
                error_message=str(e),
            )

            with cls._lock:
                session.status = StressTestStatus.FAILED
                session.completed_at = datetime.now(timezone.utc)
                session.result = result
                cls._session_progress[session_id].status = StressTestStatus.FAILED
                cls._session_progress[session_id].message = f"Test failed: {e}"

            cls._save_session(session)

        finally:
            # Always restore initial position if we have it
            if initial_position is not None:
                try:
                    with cls._lock:
                        cls._session_progress[session_id].message = "Restoring initial position..."
                    ptz.move_to_position(
                        initial_position["pan"], initial_position["tilt"], initial_position["zoom"]
                    )
                    ptz.wait_for_stabilization()
                    with cls._lock:
                        cls._session_progress[session_id].current_position = initial_position
                    logger.info(f"Restored initial position for session {session_id}")
                except Exception as restore_error:
                    logger.warning(
                        f"Failed to restore initial position for session {session_id}: {restore_error}"
                    )

    @classmethod
    def _measure_movement(
        cls,
        ptz: BasePTZCamera,
        target_pan: float,
        target_tilt: float,
        target_zoom: float,
    ) -> MovementTiming:
        """Measure timing for a single movement operation."""
        # Get starting position
        start_position = _ptz_position_to_dict(ptz.get_status())

        target_position = {
            "pan": target_pan,
            "tilt": target_tilt,
            "zoom": target_zoom,
        }

        # Send move command
        command_sent = datetime.now(timezone.utc)
        ptz.move_to_position(target_pan, target_tilt, target_zoom)

        # Wait for stabilization
        final_position = cls._wait_for_stabilization(ptz)
        stabilized = datetime.now(timezone.utc)

        duration_ms = (stabilized - command_sent).total_seconds() * 1000

        # Calculate position error
        position_error = {
            "pan": abs(final_position["pan"] - target_pan),
            "tilt": abs(final_position["tilt"] - target_tilt),
            "zoom": abs(final_position["zoom"] - target_zoom),
        }

        return MovementTiming(
            command_sent=command_sent,
            stabilized=stabilized,
            duration_ms=duration_ms,
            start_position=start_position,
            end_position=final_position,
            target_position=target_position,
            position_error=position_error,
        )

    @classmethod
    def _wait_for_stabilization(
        cls,
        ptz: BasePTZCamera,
        max_wait: float = STRESS_TEST_STABILIZATION_TIMEOUT,
    ) -> dict[str, float]:
        """Wait for PTZ position to stabilize."""
        start_time = time.time()
        last_position: dict[str, float] | None = None
        stable_since = None

        while time.time() - start_time < max_wait:
            status = ptz.get_status()
            if status is None:
                time.sleep(STRESS_TEST_POLL_INTERVAL)
                continue

            current_position = _ptz_position_to_dict(status)

            if last_position is not None:
                position_changed = False
                for axis in ["pan", "tilt", "zoom"]:
                    if abs(current_position[axis] - last_position[axis]) > 0.1:
                        position_changed = True
                        break

                if not position_changed:
                    if stable_since is None:
                        stable_since = time.time()
                    elif time.time() - stable_since >= STRESS_TEST_STABILIZATION_THRESHOLD:
                        return current_position
                else:
                    stable_since = None

            last_position = current_position
            time.sleep(STRESS_TEST_POLL_INTERVAL)

        # Return last known position even if not stabilized
        return last_position or {"pan": 0.0, "tilt": 0.0, "zoom": 0.0}

    @classmethod
    def _positions_match(
        cls,
        pos1: dict[str, float],
        pos2: dict[str, float],
        tolerance: float = POSITION_TOLERANCE,
    ) -> bool:
        """Check if two positions match within tolerance."""
        for axis in ["pan", "tilt", "zoom"]:
            if abs(pos1.get(axis, 0) - pos2.get(axis, 0)) > tolerance:
                return False
        return True

    @classmethod
    def _generate_random_steps(cls, total: float, step_min: float, step_max: float) -> list[float]:
        """Generate list of random step sizes that sum to total."""
        steps = []
        remaining = abs(total)
        direction = 1 if total >= 0 else -1

        while remaining > 0:
            max_step = min(step_max, remaining)
            if max_step < step_min:
                steps.append(remaining * direction)
                break
            step = random.uniform(step_min, max_step)
            steps.append(step * direction)
            remaining -= step

        return steps

    # -------------------------------------------------------------------------
    # Test Execution Methods
    # -------------------------------------------------------------------------

    @classmethod
    def _execute_oscillation_test(
        cls,
        session_id: str,
        ptz: BasePTZCamera,
        config: StressTestConfig,
    ) -> list[MovementTiming]:
        """Execute oscillation test - back and forth movement."""
        movements = []
        axis_config = config.pan_config or config.tilt_config

        if not axis_config:
            return movements

        # Get current position
        status = ptz.get_status()
        current_pan = (status.pan or 0) if status else 0
        current_tilt = (status.tilt or 0) if status else 0
        current_zoom = (status.zoom or 0) if status else 0

        # Calculate total movements (2 per repetition: forward and back)
        total_movements = config.repetitions * 2

        with cls._lock:
            cls._session_progress[session_id].total_movements = total_movements

        for rep in range(config.repetitions):
            if cls._abort_flags.get(session_id):
                break

            with cls._lock:
                cls._session_progress[session_id].current_repetition = rep + 1
                cls._session_progress[
                    session_id
                ].message = f"Repetition {rep + 1}/{config.repetitions}"

            # Move forward
            if axis_config.axis == "pan":
                target_pan = current_pan + axis_config.end - axis_config.start
                target_tilt = current_tilt
            else:
                target_pan = current_pan
                target_tilt = current_tilt + axis_config.end - axis_config.start
            target_zoom = current_zoom

            with cls._lock:
                cls._session_progress[session_id].current_movement = len(movements) + 1

            movement = cls._measure_movement(ptz, target_pan, target_tilt, target_zoom)
            movements.append(movement)

            with cls._lock:
                cls._session_progress[session_id].current_position = movement.end_position

            # Move back
            with cls._lock:
                cls._session_progress[session_id].current_movement = len(movements) + 1

            movement = cls._measure_movement(ptz, current_pan, current_tilt, current_zoom)
            movements.append(movement)

            with cls._lock:
                cls._session_progress[session_id].current_position = movement.end_position

        return movements

    @classmethod
    def _execute_random_step_accuracy_test(
        cls,
        session_id: str,
        ptz: BasePTZCamera,
        config: StressTestConfig,
    ) -> list[MovementTiming]:
        """Execute random step accuracy test."""
        movements = []
        axis_config = config.pan_config or config.tilt_config

        if not axis_config:
            return movements

        # Get current position
        status = ptz.get_status()
        current_pan = (status.pan or 0) if status else 0
        current_tilt = (status.tilt or 0) if status else 0
        current_zoom = (status.zoom or 0) if status else 0
        start_pan, start_tilt = current_pan, current_tilt

        # Generate random steps
        total_distance = axis_config.end - axis_config.start
        forward_steps = cls._generate_random_steps(
            total_distance, axis_config.step_min, axis_config.step_max
        )
        backward_steps = cls._generate_random_steps(
            -total_distance, axis_config.step_min, axis_config.step_max
        )

        total_movements = (len(forward_steps) + len(backward_steps)) * config.repetitions

        with cls._lock:
            cls._session_progress[session_id].total_movements = total_movements

        for rep in range(config.repetitions):
            if cls._abort_flags.get(session_id):
                break

            with cls._lock:
                cls._session_progress[session_id].current_repetition = rep + 1
                cls._session_progress[session_id].message = f"Repetition {rep + 1}: Moving forward"

            # Forward steps
            for step in forward_steps:
                if cls._abort_flags.get(session_id):
                    break

                if axis_config.axis == "pan":
                    current_pan += step
                else:
                    current_tilt += step

                with cls._lock:
                    cls._session_progress[session_id].current_movement = len(movements) + 1

                movement = cls._measure_movement(ptz, current_pan, current_tilt, current_zoom)
                movements.append(movement)

                with cls._lock:
                    cls._session_progress[session_id].current_position = movement.end_position

            # Backward steps
            with cls._lock:
                cls._session_progress[session_id].message = f"Repetition {rep + 1}: Moving backward"

            for step in backward_steps:
                if cls._abort_flags.get(session_id):
                    break

                if axis_config.axis == "pan":
                    current_pan += step
                else:
                    current_tilt += step

                with cls._lock:
                    cls._session_progress[session_id].current_movement = len(movements) + 1

                movement = cls._measure_movement(ptz, current_pan, current_tilt, current_zoom)
                movements.append(movement)

                with cls._lock:
                    cls._session_progress[session_id].current_position = movement.end_position

            # Reset to start position
            current_pan, current_tilt = start_pan, start_tilt

        return movements

    @classmethod
    def _execute_full_range_sweep(
        cls,
        session_id: str,
        ptz: BasePTZCamera,
        config: StressTestConfig,
    ) -> list[MovementTiming]:
        """Execute full range sweep test."""
        movements = []
        axis_config = config.pan_config or config.tilt_config

        if not axis_config:
            return movements

        status = ptz.get_status()
        current_pan = (status.pan or 0) if status else 0
        current_tilt = (status.tilt or 0) if status else 0
        current_zoom = (status.zoom or 0) if status else 0

        total_movements = config.repetitions * 2  # Forward and back

        with cls._lock:
            cls._session_progress[session_id].total_movements = total_movements

        for rep in range(config.repetitions):
            if cls._abort_flags.get(session_id):
                break

            with cls._lock:
                cls._session_progress[session_id].current_repetition = rep + 1
                cls._session_progress[
                    session_id
                ].message = f"Sweep {rep + 1}/{config.repetitions}: Forward"

            # Sweep forward
            if axis_config.axis == "pan":
                target_pan = axis_config.end
                target_tilt = current_tilt
            else:
                target_pan = current_pan
                target_tilt = axis_config.end

            with cls._lock:
                cls._session_progress[session_id].current_movement = len(movements) + 1

            movement = cls._measure_movement(ptz, target_pan, target_tilt, current_zoom)
            movements.append(movement)

            with cls._lock:
                cls._session_progress[session_id].current_position = movement.end_position
                cls._session_progress[
                    session_id
                ].message = f"Sweep {rep + 1}/{config.repetitions}: Return"

            # Sweep back
            if axis_config.axis == "pan":
                target_pan = axis_config.start
            else:
                target_tilt = axis_config.start

            with cls._lock:
                cls._session_progress[session_id].current_movement = len(movements) + 1

            movement = cls._measure_movement(ptz, target_pan, target_tilt, current_zoom)
            movements.append(movement)

            with cls._lock:
                cls._session_progress[session_id].current_position = movement.end_position

        return movements

    @classmethod
    def _execute_tilt_stress_test(
        cls,
        session_id: str,
        ptz: BasePTZCamera,
        config: StressTestConfig,
    ) -> list[MovementTiming]:
        """Execute tilt stress test - rapid tilt movements."""
        # Reuse full range sweep logic for tilt
        return cls._execute_full_range_sweep(session_id, ptz, config)

    @classmethod
    def _execute_combined_axis_load(
        cls,
        session_id: str,
        ptz: BasePTZCamera,
        config: StressTestConfig,
    ) -> list[MovementTiming]:
        """Execute combined axis load test - simultaneous pan and tilt."""
        movements = []

        if not config.pan_config or not config.tilt_config:
            return movements

        status = ptz.get_status()
        start_pan = (status.pan or 0) if status else 0
        start_tilt = (status.tilt or 0) if status else 0
        current_zoom = (status.zoom or 0) if status else 0

        total_movements = config.repetitions * 2

        with cls._lock:
            cls._session_progress[session_id].total_movements = total_movements

        for rep in range(config.repetitions):
            if cls._abort_flags.get(session_id):
                break

            with cls._lock:
                cls._session_progress[session_id].current_repetition = rep + 1
                cls._session_progress[
                    session_id
                ].message = f"Diagonal {rep + 1}/{config.repetitions}: Moving"

            # Move diagonally to end position
            target_pan = start_pan + (config.pan_config.end - config.pan_config.start)
            target_tilt = start_tilt + (config.tilt_config.end - config.tilt_config.start)

            with cls._lock:
                cls._session_progress[session_id].current_movement = len(movements) + 1

            movement = cls._measure_movement(ptz, target_pan, target_tilt, current_zoom)
            movements.append(movement)

            with cls._lock:
                cls._session_progress[session_id].current_position = movement.end_position
                cls._session_progress[
                    session_id
                ].message = f"Diagonal {rep + 1}/{config.repetitions}: Return"

            # Return to start
            with cls._lock:
                cls._session_progress[session_id].current_movement = len(movements) + 1

            movement = cls._measure_movement(ptz, start_pan, start_tilt, current_zoom)
            movements.append(movement)

            with cls._lock:
                cls._session_progress[session_id].current_position = movement.end_position

        return movements

    @classmethod
    def _execute_position_repeatability(
        cls,
        session_id: str,
        ptz: BasePTZCamera,
        config: StressTestConfig,
    ) -> list[MovementTiming]:
        """Execute position repeatability test - same position multiple times."""
        movements = []

        status = ptz.get_status()
        start_pan = (status.pan or 0) if status else 0
        start_tilt = (status.tilt or 0) if status else 0
        current_zoom = (status.zoom or 0) if status else 0

        # Calculate target position
        target_pan = start_pan
        target_tilt = start_tilt
        if config.pan_config:
            target_pan = start_pan + (config.pan_config.end - config.pan_config.start)
        if config.tilt_config:
            target_tilt = start_tilt + (config.tilt_config.end - config.tilt_config.start)

        total_movements = config.repetitions * 2

        with cls._lock:
            cls._session_progress[session_id].total_movements = total_movements

        for rep in range(config.repetitions):
            if cls._abort_flags.get(session_id):
                break

            with cls._lock:
                cls._session_progress[session_id].current_repetition = rep + 1
                cls._session_progress[
                    session_id
                ].message = f"Position test {rep + 1}/{config.repetitions}: To target"

            # Move to target
            with cls._lock:
                cls._session_progress[session_id].current_movement = len(movements) + 1

            movement = cls._measure_movement(ptz, target_pan, target_tilt, current_zoom)
            movements.append(movement)

            with cls._lock:
                cls._session_progress[session_id].current_position = movement.end_position
                cls._session_progress[
                    session_id
                ].message = f"Position test {rep + 1}/{config.repetitions}: Return"

            # Return to start
            with cls._lock:
                cls._session_progress[session_id].current_movement = len(movements) + 1

            movement = cls._measure_movement(ptz, start_pan, start_tilt, current_zoom)
            movements.append(movement)

            with cls._lock:
                cls._session_progress[session_id].current_position = movement.end_position

        return movements

    @classmethod
    def _execute_speed_test(
        cls,
        session_id: str,
        ptz: BasePTZCamera,
        config: StressTestConfig,
    ) -> list[MovementTiming]:
        """Execute speed test - measure degrees per second."""
        # Similar to full range sweep but with specific timing analysis
        return cls._execute_full_range_sweep(session_id, ptz, config)

    # -------------------------------------------------------------------------
    # Session Management Methods
    # -------------------------------------------------------------------------

    @classmethod
    def abort_stress_test(cls, session_id: str) -> tuple[bool, str | None]:
        """Abort a running stress test."""
        with cls._lock:
            if session_id not in cls._active_sessions:
                return False, "Session not found"

            session = cls._active_sessions[session_id]
            if session.status != StressTestStatus.RUNNING:
                return False, "Session is not running"

            cls._abort_flags[session_id] = True

        return True, None

    @classmethod
    def get_stress_test_status(cls, session_id: str) -> StressTestProgress | None:
        """Get current progress of a stress test."""
        with cls._lock:
            return cls._session_progress.get(session_id)

    @classmethod
    def get_session(cls, session_id: str) -> StressTestSession | None:
        """Get a session by ID (from memory or disk)."""
        # Check memory first
        with cls._lock:
            if session_id in cls._active_sessions:
                return cls._active_sessions[session_id]

        # Try to load from disk
        return cls._load_session(session_id)

    @classmethod
    def update_user_evaluation(
        cls,
        session_id: str,
        evaluation: UserEvaluation,
        notes: str = "",
    ) -> tuple[bool, str | None]:
        """Update user evaluation for a session."""
        session = cls.get_session(session_id)
        if not session:
            return False, "Session not found"

        session.user_evaluation = evaluation
        session.user_notes = notes

        # Update in memory if present
        with cls._lock:
            if session_id in cls._active_sessions:
                cls._active_sessions[session_id] = session

        # Save to disk
        cls._save_session(session)
        return True, None

    @classmethod
    def list_stress_test_sessions(
        cls,
        tenant_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[StressTestSession], int]:
        """List stress test sessions with optional filtering."""
        sessions = []

        # Get sessions from disk
        storage_dir = get_stress_test_storage_dir()
        if not storage_dir.exists():
            return [], 0

        # Find all session directories
        session_dirs = []
        for date_dir in sorted(storage_dir.iterdir(), reverse=True):
            if date_dir.is_dir():
                for session_dir in date_dir.iterdir():
                    if session_dir.is_dir():
                        manifest_path = session_dir / "manifest.yaml"
                        if manifest_path.exists():
                            session_dirs.append((session_dir, manifest_path))

        # Load and filter sessions
        for session_dir, manifest_path in session_dirs:
            try:
                session = cls._load_session_from_path(manifest_path)
                if session:
                    # Apply filters
                    if tenant_id and session.tenant_id != tenant_id:
                        continue
                    if camera_id and session.camera_id != camera_id:
                        continue
                    sessions.append(session)
            except Exception as e:
                logger.warning(f"Failed to load session from {manifest_path}: {e}")

        # Add active sessions from memory
        with cls._lock:
            for session in cls._active_sessions.values():
                if tenant_id and session.tenant_id != tenant_id:
                    continue
                if camera_id and session.camera_id != camera_id:
                    continue
                # Only add if not already in list
                if not any(s.id == session.id for s in sessions):
                    sessions.append(session)

        # Sort by created_at descending
        sessions.sort(key=lambda s: s.created_at, reverse=True)

        total = len(sessions)
        sessions = sessions[offset : offset + limit]

        return sessions, total

    @classmethod
    def delete_session(cls, session_id: str) -> tuple[bool, str | None]:
        """Delete a stress test session."""
        # Validate session_id to prevent path traversal
        if not _is_valid_session_id(session_id):
            return False, "Invalid session ID format"

        # Remove from memory
        with cls._lock:
            cls._active_sessions.pop(session_id, None)
            cls._session_progress.pop(session_id, None)
            cls._abort_flags.pop(session_id, None)

        # Remove from disk
        storage_dir = get_stress_test_storage_dir()
        if storage_dir.exists():
            for date_dir in storage_dir.iterdir():
                if date_dir.is_dir():
                    session_dir = date_dir / session_id
                    if session_dir.exists():
                        import shutil

                        shutil.rmtree(session_dir)
                        return True, None

        return False, "Session not found"

    # -------------------------------------------------------------------------
    # Storage Methods
    # -------------------------------------------------------------------------

    @classmethod
    def _save_session(cls, session: StressTestSession) -> None:
        """Save session to disk."""
        storage_dir = get_stress_test_storage_dir()
        date_str = session.created_at.strftime("%Y%m%d")
        session_dir = storage_dir / date_str / session.id
        session_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = session_dir / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(session.to_dict(), f, default_flow_style=False)

    @classmethod
    def _load_session(cls, session_id: str) -> StressTestSession | None:
        """Load session from disk by ID."""
        # Validate session_id to prevent path traversal
        if not _is_valid_session_id(session_id):
            return None

        storage_dir = get_stress_test_storage_dir()
        if not storage_dir.exists():
            return None

        for date_dir in storage_dir.iterdir():
            if date_dir.is_dir():
                manifest_path = date_dir / session_id / "manifest.yaml"
                if manifest_path.exists():
                    return cls._load_session_from_path(manifest_path)

        return None

    @classmethod
    def _load_session_from_path(cls, manifest_path: Path) -> StressTestSession | None:
        """Load session from manifest file."""
        try:
            with open(manifest_path) as f:
                data = yaml.safe_load(f)
            return StressTestSession.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load session from {manifest_path}: {e}")
            return None
