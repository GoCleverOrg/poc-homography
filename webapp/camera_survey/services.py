"""Business logic services for the camera survey app.

This module contains service functions for survey execution, PTZ control,
image capture, and session management.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml
from django.conf import settings

from poc_homography.camera_config import (
    get_camera_by_id,
    get_rtsp_url,
    get_tenant_by_id,
)

from .models import (
    SURVEY_PRESETS,
    CameraInfo,
    CaptureMetadata,
    CaptureRecord,
    DeviceInfo,
    PTZPosition,
    SurveyConfig,
    SurveyPreset,
    SurveyProgress,
    SurveySession,
    SurveyStatus,
    TenantInfo,
)
from .ptz import BasePTZCamera, create_ptz_camera

logger = logging.getLogger(__name__)

# RTSP connection settings
RTSP_CONNECTION_TIMEOUT_SEC = 10
JPEG_QUALITY = 90

# Survey execution lock (only one survey at a time)
_survey_lock = threading.Lock()

# In-memory storage for active survey progress
_active_surveys: dict[str, SurveyProgress] = {}
_active_surveys_lock = threading.Lock()  # Separate lock for _active_surveys access


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


def _get_survey_base_path() -> Path:
    """Get base path for survey storage.

    Returns:
        Path to survey storage directory (PROJECT_ROOT/survey/)
    """
    base_path = getattr(settings, "SURVEY_STORAGE_PATH", None)
    if base_path:
        return Path(base_path)
    return Path(settings.BASE_DIR).parent / "survey"


def _get_session_path(session_id: str, date_str: str | None = None) -> Path:
    """Get path for a survey session.

    Args:
        session_id: UUID of the session
        date_str: Date string in YYYYMMDD format, or None for today

    Returns:
        Path to session directory
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    return _get_survey_base_path() / date_str / session_id


def _format_ptz_value(value: float | None) -> str:
    """Format PTZ value with 1 decimal precision.

    Args:
        value: PTZ value or None

    Returns:
        Formatted string (e.g., "180.5") or "0.0" if None
    """
    if value is None:
        return "0.0"
    return f"{value:.1f}"


def _generate_image_filename(
    tenant_id: str,
    camera_id: str,
    timestamp: datetime,
    ptz: PTZPosition,
) -> str:
    """Generate image filename following format specification.

    Format: {tenant_id}_{camera_id}_{YYYYMMDD_HHMMSS}_{pan}_{tilt}_{zoom}.jpg

    Args:
        tenant_id: Tenant identifier
        camera_id: Camera identifier
        timestamp: Capture timestamp
        ptz: PTZ position at capture

    Returns:
        Filename string
    """
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    pan_str = _format_ptz_value(ptz.pan)
    tilt_str = _format_ptz_value(ptz.tilt)
    zoom_str = _format_ptz_value(ptz.zoom)

    return f"{tenant_id}_{camera_id}_{ts_str}_{pan_str}_{tilt_str}_{zoom_str}.jpg"


def _create_rtsp_capture(rtsp_url: str) -> cv2.VideoCapture:
    """Create OpenCV VideoCapture with RTSP-optimized settings.

    Args:
        rtsp_url: The RTSP URL to connect to

    Returns:
        Configured VideoCapture object
    """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_CONNECTION_TIMEOUT_SEC * 1000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_CONNECTION_TIMEOUT_SEC * 1000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _capture_frame(
    rtsp_url: str, output_path: Path
) -> tuple[bool, str | None, tuple[int, int] | None]:
    """Capture a single frame and save as JPEG.

    Args:
        rtsp_url: RTSP stream URL
        output_path: Path to save JPEG image

    Returns:
        Tuple of (success, error_message, resolution)
        resolution is (width, height) tuple if successful
    """
    cap = None
    try:
        cap = _create_rtsp_capture(rtsp_url)

        if not cap.isOpened():
            return False, "Failed to open RTSP stream", None

        # Read a frame (may need multiple attempts to get fresh frame)
        for _ in range(3):
            ret, frame = cap.read()
            if ret:
                break

        if not ret:
            return False, "Failed to read frame from stream", None

        # Get resolution
        height, width = frame.shape[:2]

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)

        if not success:
            return False, "Failed to encode frame as JPEG", None

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(output_path, "wb") as f:
            f.write(buffer.tobytes())

        return True, None, (width, height)

    except Exception as e:
        return False, str(e), None

    finally:
        if cap is not None:
            cap.release()


def _write_manifest(session: SurveySession, session_path: Path) -> bool:
    """Write session manifest to YAML file.

    Args:
        session: Survey session data
        session_path: Path to session directory

    Returns:
        True if successful, False otherwise
    """
    try:
        manifest_path = session_path / "manifest.yaml"
        session_path.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "w") as f:
            yaml.dump(session.to_dict(), f, default_flow_style=False, sort_keys=False)

        return True
    except Exception as e:
        logger.error(f"Failed to write manifest: {e}")
        return False


def _load_manifest(session_path: Path) -> SurveySession | None:
    """Load session manifest from YAML file.

    Args:
        session_path: Path to session directory

    Returns:
        SurveySession or None if failed
    """
    try:
        manifest_path = session_path / "manifest.yaml"
        if not manifest_path.exists():
            return None

        with open(manifest_path) as f:
            data = yaml.safe_load(f)

        # Reconstruct SurveySession from dict
        session_data = data.get("session", {})
        session = SurveySession(
            id=session_data.get("id", ""),
            start_time=datetime.fromisoformat(session_data["start_time"])
            if session_data.get("start_time")
            else datetime.now(timezone.utc),
            end_time=datetime.fromisoformat(session_data["end_time"])
            if session_data.get("end_time")
            else None,
            status=SurveyStatus(session_data.get("status", "completed")),
            abort_reason=session_data.get("abort_reason"),
        )

        # Load tenant info
        if "tenant" in data:
            session.tenant = TenantInfo(
                id=data["tenant"]["id"],
                name=data["tenant"]["name"],
            )

        # Load camera info
        if "camera" in data:
            session.camera = CameraInfo(
                id=data["camera"]["id"],
                name=data["camera"]["name"],
                ip=data["camera"]["ip"],
                model=data["camera"].get("model"),
            )

        # Load capture metadata
        if "capture_metadata" in data:
            session.capture_metadata = CaptureMetadata(
                resolution=data["capture_metadata"]["resolution"],
                codec=data["capture_metadata"].get("codec", "h264"),
            )

        # Load device info
        if "device_info" in data:
            session.device_info = DeviceInfo.from_dict(data["device_info"])

        # Load survey parameters
        if "survey_parameters" in data:
            from .models import SurveyAxis

            params = data["survey_parameters"]
            session.survey_parameters = SurveyConfig(
                tenant_id=session.tenant.id if session.tenant else "",
                camera_id=session.camera.id if session.camera else "",
                axis=SurveyAxis(params["axis"]),
                start=params["start"],
                end=params["end"],
                step=params["step"],
                restore_ptz=params.get("restore_ptz", True),
                retry_timeout=params.get("retry_timeout", 60),
            )

        # Load PTZ positions
        if "initial_ptz" in data:
            session.initial_ptz = PTZPosition.from_dict(data["initial_ptz"])
        if "final_ptz" in data:
            session.final_ptz = PTZPosition.from_dict(data["final_ptz"])

        # Load tags and captures
        session.session_tags = data.get("session_tags", [])
        session.captures = [CaptureRecord.from_dict(c) for c in data.get("captures", [])]

        return session

    except Exception as e:
        logger.error(f"Failed to load manifest from {session_path}: {e}")
        return None


class CameraSurveyService:
    """Service for managing camera survey operations."""

    def __init__(self):
        """Initialize survey service."""
        self._abort_flag = threading.Event()

    def start_survey(self, config: SurveyConfig) -> tuple[str, str | None]:
        """Start a new survey.

        Args:
            config: Survey configuration

        Returns:
            Tuple of (session_id, error_message)
            error_message is None on success
        """
        # Check if another survey is running
        if not _survey_lock.acquire(blocking=False):
            return "", "Another survey is already running"

        try:
            # Generate session ID and path
            session_id = str(uuid.uuid4())
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            session_path = _get_session_path(session_id, date_str)
            session_path.mkdir(parents=True, exist_ok=True)

            # Get camera and tenant info
            camera = get_camera_by_id(config.camera_id)
            if not camera:
                _survey_lock.release()
                return "", f"Camera not found: {config.camera_id}"

            tenant = get_tenant_by_id(config.tenant_id)
            if not tenant:
                _survey_lock.release()
                return "", f"Tenant not found: {config.tenant_id}"

            camera_ip = camera.get("ip")
            if not camera_ip:
                _survey_lock.release()
                return "", f"Camera {config.camera_id} has no IP address"

            # Create PTZ camera instance
            try:
                ptz_camera = create_ptz_camera(
                    camera_ip=camera_ip,
                    camera_name=camera.get("name", config.camera_id),
                    camera_model=camera.get("model"),
                )
            except ValueError as e:
                _survey_lock.release()
                return "", str(e)

            # Get RTSP URL - use camera_id which is the full ID like "valte_cam01"
            try:
                rtsp_url = get_rtsp_url(config.camera_id)
                if not rtsp_url:
                    _survey_lock.release()
                    return "", f"No RTSP URL configured for camera: {config.camera_id}"
            except ValueError as e:
                _survey_lock.release()
                return "", str(e)

            # Get initial PTZ position
            initial_ptz = ptz_camera.get_status()
            if initial_ptz is None:
                _survey_lock.release()
                return "", "Failed to get initial PTZ position"

            # Validate survey range
            capabilities = ptz_camera.get_capabilities()
            is_valid, error = capabilities.validate_range(config.axis, config.start, config.end)
            if not is_valid:
                _survey_lock.release()
                return "", error

            # Get device info from camera API
            device_info = ptz_camera.get_device_info()

            # Initialize session
            session = SurveySession(
                id=session_id,
                start_time=datetime.now(timezone.utc),
                status=SurveyStatus.RUNNING,
                tenant=TenantInfo(id=tenant["id"], name=tenant["name"]),
                camera=CameraInfo(
                    id=camera["id"],
                    name=camera["name"],
                    ip=camera_ip,
                    model=camera.get("model"),
                ),
                device_info=device_info,
                survey_parameters=config,
                initial_ptz=initial_ptz,
                session_tags=config.session_tags,
            )

            # Generate survey steps
            steps = config.generate_steps()
            total_steps = len(steps)

            # Initialize progress tracking
            with _active_surveys_lock:
                _active_surveys[session_id] = SurveyProgress(
                    session_id=session_id,
                    status=SurveyStatus.RUNNING,
                    step_count=0,
                    total_steps=total_steps,
                    current_ptz=initial_ptz,
                )

            # Reset abort flag
            self._abort_flag.clear()

            # Execute survey in background thread
            survey_thread = threading.Thread(
                target=self._execute_survey,
                args=(session, session_path, ptz_camera, rtsp_url, steps, config),
                daemon=True,
            )
            survey_thread.start()

            return session_id, None

        except Exception as e:
            logger.error(f"Failed to start survey: {e}")
            _survey_lock.release()
            return "", str(e)

    def _execute_survey(
        self,
        session: SurveySession,
        session_path: Path,
        ptz_camera: BasePTZCamera,
        rtsp_url: str,
        steps: list[float],
        config: SurveyConfig,
    ) -> None:
        """Execute survey loop in background thread.

        Args:
            session: Survey session object
            session_path: Path to session directory
            ptz_camera: PTZ camera instance
            rtsp_url: RTSP stream URL
            steps: List of axis values to survey
            config: Survey configuration
        """
        try:
            resolution = None

            for step_index, axis_value in enumerate(steps):
                # Check for abort
                if self._abort_flag.is_set():
                    session.status = SurveyStatus.ABORTED
                    session.abort_reason = "Survey aborted by user"
                    break

                # Build target position
                current_ptz = ptz_camera.get_status()
                if current_ptz is None:
                    logger.warning(f"Failed to get current PTZ at step {step_index}")
                    current_ptz = session.initial_ptz or PTZPosition()

                # Set target position based on axis
                if config.axis.value == "pan":
                    target_pan = axis_value
                    target_tilt = current_ptz.tilt
                    target_zoom = current_ptz.zoom
                elif config.axis.value == "tilt":
                    target_pan = current_ptz.pan
                    target_tilt = axis_value
                    target_zoom = current_ptz.zoom
                else:  # zoom
                    target_pan = current_ptz.pan
                    target_tilt = current_ptz.tilt
                    target_zoom = axis_value

                # Move to target position with retry logic
                move_success = self._move_with_retry(
                    ptz_camera, target_pan, target_tilt, target_zoom, config.retry_timeout
                )

                if not move_success:
                    session.status = SurveyStatus.FAILED
                    session.abort_reason = f"Failed to move PTZ to step {step_index} after {config.retry_timeout}s timeout"
                    break

                # Wait for stabilization
                final_pos = ptz_camera.wait_for_stabilization(max_wait=5.0)
                if final_pos is None:
                    final_pos = PTZPosition(pan=target_pan, tilt=target_tilt, zoom=target_zoom)

                # Capture frame
                timestamp = datetime.now(timezone.utc)
                filename = _generate_image_filename(
                    config.tenant_id, config.camera_id, timestamp, final_pos
                )
                image_path = session_path / filename

                capture_success, capture_error, frame_resolution = _capture_frame(
                    rtsp_url, image_path
                )

                # Track resolution from first successful capture
                if capture_success and resolution is None:
                    resolution = frame_resolution

                # Create capture record
                capture_record = CaptureRecord(
                    filename=filename,
                    timestamp=timestamp.isoformat(),
                    ptz=final_pos,
                    step_index=step_index,
                )

                if not capture_success:
                    capture_record.errors.append(capture_error or "Unknown capture error")
                    logger.warning(f"Capture failed at step {step_index}: {capture_error}")

                session.captures.append(capture_record)

                # Update progress
                with _active_surveys_lock:
                    if session.id in _active_surveys:
                        _active_surveys[session.id].step_count = step_index + 1
                        _active_surveys[session.id].current_ptz = final_pos
                        _active_surveys[session.id].last_capture_path = (
                            str(image_path) if capture_success else None
                        )

                # Write manifest after each step for crash recovery
                _write_manifest(session, session_path)

            # Survey complete or failed
            if session.status == SurveyStatus.RUNNING:
                session.status = SurveyStatus.COMPLETED

            # Set capture metadata
            if resolution:
                session.capture_metadata = CaptureMetadata(
                    resolution=f"{resolution[0]}x{resolution[1]}"
                )

            # Get final PTZ position
            session.final_ptz = ptz_camera.get_status()

            # Restore PTZ if requested and survey completed successfully
            if config.restore_ptz and session.status == SurveyStatus.COMPLETED:
                if session.initial_ptz:
                    restore_success = ptz_camera.move_to_position(
                        session.initial_ptz.pan,
                        session.initial_ptz.tilt,
                        session.initial_ptz.zoom,
                    )
                    if restore_success:
                        ptz_camera.wait_for_stabilization(max_wait=5.0)
                        session.final_ptz = ptz_camera.get_status()

            # Set end time
            session.end_time = datetime.now(timezone.utc)

            # Write final manifest
            _write_manifest(session, session_path)

            # Update progress
            with _active_surveys_lock:
                if session.id in _active_surveys:
                    _active_surveys[session.id].status = session.status

        except Exception as e:
            logger.error(f"Survey execution error: {e}")
            session.status = SurveyStatus.FAILED
            session.abort_reason = str(e)
            session.end_time = datetime.now(timezone.utc)
            _write_manifest(session, session_path)

            with _active_surveys_lock:
                if session.id in _active_surveys:
                    _active_surveys[session.id].status = SurveyStatus.FAILED
                    _active_surveys[session.id].error_message = str(e)

        finally:
            # Cleanup
            with _active_surveys_lock:
                if session.id in _active_surveys:
                    # Keep progress for a while for status queries
                    _active_surveys[session.id].status = session.status

            _survey_lock.release()

    def _move_with_retry(
        self,
        ptz_camera: BasePTZCamera,
        pan: float | None,
        tilt: float | None,
        zoom: float | None,
        timeout: int,
    ) -> bool:
        """Move PTZ with exponential backoff retry.

        Args:
            ptz_camera: PTZ camera instance
            pan: Target pan angle
            tilt: Target tilt angle
            zoom: Target zoom level
            timeout: Maximum time to retry in seconds

        Returns:
            True if move succeeded, False if timed out
        """
        start_time = time.time()
        backoff = 1.0  # Start with 1 second backoff

        while time.time() - start_time < timeout:
            if self._abort_flag.is_set():
                return False

            try:
                if ptz_camera.move_to_position(pan, tilt, zoom):
                    return True
            except Exception as e:
                logger.warning(f"PTZ move attempt failed: {e}")

            # Exponential backoff
            time.sleep(min(backoff, timeout - (time.time() - start_time)))
            backoff *= 2

        return False

    def abort_survey(self, session_id: str) -> tuple[bool, str | None]:
        """Abort a running survey.

        Args:
            session_id: ID of survey to abort

        Returns:
            Tuple of (success, error_message)
        """
        with _active_surveys_lock:
            if session_id not in _active_surveys:
                return False, "Survey not found or not running"

            if _active_surveys[session_id].status != SurveyStatus.RUNNING:
                return False, "Survey is not running"

        self._abort_flag.set()
        return True, None

    def get_survey_status(self, session_id: str) -> SurveyProgress | None:
        """Get current survey progress.

        Args:
            session_id: ID of survey

        Returns:
            SurveyProgress or None if not found
        """
        with _active_surveys_lock:
            return _active_surveys.get(session_id)

    def list_sessions(self, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """List survey sessions.

        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            Tuple of (sessions_list, total_count)
        """
        sessions = []
        base_path = _get_survey_base_path()

        if not base_path.exists():
            return [], 0

        # Scan date directories
        date_dirs = sorted(base_path.iterdir(), reverse=True)

        for date_dir in date_dirs:
            if not date_dir.is_dir():
                continue

            # Scan session directories within date
            session_dirs = sorted(date_dir.iterdir(), reverse=True)

            for session_dir in session_dirs:
                if not session_dir.is_dir():
                    continue

                manifest_path = session_dir / "manifest.yaml"
                if not manifest_path.exists():
                    continue

                try:
                    session = _load_manifest(session_dir)
                    if session:
                        sessions.append(
                            {
                                "id": session.id,
                                "date": date_dir.name,
                                "tenant_id": session.tenant.id if session.tenant else None,
                                "tenant_name": session.tenant.name if session.tenant else None,
                                "camera_id": session.camera.id if session.camera else None,
                                "camera_name": session.camera.name if session.camera else None,
                                "axis": session.survey_parameters.axis.value
                                if session.survey_parameters
                                else None,
                                "step_count": len(session.captures),
                                "status": session.status.value,
                                "start_time": session.start_time.isoformat()
                                if session.start_time
                                else None,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to load session from {session_dir}: {e}")

        total = len(sessions)
        return sessions[offset : offset + limit], total

    def get_session(self, session_id: str) -> SurveySession | None:
        """Get complete session data.

        Args:
            session_id: Session ID

        Returns:
            SurveySession or None if not found
        """
        # Validate session_id to prevent path traversal
        if not _is_valid_session_id(session_id):
            return None

        base_path = _get_survey_base_path()

        if not base_path.exists():
            return None

        # Search for session in date directories
        for date_dir in base_path.iterdir():
            if not date_dir.is_dir():
                continue

            session_path = date_dir / session_id
            if session_path.exists():
                return _load_manifest(session_path)

        return None

    def get_session_manifest_path(self, session_id: str) -> Path | None:
        """Get path to session manifest file.

        Args:
            session_id: Session ID

        Returns:
            Path to manifest.yaml or None if not found
        """
        # Validate session_id to prevent path traversal
        if not _is_valid_session_id(session_id):
            return None

        base_path = _get_survey_base_path()

        if not base_path.exists():
            return None

        for date_dir in base_path.iterdir():
            if not date_dir.is_dir():
                continue

            manifest_path = date_dir / session_id / "manifest.yaml"
            if manifest_path.exists():
                return manifest_path

        return None

    def get_session_image_path(self, session_id: str, filename: str) -> Path | None:
        """Get path to session image file.

        Args:
            session_id: Session ID
            filename: Image filename

        Returns:
            Path to image file or None if not found
        """
        # Validate session_id to prevent path traversal
        if not _is_valid_session_id(session_id):
            return None

        base_path = _get_survey_base_path()

        if not base_path.exists():
            return None

        # Validate filename to prevent path traversal
        if "/" in filename or "\\" in filename or ".." in filename:
            return None

        for date_dir in base_path.iterdir():
            if not date_dir.is_dir():
                continue

            image_path = date_dir / session_id / filename
            if image_path.exists():
                return image_path

        return None

    def delete_session(self, session_id: str) -> tuple[bool, str | None]:
        """Delete a survey session and all its data.

        Args:
            session_id: Session ID to delete

        Returns:
            Tuple of (success, error_message)
        """
        import shutil

        # Validate session_id to prevent path traversal
        if not _is_valid_session_id(session_id):
            return False, "Invalid session ID format"

        # Check if session is currently running
        with _active_surveys_lock:
            if session_id in _active_surveys:
                if _active_surveys[session_id].status == SurveyStatus.RUNNING:
                    return False, "Cannot delete a running survey session"

        base_path = _get_survey_base_path()

        if not base_path.exists():
            return False, "Session not found"

        # Find session directory
        session_path = None
        for date_dir in base_path.iterdir():
            if not date_dir.is_dir():
                continue

            candidate_path = date_dir / session_id
            if candidate_path.exists() and candidate_path.is_dir():
                session_path = candidate_path
                break

        if session_path is None:
            return False, "Session not found"

        try:
            # Remove the session directory and all its contents
            shutil.rmtree(session_path)

            # Clean up empty date directories
            date_dir = session_path.parent
            if date_dir.exists() and not any(date_dir.iterdir()):
                date_dir.rmdir()

            # Remove from active surveys if present
            with _active_surveys_lock:
                _active_surveys.pop(session_id, None)

            logger.info("Deleted session %s", session_id)
            return True, None

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False, str(e)


def get_survey_presets() -> list[SurveyPreset]:
    """Get list of available survey presets.

    Returns:
        List of SurveyPreset objects
    """
    return SURVEY_PRESETS
