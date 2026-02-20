"""Service for capturing frames from PTZ cameras."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

    from poc_homography.domain.entities.camera_config import CameraConfig
    from poc_homography.domain.entities.captured_frame import CapturedFrame
    from poc_homography.domain.protocols.camera_controller import CameraController
    from poc_homography.domain.repositories.captured_frame_repo import (
        CapturedFrameRepository,
    )


@dataclass(frozen=True)
class FrameCaptureResult:
    """Result of a frame capture operation.

    Attributes:
        frame: The captured frame entity (None if failed).
        image_path: Absolute path to the saved image (None if failed).
        success: Whether the capture succeeded.
        error: Error message if capture failed.
    """

    frame: CapturedFrame | None
    image_path: Path | None
    success: bool
    error: str | None = None


class FrameCaptureService:
    """Service for capturing and storing frames from PTZ cameras.

    This service orchestrates:
    1. Reading current PTZ state from camera
    2. Capturing a frame from RTSP stream
    3. Saving image to appropriate folder
    4. Creating and persisting CapturedFrame entity
    """

    def __init__(
        self,
        repo: CapturedFrameRepository,
    ) -> None:
        """Initialize the frame capture service.

        Args:
            repo: Repository for storing captured frames.
        """
        self._repo = repo

    def capture_frame(
        self,
        camera_config: CameraConfig,
        controller: CameraController,
        timestamp: datetime | None = None,
    ) -> FrameCaptureResult:
        """Capture a frame from a camera and store it.

        Args:
            camera_config: Camera configuration.
            controller: Camera controller for PTZ state.
            timestamp: Optional timestamp (defaults to now).

        Returns:
            FrameCaptureResult with the captured frame or error.
        """
        from poc_homography.domain.entities.captured_frame import CapturedFrame

        timestamp = timestamp or datetime.now()

        try:
            # 1. Get current PTZ state
            ptz_state = controller.get_ptz_status()

            # 2. Capture frame from RTSP
            rtsp_url = camera_config.rtsp_url("main")
            frame_data = self._capture_from_rtsp(rtsp_url)

            # 3. Determine paths
            ts_str = timestamp.isoformat().replace(":", "-")
            image_filename = f"{ts_str}.jpg"

            # Image stored alongside YAML in same folder
            image_dir = self._repo.image_dir_for(camera_config.map_id, camera_config.name)
            image_dir.mkdir(parents=True, exist_ok=True)

            image_path_abs = image_dir / image_filename
            image_path_rel = Path(image_filename)  # Relative to YAML

            # 4. Save image
            cv2.imwrite(str(image_path_abs), frame_data)

            # 5. Create and save entity
            captured_frame = CapturedFrame.create(
                map_id=camera_config.map_id,
                camera_name=camera_config.name,
                timestamp=timestamp,
                ptz_state=ptz_state,
                image_path=image_path_rel,
            )
            self._repo.save(captured_frame)

            return FrameCaptureResult(
                frame=captured_frame,
                image_path=image_path_abs,
                success=True,
            )

        except Exception as e:
            return FrameCaptureResult(
                frame=None,
                image_path=None,
                success=False,
                error=str(e),
            )

    def _capture_from_rtsp(self, rtsp_url: str, timeout_sec: float = 10.0) -> np.ndarray:
        """Capture a single frame from RTSP stream.

        Args:
            rtsp_url: RTSP URL to capture from.
            timeout_sec: Timeout for capture operation.

        Returns:
            Frame as numpy array (BGR format).

        Raises:
            RuntimeError: If capture fails.
        """
        import time

        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            start_time = time.time()
            frame = None

            while time.time() - start_time < timeout_sec:
                ret, frame = cap.read()
                if ret and frame is not None:
                    return frame
                time.sleep(0.1)

            raise RuntimeError(f"Failed to capture frame within {timeout_sec}s timeout")
        finally:
            cap.release()
