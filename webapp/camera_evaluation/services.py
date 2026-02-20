"""Business logic services for the Camera Evaluation Tool.

Note: Stress testing services have been moved to camera_diagnostic app.
Survey services are imported from camera_survey app.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import cv2
from camera_diagnostic.services import create_rtsp_capture

from poc_homography.camera_config import get_rtsp_url

logger = logging.getLogger(__name__)


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
