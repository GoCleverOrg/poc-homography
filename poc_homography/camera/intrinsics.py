"""
Camera intrinsics computation from PTZ status.

This module provides functions to query a camera's current PTZ position and
compute the intrinsic matrix based on zoom level and sensor parameters.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import requests
from requests.auth import HTTPDigestAuth

from poc_homography.types import Degrees, Millimeters, Pixels, PixelsFloat, Unitless


@dataclass
class PTZStatus:
    """Camera PTZ (Pan-Tilt-Zoom) status."""

    pan: Degrees
    """Pan angle in degrees (azimuth)"""

    tilt: Degrees
    """Tilt angle in degrees (elevation, positive = down for Hikvision)"""

    zoom: Unitless
    """Zoom factor (1.0 = no zoom)"""


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    focal_length_mm: Millimeters
    """Focal length in millimeters (zoom-adjusted)"""

    focal_length_px: PixelsFloat
    """Focal length in pixels"""

    cx: PixelsFloat
    """Principal point X coordinate (pixels)"""

    cy: PixelsFloat
    """Principal point Y coordinate (pixels)"""

    sensor_width_mm: Millimeters
    """Sensor width in millimeters"""

    base_focal_length_mm: Millimeters
    """Base focal length in millimeters (at 1x zoom)"""

    image_width: Pixels
    """Image width in pixels"""

    image_height: Pixels
    """Image height in pixels"""

    K: np.ndarray
    """Intrinsic matrix (3x3 numpy array)"""


def get_ptz_status(
    ip: str,
    username: str,
    password: str,
    timeout: float = 5.0,
) -> PTZStatus:
    """
    Get current PTZ status from camera.

    Args:
        ip: Camera IP address
        username: Camera username
        password: Camera password
        timeout: Request timeout in seconds

    Returns:
        PTZ status with pan, tilt, and zoom

    Raises:
        RuntimeError: If cannot connect or parse response
    """
    url = f"http://{ip}/ISAPI/PTZCtrl/channels/1/status"

    try:
        response = requests.get(
            url,
            auth=HTTPDigestAuth(username, password),
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to connect to camera: {e}") from e

    if response.status_code != 200:
        raise RuntimeError(f"Camera returned status {response.status_code}")

    try:
        root = ET.fromstring(response.text)
        ns = {"h": "http://www.hikvision.com/ver20/XMLSchema"}

        azimuth = root.find(".//h:azimuth", ns)
        elevation = root.find(".//h:elevation", ns)
        abs_zoom = root.find(".//h:absoluteZoom", ns)

        if (
            azimuth is None
            or elevation is None
            or abs_zoom is None
            or azimuth.text is None
            or elevation.text is None
            or abs_zoom.text is None
        ):
            raise RuntimeError("Failed to parse PTZ status XML")

        return PTZStatus(
            pan=Degrees(float(azimuth.text) / 10),
            tilt=Degrees(float(elevation.text) / 10),
            zoom=Unitless(float(abs_zoom.text) / 10),
        )
    except ET.ParseError as e:
        raise RuntimeError(f"Failed to parse XML response: {e}") from e


def compute_intrinsics(
    zoom: Unitless,
    image_width: Pixels,
    image_height: Pixels,
    sensor_width_mm: Millimeters,
    base_focal_length_mm: Millimeters,
) -> CameraIntrinsics:
    """
    Compute camera intrinsics from zoom and sensor parameters.

    Args:
        zoom: Zoom factor (1.0 = no zoom)
        image_width: Image width in pixels
        image_height: Image height in pixels
        sensor_width_mm: Sensor width in millimeters
        base_focal_length_mm: Base focal length in millimeters (at 1x zoom)

    Returns:
        Camera intrinsic parameters including intrinsic matrix
    """
    # Compute focal length
    f_mm = Millimeters(base_focal_length_mm * zoom)
    f_px = PixelsFloat(f_mm * (image_width / sensor_width_mm))

    # Principal point (image center)
    cx = PixelsFloat(image_width / 2.0)
    cy = PixelsFloat(image_height / 2.0)

    # Intrinsic matrix
    K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])

    return CameraIntrinsics(
        focal_length_mm=f_mm,
        focal_length_px=f_px,
        cx=cx,
        cy=cy,
        sensor_width_mm=sensor_width_mm,
        base_focal_length_mm=base_focal_length_mm,
        image_width=image_width,
        image_height=image_height,
        K=K,
    )


def get_camera_intrinsics(
    camera_ip: str,
    username: str,
    password: str,
    image_width: Pixels,
    image_height: Pixels,
    sensor_width_mm: Millimeters,
    base_focal_length_mm: Millimeters,
    timeout: float = 5.0,
) -> tuple[PTZStatus, CameraIntrinsics]:
    """
    Get current intrinsics for a camera.

    Args:
        camera_ip: Camera IP address
        username: Camera username
        password: Camera password
        image_width: Image width in pixels
        image_height: Image height in pixels
        sensor_width_mm: Sensor width in millimeters
        base_focal_length_mm: Base focal length in millimeters (at 1x zoom)
        timeout: Request timeout in seconds

    Returns:
        Tuple of (PTZ status, camera intrinsics)

    Raises:
        RuntimeError: If cannot connect to camera or parse response
    """
    # Get PTZ status
    ptz_status = get_ptz_status(camera_ip, username, password, timeout)

    # Compute intrinsics
    intrinsics = compute_intrinsics(
        zoom=ptz_status.zoom,
        image_width=image_width,
        image_height=image_height,
        sensor_width_mm=sensor_width_mm,
        base_focal_length_mm=base_focal_length_mm,
    )

    return ptz_status, intrinsics
