"""
Camera configuration file.
Central location for all camera settings and credentials.
"""

import os

# Camera credentials - loaded from environment variables
USERNAME = os.getenv("CAMERA_USERNAME")
PASSWORD = os.getenv("CAMERA_PASSWORD")

# =============================================================================
# CAMERA LENS SPECIFICATIONS (Hikvision DS-2DF8425IX-AELW series)
# =============================================================================
# Source: Hikvision official datasheet
# - Sensor: 1/1.8" progressive scan CMOS (diagonal ~8.86mm)
# - Resolution: 2560×1440 (4MP, 16:9 aspect ratio)
# - Focal Length: 5.9mm (wide) to 147.5mm (tele)
# - Optical Zoom: 25× (zoom_factor = focal_length / 5.9)
# - Horizontal FOV: 59.8° (wide) to 3.3° (tele)
# - Aperture: F1.5 (max)
#
# SENSOR WIDTH CALCULATION:
# Using FOV formula: FOV = 2 * arctan(sensor_width / (2 * focal_length))
# At wide end: 59.8° = 2 * arctan(sensor_width / 11.8)
# Solving: sensor_width = 11.8 * tan(29.9°) = 6.78mm
#
# Note: The geometric sensor width (6.78mm) differs from the physical sensor
# because the stated FOV accounts for some lens distortion effects.
# =============================================================================

# Default camera sensor parameters for Hikvision DS-2DF8425IX series
DEFAULT_SENSOR_WIDTH_MM = 6.78  # Calculated from 59.8° FOV at 5.9mm focal length
DEFAULT_BASE_FOCAL_LENGTH_MM = 5.9  # Minimum focal length at 1x zoom
DEFAULT_MAX_FOCAL_LENGTH_MM = 147.5  # Maximum focal length at 25x zoom
DEFAULT_MAX_ZOOM = 25.0  # Maximum optical zoom factor

# Camera configurations
CAMERAS = [
    {
        "ip": "10.207.99.178",
        "name": "Valte",
        "model": "DS-2DF8425IX-AELW(T5)",
        "lat": "39°38'25.72\"N",
        "lon": "0°13'48.63\"W",
        "height_m": 4.71,  # Calibrated 2025-12-11 with comprehensive_calibration.py
        # Pan offset: angle from North when camera reports pan=0
        # True bearing = reported_pan + pan_offset_deg
        # Calibration: Point camera at known landmark, calculate true bearing,
        # then pan_offset = true_bearing - reported_pan
        "pan_offset_deg": 51.7,  # Calibrated 2025-12-11 (was 65°, optimized to 51.7°)
        # Tilt offset: correction for reported tilt angle
        # Effective tilt = reported_tilt + tilt_offset_deg
        # Calibrated by minimizing GCP projection error
        "tilt_offset_deg": -0.25,  # Calibrated 2025-12-11 (camera reports ~0.25° higher than actual)
        # Lens distortion coefficients (OpenCV model)
        # Calibrated using checkerboard or GCP-based optimization
        "k1": -0.341052,  # Radial distortion (negative = barrel distortion)
        "k2": 0.787571,   # Secondary radial distortion
        "p1": 0.0,        # Tangential distortion (not calibrated)
        "p2": 0.0,        # Tangential distortion (not calibrated)
        # Sensor/lens parameters (use defaults if not specified)
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "description": "Valte camera location"
    },
    {
        "ip": "10.237.100.15",
        "name": "Setram",
        "model": "DS-2DF8425IX-AELW(T5)",  # Assumed same model
        "lat": "41°19'46.8\"N",
        "lon": "2°08'31.3\"E",
        "height_m": 5.0,  # Default height, calibrate with GPS validation
        "pan_offset_deg": 0.0,  # Pan=0 points north (default, needs calibration)
        "tilt_offset_deg": 0.0,  # Default, needs calibration
        # Distortion not calibrated yet
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "base_focal_length_mm": DEFAULT_BASE_FOCAL_LENGTH_MM,
        "description": "Setram camera location"
    },
]


def get_camera_configs() -> list:
    """
    Get list of all camera configurations.

    Returns:
        List of camera configuration dicts containing camera parameters,
        GPS coordinates, and calibration data. Does not require credentials
        and does not generate RTSP URLs.
    """
    return CAMERAS


def get_camera_by_name(camera_name: str) -> dict:
    """
    Find camera configuration by name.

    Args:
        camera_name: Name of the camera (e.g., "Valte", "Setram")

    Returns:
        Camera configuration dict or None if not found
    """
    return next((cam for cam in CAMERAS if cam.get("name") == camera_name), None)


def get_camera_by_name_safe(camera_name: str) -> dict:
    """
    Alias for get_camera_by_name().

    This function exists for backwards compatibility. Since credential validation
    was moved from module-level to get_rtsp_url(), get_camera_by_name() is now
    safe to call without credentials. Both functions are equivalent.

    Args:
        camera_name: Name of the camera (e.g., "Valte", "Setram")

    Returns:
        Camera configuration dict or None if not found
    """
    return get_camera_by_name(camera_name)


def get_camera_gps(camera_name: str) -> dict:
    """
    Get GPS coordinates for a camera.

    Args:
        camera_name: Name of the camera

    Returns:
        {"lat": "...", "lon": "..."} or None if not found
    """
    cam = get_camera_by_name(camera_name)
    if cam:
        return {"lat": cam["lat"], "lon": cam["lon"]}
    return None


def get_rtsp_url(camera_name: str, stream_type: str = "main") -> str:
    """
    Get RTSP URL for a camera.

    Args:
        camera_name: Name of the camera
        stream_type: "main" (101) or "sub" (102)

    Returns:
        Full RTSP URL or None if camera not found

    Raises:
        ValueError: If camera credentials are not set
    """
    # Validate that credentials are set
    if not USERNAME or not PASSWORD:
        raise ValueError(
            "Camera credentials not set. Please set CAMERA_USERNAME and CAMERA_PASSWORD "
            "environment variables. See .env.example for template."
        )

    cam = get_camera_by_name(camera_name)
    if not cam:
        return None

    channel = "101" if stream_type == "main" else "102"
    return f"rtsp://{USERNAME}:{PASSWORD}@{cam['ip']}:554/Streaming/Channels/{channel}"


# Validation
if __name__ == "__main__":
    print("Camera Configuration")
    print("=" * 70)
    print(f"\nCredentials: {USERNAME} / {'*' * len(PASSWORD)}")
    print(f"\nConfigured Cameras: {len(CAMERAS)}")

    for cam in CAMERAS:
        print(f"\n{cam['name']}:")
        print(f"  IP: {cam['ip']}")
        print(f"  GPS: {cam['lat']}, {cam['lon']}")
        print(f"  Height: {cam['height_m']}m")
        print(f"  RTSP: {get_rtsp_url(cam['name'])}")
