"""
Camera configuration file.
Central location for all camera settings and credentials.
"""

import numpy as np
from typing import Optional, Tuple

# Camera credentials
USERNAME = "admin"
PASSWORD = "CameraLab01*"

# Camera configurations
# distortion_coeffs format: [k1, k2, p1, p2, k3] following OpenCV distortion model
#   - k1, k2, k3: radial distortion coefficients
#   - p1, p2: tangential distortion coefficients
#   - None: camera is uncalibrated (zero distortion assumed)
CAMERAS = [
    {
        "ip": "10.207.99.178",
        "name": "Valte",
        "lat": "39°38'25.7\"N",
        "lon": "0°13'48.7\"W",
        "height_m": 11.3,  # Default height, calibrate with GPS validation
        "description": "Valte camera location",
        "distortion_coeffs": None  # Uncalibrated - zero distortion assumed
    },
    {
        "ip": "10.237.100.15",
        "name": "Setram",
        "lat": "41°19'46.8\"N",
        "lon": "2°08'31.3\"E",
        "height_m": 5.0,  # Default height, calibrate with GPS validation
        "description": "Setram camera location",
        "distortion_coeffs": None  # Uncalibrated - zero distortion assumed
    },
]


def get_camera_by_name(camera_name: str) -> dict:
    """
    Find camera configuration by name.

    Args:
        camera_name: Name of the camera (e.g., "Valte", "Setram")

    Returns:
        Camera configuration dict or None if not found
    """
    return next((cam for cam in CAMERAS if cam.get("name") == camera_name), None)


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
    """
    cam = get_camera_by_name(camera_name)
    if not cam:
        return None

    channel = "101" if stream_type == "main" else "102"
    return f"rtsp://{USERNAME}:{PASSWORD}@{cam['ip']}:554/Streaming/Channels/{channel}"


def get_camera_distortion(camera_name: str) -> Optional[np.ndarray]:
    """
    Returns distortion coefficients for a camera, or None if uncalibrated.

    First checks the camera config for distortion_coeffs. If None, attempts to
    load from persistent calibration storage.

    Args:
        camera_name: Name of the camera

    Returns:
        numpy array of distortion coefficients [k1, k2, p1, p2, k3, ...]
        or None if uncalibrated and no stored calibration found
        - k1, k2, k3: radial distortion coefficients
        - p1, p2: tangential distortion coefficients
    """
    cam = get_camera_by_name(camera_name)
    if not cam:
        return None

    distortion = cam.get("distortion_coeffs")

    # If distortion is defined in config, use it
    if distortion is not None:
        # Convert to numpy array if it's a list
        if isinstance(distortion, list):
            return np.array(distortion, dtype=np.float64)
        return distortion

    # If distortion is None in config, try loading from storage
    try:
        from calibration_storage import load_calibration

        calib_data = load_calibration(camera_name)
        if calib_data:
            print(f"Loaded distortion coefficients for '{camera_name}' from calibration storage")
            return calib_data['distortion_coeffs']
    except ImportError:
        # calibration_storage module not available
        pass
    except Exception as e:
        print(f"Warning: Failed to load calibration from storage for '{camera_name}': {e}")

    # No distortion data available
    return None


def get_camera_calibration(camera_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns full calibration data (camera matrix and distortion coefficients) for a camera.

    Loads calibration from persistent storage if available.

    Args:
        camera_name: Name of the camera

    Returns:
        Tuple of (camera_matrix, distortion_coeffs) or None if no calibration found
        - camera_matrix: 3x3 intrinsic camera matrix K
        - distortion_coeffs: distortion coefficients array [k1, k2, p1, p2, k3, ...]
    """
    try:
        from calibration_storage import load_calibration

        calib_data = load_calibration(camera_name)
        if calib_data:
            return (calib_data['camera_matrix'], calib_data['distortion_coeffs'])
    except ImportError:
        # calibration_storage module not available
        pass
    except Exception as e:
        print(f"Warning: Failed to load calibration from storage for '{camera_name}': {e}")

    return None


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
