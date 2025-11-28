"""
Camera configuration file.
Central location for all camera settings and credentials.

Credentials are loaded from environment variables:
- CAMERA_USERNAME: Camera login username (required)
- CAMERA_PASSWORD: Camera login password (required)

Set these in your environment or use a .env file (see .env.example for template).
"""

import os
import sys

# Camera credentials - Load from environment variables
USERNAME = os.environ.get("CAMERA_USERNAME")
PASSWORD = os.environ.get("CAMERA_PASSWORD")

# Validate that credentials are set
if not USERNAME or not PASSWORD:
    print("ERROR: Camera credentials not configured!", file=sys.stderr)
    print("", file=sys.stderr)
    print("Please set the following environment variables:", file=sys.stderr)
    print("  - CAMERA_USERNAME: Camera login username", file=sys.stderr)
    print("  - CAMERA_PASSWORD: Camera login password", file=sys.stderr)
    print("", file=sys.stderr)
    print("You can set them in your shell:", file=sys.stderr)
    print("  export CAMERA_USERNAME=your_username", file=sys.stderr)
    print("  export CAMERA_PASSWORD=your_password", file=sys.stderr)
    print("", file=sys.stderr)
    print("Or create a .env file (see .env.example for template)", file=sys.stderr)
    sys.exit(1)

# Camera configurations
CAMERAS = [
    {
        "ip": "10.207.99.178",
        "name": "Valte",
        "lat": "39°38'25.7\"N",
        "lon": "0°13'48.7\"W",
        "height_m": 11.3,  # Default height, calibrate with GPS validation
        "description": "Valte camera location"
    },
    {
        "ip": "10.237.100.15",
        "name": "Setram",
        "lat": "41°19'46.8\"N",
        "lon": "2°08'31.3\"E",
        "height_m": 5.0,  # Default height, calibrate with GPS validation
        "description": "Setram camera location"
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
