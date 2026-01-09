#!/usr/bin/env python3
"""Test script for the new PTZ command helpers.

By default this script will:
 - connect to the first camera in the CAMERAS list
 - fetch its current PTZ status
 - call enviar_comando_ptz_volver with the current status (safe no-op)

The 3D position/zoom command is potentially moving; it will only be run if the
environment variable RUN_3D is set to '1'. This prevents accidental camera movement.
"""

import os
import sys
import time

# Add parent directory to path to import ptz_discovery_and_control
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import (
    CAMERAS,
    PASSWORD,
    USERNAME,
    HikvisionPTZ,
)


def main():
    cam_info = CAMERAS[0]
    camera = HikvisionPTZ(
        ip=cam_info["ip"], username=USERNAME, password=PASSWORD, name=cam_info.get("name", "Camera")
    )

    print(f"Connecting to {camera.name} ({camera.ip})")
    status = camera.get_status()
    if not status:
        print("Failed to get status; aborting test.")
        return

    print("Current status:")
    print(f"  pan={status.get('pan')}, tilt={status.get('tilt')}, zoom={status.get('zoom')}")

    # Safe test: send PTZ absolute to the same position (expected to be a no-op)
    print("Sending send_ptz_return (return to same position)...")
    ok = camera.send_ptz_return(status)
    print("Result:", "OK" if ok else "FAILED")

    # Optional: 3D zoom/position test — only run when explicitly allowed
    run_3d = os.environ.get("RUN_3D", "0") == "1"
    if run_3d:
        # For the valte camera this will zoom in on the light pole to the left
        status["pan"] = 56.5
        status["tilt"] = -3.7
        status["zoom"] = 1.0

        ok = camera.send_ptz_return(status)
        time.sleep(5)  # Wait for camera to complete movement
        print("RUN_3D=1 detected — running 3D position/zoom command (may move the camera).")
        start_x = 34
        start_y = 223
        end_x = 60
        end_y = 178
        code, resp = camera.send_3d_zoom_command(start_x, start_y, end_x, end_y)
        print(f"3D command returned {code}")
    else:
        print("Skipping 3D command. To enable, set RUN_3D=1 in the environment.")


if __name__ == "__main__":
    main()
