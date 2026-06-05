#!/usr/bin/env python3
"""Manual smoke script for the PTZ command helpers (not a pytest test).

This module is named ``test_*`` for historical reasons but contains no pytest
test functions; it is a runnable diagnostic script. By default it will:
 - connect to the first camera in the configured CAMERAS list
 - fetch its current PTZ status
 - re-issue an absolute move to the current position (safe no-op)

The 3D position/zoom command is potentially moving; it will only be run if the
environment variable RUN_3D is set to '1'. This prevents accidental camera
movement.
"""

from __future__ import annotations

import os
import time

from poc_homography.camera_config import CAMERAS, PASSWORD, USERNAME
from poc_homography.infrastructure.clients.hikvision.isapi_client import (
    HikvisionISAPIClient,
)


def main() -> None:
    cam_info = CAMERAS[0]
    camera = HikvisionISAPIClient(cam_info["ip"], USERNAME, PASSWORD)

    print(f"Connecting to {cam_info.get('name', 'Camera')} ({cam_info['ip']})")
    status = camera.get_ptz_status()

    print("Current status:")
    print(f"  pan={status.pan_raw}, tilt={status.tilt_deg}, zoom={status.zoom}")

    # Safe test: send PTZ absolute to the same position (expected to be a no-op)
    print("Sending move_absolute (return to same position)...")
    camera.move_absolute(pan=status.pan_raw, tilt=status.tilt_deg, zoom=status.zoom)
    print("Result: OK")

    # Optional: 3D zoom/position test — only run when explicitly allowed
    run_3d = os.environ.get("RUN_3D", "0") == "1"
    if run_3d:
        # For the valte camera this will zoom in on the light pole to the left
        camera.move_absolute(pan=56.5, tilt=-3.7, zoom=1.0)
        time.sleep(5)  # Wait for camera to complete movement
        print("RUN_3D=1 detected — running 3D position/zoom command (may move the camera).")
        camera.position3d(start_x=34, start_y=223, end_x=60, end_y=178)
        print("3D command sent")
    else:
        print("Skipping 3D command. To enable, set RUN_3D=1 in the environment.")


if __name__ == "__main__":
    main()
