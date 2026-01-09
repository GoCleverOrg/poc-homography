#!/usr/bin/env python3
"""CLI for interactive calibration tool."""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.interactive_calibration import (
    CAMERA_AVAILABLE,
    CV2_AVAILABLE,
    CalibrationSession,
    run_batch_calibration,
    run_interactive_session,
)

if CV2_AVAILABLE:
    import cv2

if CAMERA_AVAILABLE:
    from tools.get_camera_intrinsics import get_ptz_status
    from tools.interactive_calibration import grab_frame

    from poc_homography.camera_config import get_camera_by_name
else:
    from poc_homography.camera_config import get_camera_by_name_safe

from poc_homography.gps_distance_calculator import dms_to_dd


def main():
    parser = argparse.ArgumentParser(
        description="Interactive calibration tool for GPS-to-image projection"
    )
    parser.add_argument("--camera", "-c", type=str, required=True, help="Camera name (e.g., Valte)")
    parser.add_argument(
        "--frame",
        "-f",
        type=str,
        help="Path to saved frame image (optional, uses live camera if not provided)",
    )
    parser.add_argument(
        "--pan-raw",
        type=float,
        default=0.0,
        help="Raw pan value from camera (default: 0, or from live camera)",
    )
    parser.add_argument(
        "--tilt", type=float, default=30.0, help="Tilt angle (default: 30, or from live camera)"
    )
    parser.add_argument(
        "--zoom", type=float, default=1.0, help="Zoom factor (default: 1.0, or from live camera)"
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        help="JSON file with pre-defined reference points for batch mode",
    )

    args = parser.parse_args()

    # Get camera config from canonical source
    if CAMERA_AVAILABLE:
        cam_config = get_camera_by_name(args.camera)
    else:
        # Fallback to safe accessor when camera modules aren't available
        cam_config = get_camera_by_name_safe(args.camera)

    if cam_config is None:
        print(f"Error: Camera '{args.camera}' not found")
        sys.exit(1)

    # Convert DMS strings to decimal degrees
    camera_lat = dms_to_dd(cam_config["lat"])
    camera_lon = dms_to_dd(cam_config["lon"])
    height_m = cam_config.get("height_m", 5.0)
    pan_offset_deg = cam_config.get("pan_offset_deg", 0.0)

    # Get frame
    pan_raw = args.pan_raw
    tilt_deg = args.tilt
    zoom = args.zoom

    if args.frame:
        if not CV2_AVAILABLE:
            print("Error: OpenCV required to load frame")
            sys.exit(1)
        frame = cv2.imread(args.frame)
        if frame is None:
            print(f"Error: Could not load frame from {args.frame}")
            sys.exit(1)
        print(f"Loaded frame from {args.frame}")
    elif CAMERA_AVAILABLE:
        print(f"Grabbing frame from camera {args.camera}...")
        try:
            frame = grab_frame(
                cam_config["ip"],
                os.environ.get("CAMERA_USERNAME", "admin"),
                os.environ.get("CAMERA_PASSWORD", ""),
            )
            ptz = get_ptz_status(
                cam_config["ip"],
                os.environ.get("CAMERA_USERNAME", "admin"),
                os.environ.get("CAMERA_PASSWORD", ""),
            )
            pan_raw = ptz["pan"]
            tilt_deg = ptz["tilt"]
            zoom = ptz["zoom"]
            print(f"Got PTZ: pan={pan_raw:.1f}, tilt={tilt_deg:.1f}, zoom={zoom:.1f}x")
        except Exception as e:
            print(f"Error: Could not connect to camera: {e}")
            sys.exit(1)
    else:
        print("Error: Either --frame or live camera connection is required")
        sys.exit(1)

    # Create session
    session = CalibrationSession(
        camera_name=args.camera,
        frame=frame,
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        height_m=height_m,
        pan_offset_deg=pan_offset_deg,
        pan_raw=pan_raw,
        tilt_deg=tilt_deg,
        zoom=zoom,
    )

    # Run calibration
    if args.reference_file:
        with open(args.reference_file) as f:
            ref_data = json.load(f)
        run_batch_calibration(session, ref_data["reference_points"])
    else:
        run_interactive_session(session)


if __name__ == "__main__":
    main()
