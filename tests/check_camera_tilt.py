#!/usr/bin/env python3
"""
Quick script to check camera tilt values.
"""

from __future__ import annotations

from poc_homography.camera_config import PASSWORD, USERNAME, get_camera_configs
from poc_homography.infrastructure.clients.hikvision.isapi_client import (
    HikvisionISAPIClient,
)

print("Checking camera tilt angles...")
print("=" * 60)

for cam_info in get_camera_configs():
    camera = HikvisionISAPIClient(cam_info["ip"], USERNAME, PASSWORD)

    status = camera.get_ptz_status()

    print(f"\n{cam_info['name']} Camera:")
    print(f"  IP: {cam_info['ip']}")
    print(f"  Pan:  {status.pan_raw:.2f}°")
    print(f"  Tilt: {status.tilt_deg:.2f}°")
    print(f"  Zoom: {status.zoom:.2f}x")

    if status.tilt_deg > 0:
        print(f"  ⚠️  WARNING: Tilt is POSITIVE ({status.tilt_deg:.2f}°)")
        print("      Camera is pointing UPWARD!")
        print("      For ground plane homography, tilt must be NEGATIVE")
        print("      (camera must point downward)")
    elif status.tilt_deg > -10:
        print(f"  ⚠️  WARNING: Tilt is nearly horizontal ({status.tilt_deg:.2f}°)")
        print("      Recommended: tilt < -30° for good homography")
    else:
        print("  ✓  Tilt is negative (camera pointing down)")

print("\n" + "=" * 60)
print("\nRECOMMENDATION:")
print("  For homography to work, cameras should:")
print("  • Point DOWNWARD (tilt < -10°)")
print("  • Ideally tilt between -30° and -60°")
print("  • See ground in bottom portion of image")
print("\n" + "=" * 60)
