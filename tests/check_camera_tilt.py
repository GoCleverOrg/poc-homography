#!/usr/bin/env python3
"""
Quick script to check camera tilt values.
"""

from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ
from poc_homography.camera_config import CAMERAS, USERNAME, PASSWORD

print("Checking camera tilt angles...")
print("="*60)

for cam_info in CAMERAS:
    camera = HikvisionPTZ(
        ip=cam_info["ip"],
        username=USERNAME,
        password=PASSWORD,
        name=cam_info["name"]
    )

    status = camera.get_status()

    print(f"\n{cam_info['name']} Camera:")
    print(f"  IP: {cam_info['ip']}")
    print(f"  Pan:  {status['pan']:.2f}°")
    print(f"  Tilt: {status['tilt']:.2f}°")
    print(f"  Zoom: {status['zoom']:.2f}x")

    if status['tilt'] > 0:
        print(f"  ⚠️  WARNING: Tilt is POSITIVE ({status['tilt']:.2f}°)")
        print(f"      Camera is pointing UPWARD!")
        print(f"      For ground plane homography, tilt must be NEGATIVE")
        print(f"      (camera must point downward)")
    elif status['tilt'] > -10:
        print(f"  ⚠️  WARNING: Tilt is nearly horizontal ({status['tilt']:.2f}°)")
        print(f"      Recommended: tilt < -30° for good homography")
    else:
        print(f"  ✓  Tilt is negative (camera pointing down)")

print("\n" + "="*60)
print("\nRECOMMENDATION:")
print("  For homography to work, cameras should:")
print("  • Point DOWNWARD (tilt < -10°)")
print("  • Ideally tilt between -30° and -60°")
print("  • See ground in bottom portion of image")
print("\n" + "="*60)
