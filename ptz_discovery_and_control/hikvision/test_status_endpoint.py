#!/usr/bin/env python3
"""Test the status endpoint with the fixed namespace handling"""

import os
import sys
import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET

# Load credentials from environment variables
USERNAME = os.environ.get("CAMERA_USERNAME")
PASSWORD = os.environ.get("CAMERA_PASSWORD")

if not USERNAME or not PASSWORD:
    print("ERROR: Camera credentials not configured!", file=sys.stderr)
    print("Please set CAMERA_USERNAME and CAMERA_PASSWORD environment variables", file=sys.stderr)
    sys.exit(1)

ip = "10.207.99.178"
url = f"http://{ip}/ISAPI/PTZCtrl/channels/1/status"

response = requests.get(url, auth=HTTPDigestAuth(USERNAME, PASSWORD), timeout=5)
print(f"Status code: {response.status_code}")

if response.status_code == 200:
    root = ET.fromstring(response.text)
    ns = {'h': 'http://www.hikvision.com/ver20/XMLSchema'}

    azimuth = root.find('.//h:azimuth', ns)
    elevation = root.find('.//h:elevation', ns)
    abs_zoom = root.find('.//h:absoluteZoom', ns)

    if azimuth is not None:
        pan = float(azimuth.text) / 10
        tilt = float(elevation.text) / 10
        zoom = float(abs_zoom.text) / 10

        print(f"\n✓ Parsed successfully!")
        print(f"Pan (azimuth): {pan:.1f}° (raw: {azimuth.text})")
        print(f"Tilt (elevation): {tilt:.1f}° (raw: {elevation.text})")
        print(f"Zoom: {zoom:.1f} (raw: {abs_zoom.text})")
    else:
        print("✗ Failed to find elements with namespace")
