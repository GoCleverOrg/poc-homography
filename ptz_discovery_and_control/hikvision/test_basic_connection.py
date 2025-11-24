#!/usr/bin/env python3
"""Quick test to check basic connectivity to Hikvision cameras"""

import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET

CAMERAS = [
    {"ip": "10.207.99.178", "name": "Camera 1"},
    {"ip": "10.237.100.15", "name": "Camera 2"}
]

USERNAME = "admin"
PASSWORD = "CameraLab01*"

for cam in CAMERAS:
    print(f"\n{'='*60}")
    print(f"Testing {cam['name']} ({cam['ip']})")
    print(f"{'='*60}")

    url = f"http://{cam['ip']}/ISAPI/PTZCtrl/channels/1/status"
    print(f"URL: {url}")

    try:
        print("Attempting connection...")
        response = requests.get(url, auth=HTTPDigestAuth(USERNAME, PASSWORD), timeout=5)
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            print(f"\n✓ SUCCESS!\n")
            print("Response content:")
            print(response.text[:500])

            # Try to parse XML
            print("\nParsing XML...")
            root = ET.fromstring(response.text)
            print(f"Root tag: {root.tag}")

            # Print all elements
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    print(f"  {elem.tag}: {elem.text}")
        else:
            print(f"✗ Failed with status: {response.status_code}")
            print(f"Response: {response.text[:200]}")

    except Exception as e:
        print(f"✗ Exception: {e}")
