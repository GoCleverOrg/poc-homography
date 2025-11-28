#!/usr/bin/env python3
"""
Hikvision PTZ Camera Discovery and Control Script

This script discovers and tests Hikvision ISAPI endpoints for PTZ control.
It queries camera status and tests movement commands.

Credentials are loaded from environment variables:
- CAMERA_USERNAME: Camera login username (required)
- CAMERA_PASSWORD: Camera login password (required)
"""

import os
import sys
import requests
from requests.auth import HTTPDigestAuth
import xml.etree.ElementTree as ET
import time
from typing import Dict, List, Tuple, Optional
import json

# Camera configurations
CAMERAS = [
    {"ip": "10.207.99.178", "name": "Camera 1"},
    {"ip": "10.237.100.15", "name": "Camera 2"}
]

# Load credentials from environment variables
USERNAME = os.environ.get("CAMERA_USERNAME")
PASSWORD = os.environ.get("CAMERA_PASSWORD")

if not USERNAME or not PASSWORD:
    print("ERROR: Camera credentials not configured!", file=sys.stderr)
    print("Please set CAMERA_USERNAME and CAMERA_PASSWORD environment variables", file=sys.stderr)
    print("See .env.example for template", file=sys.stderr)
    sys.exit(1)


class HikvisionPTZ:
    """Class to interact with Hikvision PTZ cameras via ISAPI"""

    def __init__(self, ip: str, username: str, password: str, name: str = "Camera"):
        self.ip = ip
        self.username = username
        self.password = password
        self.name = name
        self.base_url = f"http://{ip}"
        self.auth = HTTPDigestAuth(username, password)

    def _make_request(self, method: str, endpoint: str, data: Optional[str] = None) -> Tuple[int, str]:
        """Make HTTP request to camera"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, auth=self.auth, timeout=10)
            elif method.upper() == "PUT":
                headers = {'Content-Type': 'application/xml'}
                response = requests.put(url, auth=self.auth, data=data, headers=headers, timeout=10)
            else:
                return -1, f"Unsupported method: {method}"

            return response.status_code, response.text
        except requests.exceptions.RequestException as e:
            return -1, str(e)

    def get_status(self) -> Optional[Dict[str, float]]:
        """Get current PTZ status (pan, tilt, zoom)"""
        endpoint = "/ISAPI/PTZCtrl/channels/1/status"
        status_code, response = self._make_request("GET", endpoint)

        if status_code == 200:
            try:
                root = ET.fromstring(response)
                # Handle namespace - Hikvision uses xmlns
                ns = {'h': 'http://www.hikvision.com/ver20/XMLSchema'}

                # Try with namespace first, fall back to no namespace
                azimuth = root.find('.//h:azimuth', ns)
                if azimuth is None:
                    azimuth = root.find('.//azimuth')

                elevation = root.find('.//h:elevation', ns)
                if elevation is None:
                    elevation = root.find('.//elevation')

                abs_zoom = root.find('.//h:absoluteZoom', ns)
                if abs_zoom is None:
                    abs_zoom = root.find('.//absoluteZoom')

                result = {
                    'pan': float(azimuth.text) / 10 if azimuth is not None else None,
                    'tilt': float(elevation.text) / 10 if elevation is not None else None,
                    'zoom': float(abs_zoom.text) / 10 if abs_zoom is not None else None,
                    'raw_xml': response
                }
                return result
            except Exception as e:
                print(f"Error parsing status: {e}")
                print(f"Response: {response}")
                return None
        else:
            print(f"Failed to get status. Status code: {status_code}")
            print(f"Response: {response}")
            return None

    def discover_endpoints(self) -> Dict[str, Tuple[int, str]]:
        """Discover available PTZ endpoints"""
        endpoints_to_test = [
            # Status endpoints
            ("/ISAPI/PTZCtrl/channels/1/status", "GET"),
            ("/ISAPI/PTZCtrl/channels/1/capabilities", "GET"),

            # Control endpoints
            ("/ISAPI/PTZCtrl/channels/1/continuous", "PUT"),
            ("/ISAPI/PTZCtrl/channels/1/momentary", "PUT"),
            ("/ISAPI/PTZCtrl/channels/1/relative", "PUT"),
            ("/ISAPI/PTZCtrl/channels/1/absolute", "PUT"),

            # Preset endpoints
            ("/ISAPI/PTZCtrl/channels/1/presets", "GET"),
            ("/ISAPI/PTZCtrl/channels/1/presets/1/goto", "PUT"),

            # Other control
            ("/ISAPI/PTZCtrl/channels/1/auxcontrols/1", "PUT"),
            ("/ISAPI/PTZCtrl/channels/1/homeposition/goto", "PUT"),
        ]

        results = {}
        print(f"\n{'='*60}")
        print(f"Discovering endpoints for {self.name} ({self.ip})")
        print(f"{'='*60}\n")

        for endpoint, method in endpoints_to_test:
            print(f"Testing: {method:6} {endpoint}")

            # For PUT requests, we need to send minimal XML data
            if method == "PUT":
                # Test with empty/minimal XML to see if endpoint exists
                test_data = '<?xml version="1.0" encoding="UTF-8"?><PTZData></PTZData>'
                status_code, response = self._make_request(method, endpoint, test_data)
            else:
                status_code, response = self._make_request(method, endpoint)

            results[endpoint] = (status_code, response[:200] if len(response) > 200 else response)

            if status_code == 200:
                print(f"  ✓ SUCCESS (200)")
            elif status_code == 400:
                print(f"  ⚠ Bad Request (400) - Endpoint exists but needs valid data")
            elif status_code == 404:
                print(f"  ✗ Not Found (404)")
            else:
                print(f"  ? Status: {status_code}")

            time.sleep(0.2)  # Small delay between requests

        return results

    def move_continuous(self, pan: int = 0, tilt: int = 0, zoom: int = 0) -> bool:
        """
        Move camera continuously
        pan: -100 to 100 (negative = left, positive = right)
        tilt: -100 to 100 (negative = down, positive = up)
        zoom: -100 to 100 (negative = zoom out, positive = zoom in)
        """
        endpoint = "/ISAPI/PTZCtrl/channels/1/continuous"
        xml_data = f'''<?xml version="1.0" encoding="UTF-8"?>
<PTZData>
    <pan>{pan}</pan>
    <tilt>{tilt}</tilt>
    <zoom>{zoom}</zoom>
</PTZData>'''

        status_code, response = self._make_request("PUT", endpoint, xml_data)
        return status_code == 200

    def stop_movement(self) -> bool:
        """Stop all PTZ movement"""
        return self.move_continuous(pan=0, tilt=0, zoom=0)

    def send_3d_zoom_command(self, x_start: float, y_start: float, x_end: float, y_end: float) -> Tuple[int, str]:
        """Send a 3D position/zoom command to the camera (position3D endpoint).

        Returns a tuple of (status_code, response_text).
        Note: this will send a PUT request that may move the camera. Use with caution.
        """
        endpoint = "/ISAPI/PTZCtrl/channels/1/position3D"
        body = f'''<Position3D version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <StartPoint>
        <positionX>{x_start}</positionX>
        <positionY>{y_start}</positionY>
    </StartPoint>
    <EndPoint>
        <positionX>{x_end}</positionX>
        <positionY>{y_end}</positionY>
    </EndPoint>
</Position3D>'''

        status_code, response = self._make_request("PUT", endpoint, body)
        if status_code == 200:
            print("3D zoom/position command accepted (200).")
        else:
            print(f"3D zoom/position command returned {status_code}: {response[:200]}")
        return status_code, response

    def send_ptz_return(self, status: Dict[str, float]) -> bool:
        """Send an absolute PTZ command to return to a given status.

        Accepts the dictionary returned by `get_status()` (keys: 'pan','tilt','zoom').
        The camera expects integer values in its native units (we multiply by 10 to reverse
        the division used when parsing status).
        Returns True on HTTP 200.
        """
        endpoint = "/ISAPI/PTZCtrl/channels/1/absolute"

        # If status values are None, fallback to 0
        pan = int(status.get('pan', 0) * 10) if status.get('pan') is not None else 0
        tilt = int(status.get('tilt', 0) * 10) if status.get('tilt') is not None else 0
        zoom = int(status.get('zoom', 0) * 10) if status.get('zoom') is not None else 0

        body = f'''<PTZData version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
    <AbsoluteHigh>
        <elevation>{tilt}</elevation>
        <azimuth>{pan}</azimuth>
        <absoluteZoom>{zoom}</absoluteZoom>
    </AbsoluteHigh>
</PTZData>'''

        status_code, response = self._make_request("PUT", endpoint, body)
        if status_code == 200:
            print("PTZ command sent successfully.")
            return True
        else:
            print(f"Error sending PTZ command: {status_code}, Response: {response}")
            return False

    # NOTE: Spanish aliases removed in favor of English method names (clean rename).

    def test_movement_with_polling(self, pan: int = 0, tilt: int = 0, zoom: int = 0,
                                   duration: float = 2.0, poll_interval: float = 0.2) -> List[Dict]:
        """
        Test PTZ movement and poll status until stabilization

        Args:
            pan, tilt, zoom: Movement speeds (-100 to 100)
            duration: How long to move (seconds)
            poll_interval: How often to check status (seconds)

        Returns:
            List of status readings over time
        """
        print(f"\n{'='*60}")
        print(f"Testing movement: pan={pan}, tilt={tilt}, zoom={zoom}")
        print(f"Duration: {duration}s, Poll interval: {poll_interval}s")
        print(f"{'='*60}\n")

        status_history = []

        # Get initial position
        print("Getting initial position...")
        initial = self.get_status()
        if initial:
            status_history.append({
                'time': 0,
                'phase': 'initial',
                **initial
            })
            pan_str = f"{initial['pan']:.1f}°" if initial['pan'] is not None else "N/A"
            tilt_str = f"{initial['tilt']:.1f}°" if initial['tilt'] is not None else "N/A"
            zoom_str = f"{initial['zoom']:.1f}" if initial['zoom'] is not None else "N/A"
            print(f"Initial position - Pan: {pan_str}, Tilt: {tilt_str}, Zoom: {zoom_str}")

        # Start movement
        print(f"\nStarting movement...")
        if not self.move_continuous(pan, tilt, zoom):
            print("Failed to start movement!")
            return status_history

        # Poll during movement
        start_time = time.time()
        print("\nMoving...")
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            status = self.get_status()
            if status:
                status_history.append({
                    'time': elapsed,
                    'phase': 'moving',
                    **status
                })
                pan_str = f"{status['pan']:.1f}°" if status['pan'] is not None else "N/A"
                tilt_str = f"{status['tilt']:.1f}°" if status['tilt'] is not None else "N/A"
                zoom_str = f"{status['zoom']:.1f}" if status['zoom'] is not None else "N/A"
                print(f"  [{elapsed:.2f}s] Pan: {pan_str}, Tilt: {tilt_str}, Zoom: {zoom_str}")
            time.sleep(poll_interval)

        # Stop movement
        print("\nStopping movement...")
        self.stop_movement()

        # Poll until stabilized (no change for 3 consecutive readings)
        print("\nWaiting for stabilization...")
        stable_count = 0
        last_position = None
        max_stable_checks = 15

        for i in range(max_stable_checks):
            time.sleep(poll_interval)
            status = self.get_status()
            if status:
                elapsed = time.time() - start_time
                status_history.append({
                    'time': elapsed,
                    'phase': 'stabilizing',
                    **status
                })
                pan_str = f"{status['pan']:.1f}°" if status['pan'] is not None else "N/A"
                tilt_str = f"{status['tilt']:.1f}°" if status['tilt'] is not None else "N/A"
                zoom_str = f"{status['zoom']:.1f}" if status['zoom'] is not None else "N/A"
                print(f"  [{elapsed:.2f}s] Pan: {pan_str}, Tilt: {tilt_str}, Zoom: {zoom_str}")

                # Check if position has stabilized
                if last_position:
                    pan_stable = (status['pan'] is None or last_position['pan'] is None or
                                 abs(status['pan'] - last_position['pan']) < 0.1)
                    tilt_stable = (status['tilt'] is None or last_position['tilt'] is None or
                                  abs(status['tilt'] - last_position['tilt']) < 0.1)
                    zoom_stable = (status['zoom'] is None or last_position['zoom'] is None or
                                  abs(status['zoom'] - last_position['zoom']) < 0.1)

                    if pan_stable and tilt_stable and zoom_stable:
                        stable_count += 1
                        if stable_count >= 3:
                            print(f"\n✓ Position stabilized after {elapsed:.2f}s")
                            break
                    else:
                        stable_count = 0

                last_position = status

        # Get final position
        final = self.get_status()
        if final:
            pan_str = f"{final['pan']:.1f}°" if final['pan'] is not None else "N/A"
            tilt_str = f"{final['tilt']:.1f}°" if final['tilt'] is not None else "N/A"
            zoom_str = f"{final['zoom']:.1f}" if final['zoom'] is not None else "N/A"
            print(f"\nFinal position - Pan: {pan_str}, Tilt: {tilt_str}, Zoom: {zoom_str}")
            if initial and final['pan'] is not None and initial['pan'] is not None:
                delta_pan = final['pan'] - initial['pan']
                delta_tilt = final['tilt'] - initial['tilt'] if final['tilt'] and initial['tilt'] else 0
                delta_zoom = final['zoom'] - initial['zoom'] if final['zoom'] and initial['zoom'] else 0
                print(f"Total change - Pan: {delta_pan:+.1f}°, Tilt: {delta_tilt:+.1f}°, Zoom: {delta_zoom:+.1f}")

        return status_history


def main():
    """Main function to discover and test PTZ cameras"""

    print("\n" + "="*60)
    print("Hikvision PTZ Camera Discovery and Control")
    print("="*60)

    for cam_config in CAMERAS:
        camera = HikvisionPTZ(
            ip=cam_config["ip"],
            username=USERNAME,
            password=PASSWORD,
            name=cam_config["name"]
        )

        # Test basic connectivity and get status
        print(f"\n\n{'#'*60}")
        print(f"# Testing {camera.name} ({camera.ip})")
        print(f"{'#'*60}\n")

        print("Getting current PTZ status...")
        status = camera.get_status()
        if status:
            print(f"✓ Successfully connected to {camera.name}")
            print(f"  Pan: {status['pan']:.1f}° " if status['pan'] is not None else "  Pan: N/A")
            print(f"  Tilt: {status['tilt']:.1f}°" if status['tilt'] is not None else "  Tilt: N/A")
            print(f"  Zoom: {status['zoom']:.1f}" if status['zoom'] is not None else "  Zoom: N/A")
            print(f"\nRaw XML response:")
            print(status['raw_xml'][:500])
        else:
            print(f"✗ Failed to connect to {camera.name}")
            continue

        # Discover endpoints
        endpoints = camera.discover_endpoints()

        # Print summary
        print(f"\n{'='*60}")
        print(f"Endpoint Discovery Summary for {camera.name}")
        print(f"{'='*60}\n")

        working_endpoints = {k: v for k, v in endpoints.items() if v[0] in [200, 400]}
        print(f"Working endpoints (200/400): {len(working_endpoints)}/{len(endpoints)}")
        for endpoint, (status_code, _) in working_endpoints.items():
            print(f"  {endpoint} - Status: {status_code}")

        # Test movements
        print(f"\n\n{'#'*60}")
        print(f"# Testing PTZ Movements on {camera.name}")
        print(f"{'#'*60}")

        movements = [
            {"name": "Pan Right", "pan": 30, "tilt": 0, "zoom": 0},
            {"name": "Pan Left", "pan": -30, "tilt": 0, "zoom": 0},
            {"name": "Tilt Up", "pan": 0, "tilt": 30, "zoom": 0},
            {"name": "Tilt Down", "pan": 0, "tilt": -30, "zoom": 0},
            {"name": "Zoom In", "pan": 0, "tilt": 0, "zoom": 30},
            {"name": "Zoom Out", "pan": 0, "tilt": 0, "zoom": -30},
        ]

        all_results = {}

        for movement in movements:
            print(f"\n\n--- {movement['name']} ---")
            history = camera.test_movement_with_polling(
                pan=movement['pan'],
                tilt=movement['tilt'],
                zoom=movement['zoom'],
                duration=1.5,
                poll_interval=0.3
            )
            all_results[movement['name']] = history
            time.sleep(1)  # Pause between tests

        # Save results to file
        output_file = f"ptz_test_results_{camera.name.replace(' ', '_')}_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'camera': camera.name,
                'ip': camera.ip,
                'endpoints': {k: {'status': v[0], 'response': v[1]} for k, v in endpoints.items()},
                'movements': all_results
            }, f, indent=2)
        print(f"\n\n✓ Results saved to: {output_file}")

        print("\n" + "="*60)
        print(f"Completed testing {camera.name}")
        print("="*60)


if __name__ == "__main__":
    main()
