#!/usr/bin/env python3
"""
GCP Capture Tool - Interactive Ground Control Point collection.

This utility helps capture Ground Control Points (GCPs) from a live camera feed.
It records pixel coordinates from mouse clicks and allows you to enter GPS
coordinates for each point.

Workflow:
1. Connect to camera and display live feed
2. Press 's' to start GCP capture mode
3. Click on points in the image - enter GPS coordinates when prompted
4. Press 's' again to stop capture and save config
5. Output YAML config with camera context and GCPs

Usage:
    python tools/capture_gcps.py Valte
    python tools/capture_gcps.py Valte --output my_gcps.yaml
    python tools/capture_gcps.py Valte --output-dir config/

Controls:
    s - Start/Stop GCP capture mode
    z - Undo last GCP
    c - Clear all GCPs
    r - Enter outlier review mode (click to remove, 'a' for auto-remove)
    q - Quit (prompts to save if GCPs captured)
    Click - Add GCP at cursor position (in capture mode)
            Remove GCP near cursor (in outlier review mode)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from poc_homography.camera_config import (
        get_camera_by_name, get_rtsp_url, USERNAME, PASSWORD, CAMERAS
    )
    CAMERA_CONFIG_AVAILABLE = True
except (ValueError, ImportError) as e:
    CAMERA_CONFIG_AVAILABLE = False
    print(f"Error: Camera config not available ({e})")
    print("Set CAMERA_USERNAME and CAMERA_PASSWORD environment variables.")
    sys.exit(1)

# Import the intrinsics utility
from tools.get_camera_intrinsics import get_ptz_status, compute_intrinsics

# Import for real-time reprojection error calculation
try:
    from poc_homography.coordinate_converter import gps_to_local_xy
    COORDINATE_CONVERTER_AVAILABLE = True
except ImportError:
    COORDINATE_CONVERTER_AVAILABLE = False

# Default camera parameters
DEFAULT_SENSOR_WIDTH_MM = 7.18
DEFAULT_BASE_FOCAL_LENGTH_MM = 5.9

# Reprojection error thresholds for feedback
REPROJ_ERROR_GOOD = 5.0  # pixels - considered good fit
REPROJ_ERROR_WARNING = 10.0  # pixels - warning threshold
REPROJ_ERROR_BAD = 20.0  # pixels - likely outlier

# Outlier review mode settings
OUTLIER_CLICK_RADIUS = 30  # pixels - click within this radius to select GCP
OUTLIER_AUTO_REMOVE_THRESHOLD = 15.0  # pixels - auto-remove GCPs above this error


class GCPCaptureSession:
    """Interactive GCP capture session."""

    def __init__(
        self,
        camera_name: str,
        sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
        base_focal_length_mm: float = DEFAULT_BASE_FOCAL_LENGTH_MM
    ):
        self.camera_name = camera_name
        self.sensor_width_mm = sensor_width_mm
        self.base_focal_length_mm = base_focal_length_mm

        self.cam_info = get_camera_by_name(camera_name)
        if not self.cam_info:
            available = [c['name'] for c in CAMERAS]
            raise ValueError(
                f"Camera '{camera_name}' not found. "
                f"Available: {', '.join(available)}"
            )

        self.gcps: List[Dict] = []
        self.capture_mode = False
        self.current_frame = None
        self.frozen_frame = None  # Frame captured when entering capture mode
        self.frame_width = 0
        self.frame_height = 0
        self.ptz_status = None
        self.capture_timestamp = None

        # Mouse state
        self.mouse_pos = (0, 0)
        self.pending_click = None

        # Outlier review mode state
        self.outlier_review_mode = False
        self.selected_gcp_index = None  # Index of GCP selected for removal
        self.highlighted_gcp_index = None  # Index of GCP under mouse cursor

        # Homography state for real-time validation
        self.current_homography = None
        self.reference_lat = None
        self.reference_lon = None
        self.last_reproj_errors = []  # Store reprojection errors for display

    def get_camera_context(self) -> dict:
        """Get current camera context (PTZ + intrinsics)."""
        ptz = get_ptz_status(self.cam_info['ip'], USERNAME, PASSWORD)
        self.ptz_status = ptz

        intrinsics = compute_intrinsics(
            zoom=ptz['zoom'],
            image_width=self.frame_width,
            image_height=self.frame_height,
            sensor_width_mm=self.sensor_width_mm,
            base_focal_length_mm=self.base_focal_length_mm,
        )

        return {
            'camera_name': self.camera_name,
            'image_width': self.frame_width,
            'image_height': self.frame_height,
            'ptz_position': {
                'pan': round(ptz['pan'], 1),
                'tilt': round(ptz['tilt'], 1),
                'zoom': round(ptz['zoom'], 1),
            },
            'intrinsics': {
                'focal_length_px': round(intrinsics['focal_length_px'], 2),
                'principal_point': {
                    'cx': round(intrinsics['principal_point']['cx'], 1),
                    'cy': round(intrinsics['principal_point']['cy'], 1),
                },
            },
            'capture_timestamp': self.capture_timestamp or datetime.now().isoformat(),
            'notes': '',
        }

    def update_homography(self) -> Optional[Dict[str, Any]]:
        """Compute homography from current GCPs and return quality metrics.

        Returns:
            Dictionary with homography quality metrics, or None if insufficient GCPs.
        """
        if not COORDINATE_CONVERTER_AVAILABLE:
            return None

        if len(self.gcps) < 4:
            self.current_homography = None
            self.last_reproj_errors = []
            return None

        # Set reference point from camera GPS or first GCP
        if self.reference_lat is None:
            # Use camera GPS if available
            if hasattr(self, 'cam_info') and self.cam_info:
                cam_gps = self.cam_info.get('gps')
                if cam_gps:
                    self.reference_lat = cam_gps.get('latitude')
                    self.reference_lon = cam_gps.get('longitude')

            # Fall back to centroid of GCPs
            if self.reference_lat is None:
                lats = [g['gps']['latitude'] for g in self.gcps]
                lons = [g['gps']['longitude'] for g in self.gcps]
                self.reference_lat = sum(lats) / len(lats)
                self.reference_lon = sum(lons) / len(lons)

        # Extract points
        image_points = []
        local_points = []

        for gcp in self.gcps:
            u, v = gcp['image']['u'], gcp['image']['v']
            lat, lon = gcp['gps']['latitude'], gcp['gps']['longitude']

            image_points.append([u, v])
            x, y = gps_to_local_xy(self.reference_lat, self.reference_lon, lat, lon)
            local_points.append([x, y])

        image_points = np.array(image_points, dtype=np.float32)
        local_points = np.array(local_points, dtype=np.float32)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(local_points, image_points, cv2.RANSAC, 5.0)

        if H is None:
            self.current_homography = None
            self.last_reproj_errors = []
            return None

        self.current_homography = H

        # Calculate reprojection errors for all points
        projected = cv2.perspectiveTransform(
            local_points.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        errors = np.linalg.norm(projected - image_points, axis=1)
        self.last_reproj_errors = errors.tolist()

        # Calculate metrics
        num_inliers = int(np.sum(mask)) if mask is not None else 0
        inlier_errors = errors[mask.ravel() == 1] if mask is not None else errors

        return {
            'num_gcps': len(self.gcps),
            'num_inliers': num_inliers,
            'inlier_ratio': num_inliers / len(self.gcps),
            'mean_error_px': float(np.mean(errors)),
            'max_error_px': float(np.max(errors)),
            'inlier_mean_error_px': float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else 0,
            'errors': self.last_reproj_errors,
            'inlier_mask': mask.flatten().tolist() if mask is not None else None
        }

    def predict_new_gcp_error(self, pixel_coords: tuple, gps_coords: tuple) -> Optional[float]:
        """Predict reprojection error for a new GCP before adding it.

        Args:
            pixel_coords: (u, v) pixel coordinates
            gps_coords: (latitude, longitude) GPS coordinates

        Returns:
            Predicted reprojection error in pixels, or None if cannot compute.
        """
        if self.current_homography is None or not COORDINATE_CONVERTER_AVAILABLE:
            return None

        if self.reference_lat is None or self.reference_lon is None:
            return None

        try:
            lat, lon = gps_coords
            x, y = gps_to_local_xy(self.reference_lat, self.reference_lon, lat, lon)

            # Project through current homography
            local_pt = np.array([[x, y]], dtype=np.float32)
            projected = cv2.perspectiveTransform(
                local_pt.reshape(-1, 1, 2), self.current_homography
            ).reshape(2)

            # Calculate error
            u, v = pixel_coords
            error = np.sqrt((projected[0] - u)**2 + (projected[1] - v)**2)
            return float(error)

        except Exception:
            return None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN and self.capture_mode:
            self.pending_click = (x, y)

    def prompt_gps_coordinates(self, pixel_coords: tuple) -> Optional[Dict]:
        """Prompt user for GPS coordinates in terminal."""
        u, v = pixel_coords

        print(f"\n{'='*50}")
        print(f"GCP #{len(self.gcps) + 1} - Pixel: ({u}, {v})")
        print(f"{'='*50}")
        print("Enter GPS coordinates for this point.")
        print("Format: latitude, longitude (decimal degrees)")
        print("Example: 39.640583, -0.230194")
        print("Press Enter with empty input to cancel.")
        print()

        try:
            coords_input = input("GPS coordinates: ").strip()

            if not coords_input:
                print("Cancelled.")
                return None

            # Parse coordinates
            parts = coords_input.replace(',', ' ').split()
            if len(parts) < 2:
                print("Error: Need both latitude and longitude.")
                return None

            latitude = float(parts[0])
            longitude = float(parts[1])

            # Validate ranges
            if not (-90 <= latitude <= 90):
                print(f"Error: Latitude {latitude} out of range [-90, 90]")
                return None
            if not (-180 <= longitude <= 180):
                print(f"Error: Longitude {longitude} out of range [-180, 180]")
                return None

            # Optional: elevation
            elevation = None
            if len(parts) >= 3:
                elevation = float(parts[2])

            # Optional: description
            description = input("Description (optional): ").strip()
            if not description:
                description = f"GCP {len(self.gcps) + 1}"

            # Optional: accuracy
            print("Accuracy level: (h)igh, (m)edium, (l)ow [default: medium]")
            accuracy_input = input("Accuracy: ").strip().lower()
            accuracy_map = {'h': 'high', 'm': 'medium', 'l': 'low', '': 'medium'}
            accuracy = accuracy_map.get(accuracy_input, 'medium')

            gcp = {
                'gps': {
                    'latitude': latitude,
                    'longitude': longitude,
                },
                'image': {
                    'u': float(u),
                    'v': float(v),
                },
                'metadata': {
                    'description': description,
                    'accuracy': accuracy,
                    'timestamp': datetime.now().isoformat(),
                },
            }

            if elevation is not None:
                gcp['gps']['elevation'] = elevation

            # Predict reprojection error for this new GCP
            predicted_error = self.predict_new_gcp_error(
                (u, v), (latitude, longitude)
            )

            print(f"\n✓ GCP added: {description}")
            print(f"  GPS: ({latitude:.6f}, {longitude:.6f})")
            print(f"  Pixel: ({u}, {v})")

            if predicted_error is not None:
                if predicted_error < REPROJ_ERROR_GOOD:
                    error_status = "✓ GOOD"
                    color_hint = "green"
                elif predicted_error < REPROJ_ERROR_WARNING:
                    error_status = "⚠ WARNING"
                    color_hint = "yellow"
                else:
                    error_status = "✗ HIGH ERROR"
                    color_hint = "red"
                print(f"  Predicted error: {predicted_error:.1f}px ({error_status})")

                if predicted_error >= REPROJ_ERROR_WARNING:
                    print(f"  >> This GCP may be an outlier. Consider:")
                    print(f"     - Verifying the GPS coordinates are correct")
                    print(f"     - Re-clicking the pixel location more precisely")
                    confirm = input("  Add anyway? (y/n) [n]: ").strip().lower()
                    if confirm != 'y':
                        print("  GCP discarded.")
                        return None

            return gcp

        except ValueError as e:
            print(f"Error parsing coordinates: {e}")
            return None
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw GCPs and UI overlay on frame."""
        display = frame.copy()

        # Draw existing GCPs with color-coded reprojection error
        for i, gcp in enumerate(self.gcps):
            u = int(gcp['image']['u'])
            v = int(gcp['image']['v'])
            desc = gcp.get('metadata', {}).get('description', f'GCP {i+1}')

            # Get reprojection error for this GCP if available
            reproj_error = None
            if i < len(self.last_reproj_errors):
                reproj_error = self.last_reproj_errors[i]

            # Color based on reprojection error
            if reproj_error is None:
                color = (0, 255, 0)  # Green - no error info
                error_text = ""
            elif reproj_error < REPROJ_ERROR_GOOD:
                color = (0, 255, 0)  # Green - good
                error_text = f" ({reproj_error:.1f}px)"
            elif reproj_error < REPROJ_ERROR_WARNING:
                color = (0, 255, 255)  # Yellow - warning
                error_text = f" ({reproj_error:.1f}px)"
            else:
                color = (0, 0, 255)  # Red - high error (likely outlier)
                error_text = f" ({reproj_error:.1f}px!)"

            # Draw marker
            cv2.drawMarker(display, (u, v), color, cv2.MARKER_CROSS, 20, 2)
            cv2.circle(display, (u, v), 10, color, 2)

            # Draw label with error
            label = f"{i+1}: {desc}{error_text}"
            cv2.putText(display, label, (u + 15, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw crosshair at mouse position if in capture mode
        if self.capture_mode:
            mx, my = self.mouse_pos
            color = (0, 255, 255)  # Yellow
            cv2.line(display, (mx - 20, my), (mx + 20, my), color, 1)
            cv2.line(display, (mx, my - 20), (mx, my + 20), color, 1)
            cv2.putText(display, f"({mx}, {my})", (mx + 10, my - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw status bar at top
        status_bg = np.zeros((60, display.shape[1], 3), dtype=np.uint8)
        status_bg[:] = (40, 40, 40)
        display[:60] = cv2.addWeighted(display[:60], 0.3, status_bg, 0.7, 0)

        # Mode indicator
        if self.capture_mode:
            mode_text = "CAPTURE MODE - Click to add GCPs"
            mode_color = (0, 255, 255)
        else:
            mode_text = "VIEW MODE - Press 's' to start capture"
            mode_color = (200, 200, 200)

        cv2.putText(display, mode_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # GCP count and homography quality
        gcp_text = f"GCPs: {len(self.gcps)}"
        if len(self.gcps) >= 4 and self.last_reproj_errors:
            mean_err = sum(self.last_reproj_errors) / len(self.last_reproj_errors)
            gcp_text += f" | Mean err: {mean_err:.1f}px"
            if mean_err < REPROJ_ERROR_GOOD:
                gcp_text += " (Good)"
            elif mean_err < REPROJ_ERROR_WARNING:
                gcp_text += " (OK)"
            else:
                gcp_text += " (Check outliers!)"
        cv2.putText(display, gcp_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # PTZ info
        if self.ptz_status:
            ptz_text = (f"PTZ: P={self.ptz_status['pan']:.1f} "
                        f"T={self.ptz_status['tilt']:.1f} "
                        f"Z={self.ptz_status['zoom']:.1f}x")
            cv2.putText(display, ptz_text, (200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Controls help at bottom
        help_y = display.shape[0] - 20
        help_text = "s: Start/Stop | z: Undo | c: Clear | q: Quit"
        cv2.putText(display, help_text, (10, help_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return display

    def generate_yaml(self) -> str:
        """Generate YAML config content."""
        context = self.get_camera_context()

        lines = [
            "# GCP Configuration",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Camera: {self.camera_name}",
            f"# GCPs captured: {len(self.gcps)}",
            "",
            "homography:",
            "  approach: feature_match",
            "",
            "  feature_match:",
            "    detector: sift",
            "    min_matches: 4",
            "    ransac_threshold: 5.0",
            "",
            "    # Camera Capture Context",
            "    camera_capture_context:",
            f"      camera_name: \"{context['camera_name']}\"",
            f"      image_width: {context['image_width']}",
            f"      image_height: {context['image_height']}",
            "      ptz_position:",
            f"        pan: {context['ptz_position']['pan']}",
            f"        tilt: {context['ptz_position']['tilt']}",
            f"        zoom: {context['ptz_position']['zoom']}",
            "      intrinsics:",
            f"        focal_length_px: {context['intrinsics']['focal_length_px']}",
            "        principal_point:",
            f"          cx: {context['intrinsics']['principal_point']['cx']}",
            f"          cy: {context['intrinsics']['principal_point']['cy']}",
            f"      capture_timestamp: \"{context['capture_timestamp']}\"",
            f"      notes: \"{context['notes']}\"",
            "",
            "    # Ground Control Points",
            "    ground_control_points:",
        ]

        for i, gcp in enumerate(self.gcps):
            lat = gcp['gps']['latitude']
            lon = gcp['gps']['longitude']
            u = gcp['image']['u']
            v = gcp['image']['v']
            desc = gcp.get('metadata', {}).get('description', f'GCP {i+1}')
            accuracy = gcp.get('metadata', {}).get('accuracy', 'medium')
            timestamp = gcp.get('metadata', {}).get('timestamp', '')

            lines.extend([
                f"      # GCP {i+1}: {desc}",
                "      - gps:",
                f"          latitude: {lat}",
                f"          longitude: {lon}",
            ])

            if 'elevation' in gcp['gps']:
                lines.append(f"          elevation: {gcp['gps']['elevation']}")

            lines.extend([
                "        image:",
                f"          u: {u}",
                f"          v: {v}",
                "        metadata:",
                f"          description: \"{desc}\"",
                f"          accuracy: {accuracy}",
                f"          timestamp: \"{timestamp}\"",
                "",
            ])

        return "\n".join(lines)

    def save_config(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        yaml_content = self.generate_yaml()

        with open(output_path, 'w') as f:
            f.write(yaml_content)

        print(f"\n✓ Saved configuration to: {output_path}")
        print(f"  Camera: {self.camera_name}")
        print(f"  GCPs: {len(self.gcps)}")

    def run(self, output_path: Optional[str] = None) -> None:
        """Run the interactive capture session."""
        rtsp_url = get_rtsp_url(self.camera_name)
        if not rtsp_url:
            raise RuntimeError(f"Cannot get RTSP URL for {self.camera_name}")

        print(f"Connecting to camera '{self.camera_name}'...")
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to connect to camera: {rtsp_url}")

        # Get frame dimensions
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Stream: {self.frame_width}x{self.frame_height}")

        # Get initial PTZ status
        try:
            self.ptz_status = get_ptz_status(self.cam_info['ip'], USERNAME, PASSWORD)
            print(f"PTZ: pan={self.ptz_status['pan']:.1f}°, "
                  f"tilt={self.ptz_status['tilt']:.1f}°, "
                  f"zoom={self.ptz_status['zoom']:.1f}x")
        except Exception as e:
            print(f"Warning: Could not get PTZ status: {e}")

        # Create window and set mouse callback
        window_name = f"GCP Capture - {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\nControls:")
        print("  s - Start/Stop GCP capture mode (freezes frame)")
        print("  z - Undo last GCP")
        print("  c - Clear all GCPs")
        print("  q - Quit")
        print()

        try:
            while True:
                # In capture mode, use frozen frame; otherwise read live feed
                if self.capture_mode:
                    # Use the frozen frame - no camera reads during capture
                    frame = self.frozen_frame.copy()
                else:
                    # Live feed mode - read from camera
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame, reconnecting...")
                        time.sleep(1)
                        cap.release()
                        cap = cv2.VideoCapture(rtsp_url)
                        continue
                    self.current_frame = frame

                # Handle pending click (add GCP) - only in capture mode
                if self.pending_click is not None and self.capture_mode:
                    gcp = self.prompt_gps_coordinates(self.pending_click)
                    if gcp:
                        self.gcps.append(gcp)
                        if self.capture_timestamp is None:
                            self.capture_timestamp = datetime.now().isoformat()
                        # Update homography and show quality metrics
                        metrics = self.update_homography()
                        if metrics:
                            print(f"\n  Homography updated: {metrics['num_inliers']}/{metrics['num_gcps']} inliers "
                                  f"({metrics['inlier_ratio']:.0%}), mean error: {metrics['mean_error_px']:.1f}px")
                    self.pending_click = None

                # Draw overlay
                display = self.draw_overlay(frame)
                cv2.imshow(window_name, display)

                # Handle key presses
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    # Quit
                    if self.gcps:
                        print(f"\n{len(self.gcps)} GCPs captured. Save before quitting?")
                        save = input("Save? (y/n): ").strip().lower()
                        if save == 'y':
                            if output_path is None:
                                output_path = input("Output file: ").strip()
                                if not output_path:
                                    output_path = f"gcps_{self.camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                            self.save_config(output_path)
                    break

                elif key == ord('s'):
                    # Toggle capture mode
                    if not self.capture_mode:
                        # Entering capture mode - freeze the current frame
                        if self.current_frame is not None:
                            self.frozen_frame = self.current_frame.copy()
                            self.capture_mode = True
                            self.capture_timestamp = datetime.now().isoformat()

                            # Get actual frame dimensions from the frozen frame
                            # (more reliable than cv2.CAP_PROP which can be wrong)
                            self.frame_height, self.frame_width = self.frozen_frame.shape[:2]

                            print("\n>>> CAPTURE MODE STARTED - Frame frozen")
                            print(f"    Frame size: {self.frame_width}x{self.frame_height}")
                            print("    Click on points to add GCPs")
                            print("    All GCPs will be on this same frame")
                            # Update PTZ status for the frozen frame
                            try:
                                self.ptz_status = get_ptz_status(
                                    self.cam_info['ip'], USERNAME, PASSWORD
                                )
                            except Exception:
                                pass
                            # Release the camera connection - not needed during capture
                            cap.release()
                            print("    Camera connection released.")
                        else:
                            print("Error: No frame available to capture")
                    else:
                        # Exiting capture mode
                        self.capture_mode = False
                        self.frozen_frame = None
                        print("\n>>> CAPTURE MODE STOPPED")
                        if self.gcps:
                            print(f"  {len(self.gcps)} GCPs captured.")
                            if output_path:
                                self.save_config(output_path)
                            else:
                                print("  Press 'q' to quit and save.")
                        # Reconnect to camera for live feed
                        print("  Reconnecting to camera...")
                        cap = cv2.VideoCapture(rtsp_url)
                        if not cap.isOpened():
                            print("  Warning: Could not reconnect to camera")

                elif key == ord('z'):
                    # Undo last GCP
                    if self.gcps:
                        removed = self.gcps.pop()
                        desc = removed.get('metadata', {}).get('description', 'GCP')
                        print(f"Removed: {desc}")
                    else:
                        print("No GCPs to undo.")

                elif key == ord('c'):
                    # Clear all GCPs
                    if self.gcps:
                        confirm = input(f"Clear all {len(self.gcps)} GCPs? (y/n): ")
                        if confirm.strip().lower() == 'y':
                            self.gcps.clear()
                            print("All GCPs cleared.")

        finally:
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive GCP Capture Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'camera',
        type=str,
        help='Camera name (e.g., Valte, Setram)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output YAML file path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (filename auto-generated)'
    )
    parser.add_argument(
        '--sensor-width',
        type=float,
        default=DEFAULT_SENSOR_WIDTH_MM,
        help=f'Sensor width in mm (default: {DEFAULT_SENSOR_WIDTH_MM})'
    )
    parser.add_argument(
        '--base-focal',
        type=float,
        default=DEFAULT_BASE_FOCAL_LENGTH_MM,
        help=f'Base focal length in mm (default: {DEFAULT_BASE_FOCAL_LENGTH_MM})'
    )
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List available cameras and exit'
    )

    args = parser.parse_args()

    if args.list_cameras:
        print("Available cameras:")
        for cam in CAMERAS:
            print(f"  - {cam['name']} ({cam['ip']})")
        sys.exit(0)

    # Determine output path
    output_path = args.output
    if output_path is None and args.output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = str(Path(args.output_dir) / f"gcps_{args.camera}_{timestamp}.yaml")

    try:
        session = GCPCaptureSession(
            camera_name=args.camera,
            sensor_width_mm=args.sensor_width,
            base_focal_length_mm=args.base_focal,
        )
        session.run(output_path=output_path)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == '__main__':
    main()
