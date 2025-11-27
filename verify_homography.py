#!/usr/bin/env python3
"""
Homography verification tool - click points in the stream to verify projection.

Usage:
  python verify_homography.py CAMERA_NAME

Click on known ground points in the video window to see their projected world coordinates.
"""

import sys
import cv2
import numpy as np
from camera_config import get_camera_by_name, get_rtsp_url, USERNAME, PASSWORD
from camera_geometry import CameraGeometry
from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ


class HomographyVerifier:
    """Interactive tool to verify homography by clicking points."""

    def __init__(self, camera_name: str):
        self.camera_name = camera_name
        self.cam_info = get_camera_by_name(camera_name)
        if not self.cam_info:
            raise ValueError(f"Camera '{camera_name}' not found")

        # Get camera status
        self.camera = HikvisionPTZ(
            ip=self.cam_info["ip"],
            username=USERNAME,
            password=PASSWORD,
            name=self.cam_info["name"]
        )
        self.status = self.camera.get_status()

        # RTSP setup
        self.rtsp_url = get_rtsp_url(camera_name)
        self.cap = None
        self.geo = None
        self.current_frame = None
        self.clicked_points = []  # Store (image_x, image_y, world_x, world_y)

    def setup_geometry(self, height: float = 5.0):
        """Initialize geometry with camera parameters."""
        # Open stream
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open stream: {self.rtsp_url}")

        # Get frame dimensions
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")

        h, w = frame.shape[:2]
        self.current_frame = frame

        # Setup geometry
        self.geo = CameraGeometry(w, h)
        K = self.geo.get_intrinsics(
            zoom_factor=self.status["zoom"],
            W_px=w,
            H_px=h
        )

        w_pos = np.array([0.0, 0.0, height])

        # IMPORTANT: Hikvision cameras use inverted tilt convention
        # Positive tilt = pointing down, so we negate it
        self.geo.set_camera_parameters(
            K=K,
            w_pos=w_pos,
            pan_deg=self.status["pan"],
            tilt_deg=-self.status["tilt"],  # Negate tilt for Hikvision
            map_width=w,
            map_height=h
        )

        print(f"\n✓ Geometry initialized for {self.camera_name}")
        print(f"  Pan: {self.status['pan']:.1f}°, Tilt: {self.status['tilt']:.1f}°, Zoom: {self.status['zoom']:.2f}")
        print(f"  Height: {height}m\n")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to project points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Project image point to world
            pts_image = [(x, y)]

            # Use homography to project to ground plane
            if self.geo and self.geo.H_inv is not None:
                pts_homogeneous = np.array([[x], [y], [1.0]], dtype=np.float64)
                pts_world = self.geo.H_inv @ pts_homogeneous

                # Normalize
                Xw = pts_world[0, 0] / pts_world[2, 0]
                Yw = pts_world[1, 0] / pts_world[2, 0]

                # Calculate distance from camera
                distance = np.sqrt(Xw**2 + Yw**2)

                self.clicked_points.append((x, y, Xw, Yw))

                print(f"\n📍 Point {len(self.clicked_points)}:")
                print(f"  Image: ({x}, {y}) pixels")
                print(f"  World: ({Xw:.2f}, {Yw:.2f}) meters")
                print(f"  Distance from camera: {distance:.2f}m")
                print(f"  Angle from camera: {np.degrees(np.arctan2(Xw, Yw)):.1f}°")

    def draw_annotations(self, frame):
        """Draw clicked points and their projections."""
        annotated = frame.copy()

        # Draw clicked points
        for i, (x, y, Xw, Yw) in enumerate(self.clicked_points):
            # Draw point
            cv2.circle(annotated, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(annotated, (x, y), 10, (255, 255, 255), 2)

            # Draw label
            label = f"P{i+1}: ({Xw:.1f}m, {Yw:.1f}m)"
            cv2.putText(annotated, label, (x + 15, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw instructions
        cv2.putText(annotated, "Click on ground points to verify homography",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, "Press 'q' to quit, 'c' to clear points, 's' to save",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return annotated

    def run(self):
        """Run interactive verification."""
        cv2.namedWindow('Homography Verification')
        cv2.setMouseCallback('Homography Verification', self.mouse_callback)

        print("\n" + "="*60)
        print("HOMOGRAPHY VERIFICATION - INTERACTIVE MODE")
        print("="*60)
        print("\nInstructions:")
        print("  1. Click on known ground points (e.g., objects, markings)")
        print("  2. Note the displayed world coordinates (in meters)")
        print("  3. Compare with actual measured distances")
        print("  4. Press 'c' to clear points, 's' to save results, 'q' to quit")
        print("\nVerification Tips:")
        print("  • Place objects at known distances (e.g., 5m, 10m, 15m)")
        print("  • Click on the BASE of objects (where they touch the ground)")
        print("  • Verify distances match your measurements")
        print("  • Check angles are correct (X=East, Y=North)")
        print("="*60 + "\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Stream interrupted, reconnecting...")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                continue

            self.current_frame = frame
            annotated = self.draw_annotations(frame)

            cv2.imshow('Homography Verification', annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clicked_points.clear()
                print("\n✓ Points cleared")
            elif key == ord('s'):
                self.save_results()

        self.cleanup()

    def save_results(self):
        """Save verification results to file."""
        if not self.clicked_points:
            print("No points to save")
            return

        filename = f"homography_verification_{self.camera_name}.txt"
        with open(filename, 'w') as f:
            f.write(f"Homography Verification Results\n")
            f.write(f"Camera: {self.camera_name}\n")
            f.write(f"Pan: {self.status['pan']:.1f}°, Tilt: {self.status['tilt']:.1f}°, Zoom: {self.status['zoom']:.2f}\n\n")
            f.write(f"{'Point':<8} {'Image (px)':<20} {'World (m)':<25} {'Distance (m)':<15}\n")
            f.write("-" * 70 + "\n")

            for i, (x, y, Xw, Yw) in enumerate(self.clicked_points):
                dist = np.sqrt(Xw**2 + Yw**2)
                f.write(f"P{i+1:<7} ({x:>4}, {y:>4}){' '*8} ({Xw:>6.2f}, {Yw:>6.2f}){' '*10} {dist:>6.2f}\n")

        print(f"\n✓ Results saved to {filename}")

    def cleanup(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_homography.py CAMERA_NAME")
        print(f"Available cameras: {', '.join([cam['name'] for cam in CAMERAS])}")
        sys.exit(1)

    camera_name = sys.argv[1]
    height = 5.0  # Default height in meters

    # You can add height as optional argument
    if len(sys.argv) > 2:
        height = float(sys.argv[2])

    verifier = HomographyVerifier(camera_name)
    verifier.setup_geometry(height=height)
    verifier.run()


if __name__ == "__main__":
    main()
