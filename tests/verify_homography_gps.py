#!/usr/bin/env python3
"""
Interactive homography verification with GPS ground truth.
Click points, enter their GPS coordinates, and compare with homography.
"""

import sys
import cv2
import numpy as np
from poc_homography.camera_config import get_camera_by_name, get_camera_gps, get_rtsp_url, USERNAME, PASSWORD
from poc_homography.camera_geometry import CameraGeometry
from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ
from poc_homography.gps_distance_calculator import (dms_to_dd, haversine_distance, bearing_between_points,
                                      compare_distances, local_xy_to_gps, dd_to_dms)


class GPSHomographyVerifier:
    """Interactive homography verifier with GPS validation."""

    def __init__(self, camera_name: str, camera_gps: dict = None):
        self.camera_name = camera_name
        self.cam_info = get_camera_by_name(camera_name)

        # Use provided GPS or get from config
        self.camera_gps = camera_gps if camera_gps else get_camera_gps(camera_name)

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

        # Validation data
        self.validation_points = []  # [(img_x, img_y, world_x, world_y, gps_lat, gps_lon)]
        self.current_height = 5.0
        self.height_estimates = []

    def setup_geometry(self, height: float = 5.0):
        """Initialize geometry with camera parameters."""
        self.current_height = height

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

        # Pass tilt directly - the internal _get_rotation_matrix() handles
        # the Hikvision convention (positive = down) conversion
        self.geo.set_camera_parameters(
            K=K,
            w_pos=w_pos,
            pan_deg=self.status["pan"],
            tilt_deg=self.status["tilt"],
            map_width=640,
            map_height=h
        )

        print(f"\n✓ Geometry initialized for {self.camera_name}")
        print(f"  Pan: {self.status['pan']:.1f}°, Tilt: {self.status['tilt']:.1f}° (Hikvision)")
        print(f"  Zoom: {self.status['zoom']:.2f}x")
        print(f"  Height: {height}m\n")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to add validation points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Project image point to world
            pt_img = np.array([[x], [y], [1.0]])
            pt_world = self.geo.H_inv @ pt_img

            if abs(pt_world[2, 0]) < 1e-6:
                print(f"\n⚠️  Invalid point (near horizon)")
                return

            Xw = pt_world[0, 0] / pt_world[2, 0]
            Yw = pt_world[1, 0] / pt_world[2, 0]
            dist_homography = np.sqrt(Xw**2 + Yw**2)
            angle = np.degrees(np.arctan2(Xw, Yw))

            # Convert camera GPS to decimal degrees
            cam_lat_dd = dms_to_dd(self.camera_gps["lat"])
            cam_lon_dd = dms_to_dd(self.camera_gps["lon"])

            # Estimate GPS coordinates from homography
            estimated_lat, estimated_lon = local_xy_to_gps(cam_lat_dd, cam_lon_dd, Xw, Yw)
            estimated_lat_dms = dd_to_dms(estimated_lat, is_latitude=True)
            estimated_lon_dms = dd_to_dms(estimated_lon, is_latitude=False)

            print(f"\n" + "="*70)
            print(f"📍 POINT {len(self.validation_points) + 1} CLICKED")
            print("="*70)
            print(f"\nImage coordinates: ({x}, {y}) pixels")
            print(f"Homography world: ({Xw:.2f}, {Yw:.2f}) meters")
            print(f"Homography distance: {dist_homography:.2f}m")
            print(f"Angle from camera: {angle:.1f}°")

            print(f"\n🌍 ESTIMATED GPS (from homography):")
            print(f"  Latitude:  {estimated_lat_dms} ({estimated_lat:.6f}°)")
            print(f"  Longitude: {estimated_lon_dms} ({estimated_lon:.6f}°)")

            # Ask for GPS coordinates
            print(f"\nEnter ACTUAL GPS coordinates of this point:")
            print(f"(Press Enter without coordinates to skip validation)")
            print(f"(Copy estimated GPS if it looks correct)")

            try:
                lat_input = input("  Latitude (e.g., 39°38'25.6\"N): ").strip()
                if not lat_input:
                    print("  Skipped GPS validation")
                    self.validation_points.append((x, y, Xw, Yw, None, None))
                    return

                lon_input = input("  Longitude (e.g., 0°13'48.4\"W): ").strip()

                # Validate with GPS
                point_gps = {"lat": lat_input, "lon": lon_input}

                results = compare_distances(
                    self.camera_gps,
                    point_gps,
                    dist_homography,
                    verbose=True
                )

                # Store validation point
                self.validation_points.append((
                    x, y, Xw, Yw,
                    results["point_dd"][0], results["point_dd"][1]
                ))

                # Estimate better height
                gps_dist = results["gps_distance_m"]
                if abs(results["error_m"]) > 0.5:
                    # Scale factor = gps_distance / homography_distance
                    scale_factor = gps_dist / dist_homography
                    estimated_height = self.current_height * scale_factor

                    print(f"\n💡 HEIGHT CALIBRATION HINT:")
                    print(f"   Current height: {self.current_height:.2f}m")
                    print(f"   Distance scale factor: {scale_factor:.2f}x")
                    print(f"   Suggested height: {estimated_height:.2f}m")
                    print(f"   (Try: python verify_homography_gps.py {self.camera_name} {estimated_height:.1f})")

                    self.height_estimates.append(estimated_height)

            except KeyboardInterrupt:
                print("\n  Cancelled")
            except Exception as e:
                print(f"\n  Error: {e}")

    def draw_annotations(self, frame):
        """Draw validation points."""
        annotated = frame.copy()

        # Draw validation points
        for i, point in enumerate(self.validation_points):
            x, y, Xw, Yw, lat, lon = point

            # Draw point
            color = (0, 255, 0) if lat is not None else (128, 128, 128)
            cv2.circle(annotated, (x, y), 8, color, -1)
            cv2.circle(annotated, (x, y), 10, (255, 255, 255), 2)

            # Draw label
            dist = np.sqrt(Xw**2 + Yw**2)
            label = f"P{i+1}: {dist:.1f}m"
            if lat is not None:
                label += " ✓"

            cv2.putText(annotated, label, (x + 15, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw instructions
        cv2.putText(annotated, "Click point, enter GPS to validate homography",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, "Press 'q' to quit, 'c' to clear, 's' to save, 'r' to recalibrate",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show current height
        cv2.putText(annotated, f"Height: {self.current_height:.1f}m",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return annotated

    def recalibrate_height(self):
        """Recalibrate height based on collected data."""
        if not self.height_estimates:
            print("\n⚠️  No height estimates available. Validate some GPS points first.")
            return

        avg_height = np.mean(self.height_estimates)
        std_height = np.std(self.height_estimates)

        print(f"\n" + "="*70)
        print("HEIGHT RECALIBRATION")
        print("="*70)
        print(f"\nCurrent height: {self.current_height:.2f}m")
        print(f"Estimated heights from {len(self.height_estimates)} points:")
        for i, h in enumerate(self.height_estimates):
            print(f"  Point {i+1}: {h:.2f}m")

        print(f"\nAverage: {avg_height:.2f}m ± {std_height:.2f}m")
        print(f"\nRecalibrate to {avg_height:.2f}m? (y/n): ", end='')

        response = input().strip().lower()
        if response == 'y':
            self.setup_geometry(height=avg_height)
            self.validation_points.clear()
            self.height_estimates.clear()
            print(f"\n✓ Recalibrated to height={avg_height:.2f}m")
            print("Click new validation points to verify")
        else:
            print("\nCalibration cancelled")

    def save_results(self):
        """Save validation results to file."""
        if not self.validation_points:
            print("\n⚠️  No validation points to save")
            return

        filename = f"gps_validation_{self.camera_name}.txt"
        with open(filename, 'w') as f:
            f.write(f"GPS Homography Validation Results\n")
            f.write(f"Camera: {self.camera_name}\n")
            f.write(f"Camera GPS: {self.camera_gps['lat']}, {self.camera_gps['lon']}\n")
            f.write(f"Height: {self.current_height:.2f}m\n")
            f.write(f"Pan: {self.status['pan']:.1f}°, Tilt: {self.status['tilt']:.1f}°, Zoom: {self.status['zoom']:.2f}x\n\n")

            f.write(f"{'Point':<8} {'Image (px)':<20} {'Homography (m)':<25} {'GPS':<30} {'Status':<10}\n")
            f.write("-" * 100 + "\n")

            for i, point in enumerate(self.validation_points):
                x, y, Xw, Yw, lat, lon = point
                dist = np.sqrt(Xw**2 + Yw**2)

                if lat is not None:
                    status = "GPS ✓"
                else:
                    status = "No GPS"

                f.write(f"P{i+1:<7} ({x:>4}, {y:>4}){' '*8} ({Xw:>6.2f}, {Yw:>6.2f}) {dist:>6.2f}m{' '*5} {status}\n")

        print(f"\n✓ Results saved to {filename}")

    def run(self):
        """Run interactive verification with GPS validation."""
        cv2.namedWindow('GPS Homography Verification')
        cv2.setMouseCallback('GPS Homography Verification', self.mouse_callback)

        print("\n" + "="*70)
        print("GPS-VALIDATED HOMOGRAPHY VERIFICATION")
        print("="*70)
        print(f"\nCamera: {self.camera_name}")
        print(f"Camera GPS: {self.camera_gps['lat']}, {self.camera_gps['lon']}")
        print("\nInstructions:")
        print("  1. Click on a ground point in the video")
        print("  2. Enter its GPS coordinates when prompted")
        print("  3. Compare homography vs GPS distance")
        print("  4. Collect multiple points for calibration")
        print("  5. Press 'r' to recalibrate height based on collected data")
        print("\nControls:")
        print("  Click: Add validation point")
        print("  'c': Clear all points")
        print("  's': Save results")
        print("  'r': Recalibrate height")
        print("  'q': Quit")
        print("="*70 + "\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Stream interrupted, reconnecting...")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                continue

            self.current_frame = frame
            annotated = self.draw_annotations(frame)

            cv2.imshow('GPS Homography Verification', annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.validation_points.clear()
                self.height_estimates.clear()
                print("\n✓ Points cleared")
            elif key == ord('s'):
                self.save_results()
            elif key == ord('r'):
                self.recalibrate_height()

        self.cleanup()

    def cleanup(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_homography_gps.py CAMERA_NAME [HEIGHT]")
        print('Example: python verify_homography_gps.py Valte 5.0')
        print("\nCamera GPS coordinates are automatically loaded from camera_config.py")
        print("Available cameras: Valte, Setram")
        sys.exit(1)

    camera_name = sys.argv[1]

    # Get height from args or camera config
    if len(sys.argv) > 2:
        height = float(sys.argv[2])
    else:
        cam_info = get_camera_by_name(camera_name)
        height = cam_info.get('height_m', 5.0) if cam_info else 5.0

    # GPS coordinates loaded automatically from config
    verifier = GPSHomographyVerifier(camera_name)
    verifier.setup_geometry(height=height)
    verifier.run()


if __name__ == "__main__":
    main()
