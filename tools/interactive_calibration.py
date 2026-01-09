"""
Interactive Calibration Tool for GPS-to-Image Projection.

This tool guides you through calibrating the projection parameters using
known reference points. It can either use the live camera or a saved frame.

The tool will:
1. Display the camera frame
2. Let you click on points you know the GPS coordinates of
3. Enter the GPS coordinates for each clicked point
4. Calculate the optimal parameters (pan_offset, height)
5. Show you what to update in camera_config.py
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Interactive mode disabled.")

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.coordinate_converter import gps_to_local_xy

# Try to import camera modules
try:
    from poc_homography.camera_config import CAMERAS, get_camera_by_name, get_camera_by_name_safe
    from poc_homography.frame_grabber import grab_frame
    from poc_homography.gps_distance_calculator import dms_to_dd
    from poc_homography.ptz_control import get_ptz_status

    CAMERA_AVAILABLE = True
except (ImportError, ValueError):
    CAMERA_AVAILABLE = False
    # Import safe fallback functions that don't require credentials
    from poc_homography.camera_config import get_camera_by_name_safe
    from poc_homography.gps_distance_calculator import dms_to_dd


class CalibrationSession:
    """Manages an interactive calibration session."""

    def __init__(
        self,
        camera_name: str,
        frame: np.ndarray,
        camera_lat: float,
        camera_lon: float,
        height_m: float,
        pan_offset_deg: float,
        pan_raw: float,
        tilt_deg: float,
        zoom: float,
    ):
        self.camera_name = camera_name
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.camera_lat = camera_lat
        self.camera_lon = camera_lon
        self.height_m = height_m
        self.pan_offset_deg = pan_offset_deg
        self.pan_raw = pan_raw
        self.tilt_deg = tilt_deg
        self.zoom = zoom

        self.image_height, self.image_width = frame.shape[:2]

        # Reference points: list of {pixel_u, pixel_v, gps_lat, gps_lon, name}
        self.reference_points: list[dict] = []

        # Current click position (for entering GPS)
        self.pending_click: tuple[int, int] | None = None

        # Calibration results
        self.best_pan_offset: float | None = None
        self.best_height: float | None = None
        self.best_error: float | None = None

    def add_reference_point(
        self, pixel_u: int, pixel_v: int, gps_lat: float, gps_lon: float, name: str = None
    ):
        """Add a reference point with known GPS coordinates."""
        if name is None:
            name = f"Point {len(self.reference_points) + 1}"

        self.reference_points.append(
            {
                "pixel_u": pixel_u,
                "pixel_v": pixel_v,
                "gps_lat": gps_lat,
                "gps_lon": gps_lon,
                "name": name,
            }
        )
        self._update_display()

    def _update_display(self):
        """Update the display frame with reference points and projections."""
        self.display_frame = self.frame.copy()

        # Draw reference points (green circles with labels)
        for i, pt in enumerate(self.reference_points):
            u, v = int(pt["pixel_u"]), int(pt["pixel_v"])
            cv2.circle(self.display_frame, (u, v), 8, (0, 255, 0), 2)
            cv2.circle(self.display_frame, (u, v), 3, (0, 255, 0), -1)
            label = f"{pt['name']}"
            cv2.putText(
                self.display_frame,
                label,
                (u + 12, v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # If we have calibration results, show projected points (red)
        if self.best_pan_offset is not None and self.best_height is not None:
            K = CameraGeometry.get_intrinsics(self.zoom, self.image_width, self.image_height, 7.18)
            geo = CameraGeometry(w=self.image_width, h=self.image_height)
            pan_deg = self.pan_raw + self.best_pan_offset
            w_pos = np.array([0.0, 0.0, self.best_height])

            try:
                geo.set_camera_parameters(K, w_pos, pan_deg, self.tilt_deg, 640, 640)

                for pt in self.reference_points:
                    x_m, y_m = gps_to_local_xy(
                        self.camera_lat, self.camera_lon, pt["gps_lat"], pt["gps_lon"]
                    )
                    world_pt = np.array([[x_m], [y_m], [1.0]])
                    img_pt = geo.H @ world_pt
                    if img_pt[2, 0] > 0:
                        proj_u = int(img_pt[0, 0] / img_pt[2, 0])
                        proj_v = int(img_pt[1, 0] / img_pt[2, 0])
                        # Draw projected point (red)
                        cv2.circle(self.display_frame, (proj_u, proj_v), 8, (0, 0, 255), 2)
                        cv2.drawMarker(
                            self.display_frame,
                            (proj_u, proj_v),
                            (0, 0, 255),
                            cv2.MARKER_CROSS,
                            16,
                            2,
                        )
                        # Draw line from actual to projected
                        cv2.line(
                            self.display_frame,
                            (int(pt["pixel_u"]), int(pt["pixel_v"])),
                            (proj_u, proj_v),
                            (255, 0, 255),
                            1,
                        )
            except ValueError:
                pass

        # Draw status text
        status_lines = [
            f"Camera: {self.camera_name}",
            f"Pan raw: {self.pan_raw:.1f}, Tilt: {self.tilt_deg:.1f}, Zoom: {self.zoom:.1f}x",
            f"Reference points: {len(self.reference_points)}",
            "",
            "Controls:",
            "  Click: Add reference point",
            "  C: Calibrate with current points",
            "  S: Save calibration results",
            "  R: Reset all points",
            "  Q/ESC: Quit",
        ]

        if self.best_pan_offset is not None:
            status_lines.insert(
                3,
                f"Best: pan_offset={self.best_pan_offset:.1f}, "
                f"height={self.best_height:.2f}m, error={self.best_error:.1f}px",
            )
            status_lines.insert(4, "Green=actual, Red=projected")

        y = 30
        for line in status_lines:
            cv2.putText(
                self.display_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y += 20

    def calibrate(self) -> bool:
        """Run calibration optimization using current reference points."""
        if len(self.reference_points) < 1:
            print("Need at least 1 reference point to calibrate")
            return False

        print(f"\nCalibrating with {len(self.reference_points)} reference points...")

        K = CameraGeometry.get_intrinsics(self.zoom, self.image_width, self.image_height, 7.18)
        geo = CameraGeometry(w=self.image_width, h=self.image_height)

        best_error = float("inf")
        best_pan_offset = self.pan_offset_deg
        best_height = self.height_m

        # Joint optimization over pan offset and height
        for test_offset in np.arange(-180, 180, 1):
            test_pan = self.pan_raw + test_offset
            for test_height in np.arange(1.0, 20.0, 0.2):
                test_w_pos = np.array([0.0, 0.0, test_height])
                try:
                    geo.set_camera_parameters(K, test_w_pos, test_pan, self.tilt_deg, 640, 640)
                except ValueError:
                    continue

                total_error = 0
                valid_points = 0

                for pt in self.reference_points:
                    x_m, y_m = gps_to_local_xy(
                        self.camera_lat, self.camera_lon, pt["gps_lat"], pt["gps_lon"]
                    )
                    world_pt = np.array([[x_m], [y_m], [1.0]])
                    img_pt = geo.H @ world_pt

                    if img_pt[2, 0] > 0:
                        proj_u = img_pt[0, 0] / img_pt[2, 0]
                        proj_v = img_pt[1, 0] / img_pt[2, 0]
                        error = math.sqrt(
                            (pt["pixel_u"] - proj_u) ** 2 + (pt["pixel_v"] - proj_v) ** 2
                        )
                        total_error += error
                        valid_points += 1

                if valid_points > 0:
                    avg_error = total_error / valid_points
                    if avg_error < best_error:
                        best_error = avg_error
                        best_pan_offset = test_offset
                        best_height = test_height

        self.best_pan_offset = best_pan_offset
        self.best_height = best_height
        self.best_error = best_error

        print("\nCalibration Results:")
        print(f"  Best pan_offset: {best_pan_offset:.1f}° (was {self.pan_offset_deg:.1f}°)")
        print(f"  Best height: {best_height:.2f}m (was {self.height_m:.2f}m)")
        print(f"  Average error: {best_error:.1f} pixels")

        self._update_display()
        return True

    def save_results(self, output_path: str = None):
        """Save calibration results to a file."""
        if self.best_pan_offset is None:
            print("No calibration results to save. Run calibration first.")
            return

        if output_path is None:
            output_path = f"calibration_{self.camera_name}.json"

        results = {
            "camera_name": self.camera_name,
            "original_params": {
                "pan_offset_deg": self.pan_offset_deg,
                "height_m": self.height_m,
            },
            "calibrated_params": {
                "pan_offset_deg": self.best_pan_offset,
                "height_m": self.best_height,
            },
            "calibration_error_px": self.best_error,
            "ptz_at_calibration": {
                "pan_raw": self.pan_raw,
                "tilt_deg": self.tilt_deg,
                "zoom": self.zoom,
            },
            "reference_points": self.reference_points,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nCalibration results saved to: {output_path}")
        print("\nTo apply these settings, update camera_config.py:")
        print(f"  'height_m': {self.best_height:.1f},")
        print(f"  'pan_offset_deg': {self.best_pan_offset:.1f},")


def mouse_callback(event, x, y, flags, session: CalibrationSession):
    """Handle mouse events for the calibration window."""
    if event == cv2.EVENT_LBUTTONDOWN:
        session.pending_click = (x, y)
        print(f"\nClicked at pixel ({x}, {y})")
        print("Enter GPS coordinates (lat,lon): ", end="", flush=True)


def run_interactive_session(session: CalibrationSession):
    """Run the interactive calibration session."""
    if not CV2_AVAILABLE:
        print("Error: OpenCV is required for interactive mode")
        return

    window_name = f"Calibration - {session.camera_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1920, session.image_width), min(1080, session.image_height))
    cv2.setMouseCallback(window_name, mouse_callback, session)

    session._update_display()

    print("\n" + "=" * 60)
    print("INTERACTIVE CALIBRATION MODE")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Click on a point in the image that you know the GPS coordinates of")
    print("2. Enter the GPS coordinates when prompted (format: lat,lon)")
    print("3. Repeat for multiple points (more points = better calibration)")
    print("4. Press 'C' to run calibration")
    print("5. Press 'S' to save results")
    print("6. Press 'Q' or ESC to quit")
    print("=" * 60 + "\n")

    while True:
        cv2.imshow(window_name, session.display_frame)
        key = cv2.waitKey(100) & 0xFF

        # Check for pending GPS input
        if session.pending_click is not None:
            # This is handled in the console, check for input
            import select

            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    line = sys.stdin.readline().strip()
                    if line:
                        parts = line.split(",")
                        lat = float(parts[0])
                        lon = float(parts[1])
                        session.add_reference_point(
                            session.pending_click[0], session.pending_click[1], lat, lon
                        )
                        print(
                            f"Added reference point at ({session.pending_click[0]}, "
                            f"{session.pending_click[1]}) -> ({lat:.6f}, {lon:.6f})"
                        )
                except (ValueError, IndexError):
                    print("Invalid format. Use: lat,lon (e.g., 39.640500,-0.230000)")
                session.pending_click = None

        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord("c"):  # Calibrate
            session.calibrate()
        elif key == ord("s"):  # Save
            session.save_results()
        elif key == ord("r"):  # Reset
            session.reference_points = []
            session.best_pan_offset = None
            session.best_height = None
            session.best_error = None
            session._update_display()
            print("Reset all reference points")

    cv2.destroyAllWindows()


def run_batch_calibration(session: CalibrationSession, reference_points: list[dict]):
    """Run calibration in batch mode with pre-defined reference points."""
    print(f"\nBatch calibration with {len(reference_points)} reference points")

    for pt in reference_points:
        session.add_reference_point(
            pt["pixel_u"], pt["pixel_v"], pt["gps_lat"], pt["gps_lon"], pt.get("name")
        )

    session.calibrate()
    session.save_results()
