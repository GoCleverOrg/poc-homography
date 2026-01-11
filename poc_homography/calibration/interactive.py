"""
Interactive Calibration Tool for Map Point-to-Image Projection.

This module provides interactive calibration sessions that guide users through
calibrating projection parameters using known reference points. It can use either
live camera feeds or saved frames.

The calibration process:
1. Display the camera frame
2. Let user click on points with known Map Point IDs
3. Enter the Map Point ID for each clicked point
4. Calculate optimal parameters (pan_offset, height)
5. Show recommended parameter updates
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.camera_parameters import CameraParameters
from poc_homography.map_points import MapPointRegistry
from poc_homography.types import Degrees, Meters, Millimeters, Pixels, Unitless

# OpenCV is optional - not all environments have GUI support
CV2_AVAILABLE = False
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ReferencePoint:
    """A reference point with known pixel and Map Point ID."""

    pixel_u: int
    pixel_v: int
    map_point_id: str


@dataclass
class CalibrationResults:
    """Results from a calibration session."""

    camera_name: str
    original_pan_offset: Degrees
    original_height: Meters
    calibrated_pan_offset: Degrees
    calibrated_height: Meters
    calibration_error_px: float
    pan_raw: Degrees
    tilt_deg: Degrees
    zoom: Unitless
    reference_points: list[ReferencePoint]

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            "camera_name": self.camera_name,
            "original_params": {
                "pan_offset_deg": float(self.original_pan_offset),
                "height_m": float(self.original_height),
            },
            "calibrated_params": {
                "pan_offset_deg": float(self.calibrated_pan_offset),
                "height_m": float(self.calibrated_height),
            },
            "calibration_error_px": self.calibration_error_px,
            "ptz_at_calibration": {
                "pan_raw": float(self.pan_raw),
                "tilt_deg": float(self.tilt_deg),
                "zoom": float(self.zoom),
            },
            "reference_points": [
                {
                    "pixel_u": pt.pixel_u,
                    "pixel_v": pt.pixel_v,
                    "map_point_id": pt.map_point_id,
                }
                for pt in self.reference_points
            ],
        }


class CalibrationSession:
    """Manages an interactive calibration session."""

    def __init__(
        self,
        camera_name: str,
        frame: npt.NDArray[np.uint8],
        registry: MapPointRegistry,
        height_m: Meters,
        pan_offset_deg: Degrees,
        pan_raw: Degrees,
        tilt_deg: Degrees,
        zoom: Unitless,
        sensor_width_mm: Millimeters = Millimeters(7.18),
    ):
        """
        Initialize a calibration session.

        Args:
            camera_name: Name of the camera being calibrated
            frame: Camera frame image (numpy array)
            registry: Map point registry with known reference points
            height_m: Initial camera height estimate in meters
            pan_offset_deg: Initial pan offset in degrees
            pan_raw: Raw pan value from PTZ in degrees
            tilt_deg: Tilt angle in degrees
            zoom: Zoom factor (unitless)
            sensor_width_mm: Camera sensor width in millimeters
        """
        self.camera_name = camera_name
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.registry = registry
        self.height_m = height_m
        self.pan_offset_deg = pan_offset_deg
        self.pan_raw = pan_raw
        self.tilt_deg = tilt_deg
        self.zoom = zoom
        self.sensor_width_mm = sensor_width_mm

        self.image_height, self.image_width = frame.shape[:2]

        # Reference points collected during the session
        self.reference_points: list[ReferencePoint] = []

        # Current click position (for entering Map Point ID)
        self.pending_click: tuple[int, int] | None = None

        # Calibration results
        self.best_pan_offset: Degrees | None = None
        self.best_height: Meters | None = None
        self.best_error: float | None = None

    def add_reference_point(self, pixel_u: int, pixel_v: int, map_point_id: str) -> bool:
        """
        Add a reference point with known Map Point ID.

        Args:
            pixel_u: Pixel U coordinate (horizontal)
            pixel_v: Pixel V coordinate (vertical)
            map_point_id: ID of the map point in the registry

        Returns:
            True if point was added successfully, False if map_point_id not found
        """
        if map_point_id not in self.registry.points:
            print(f"Error: Map Point ID '{map_point_id}' not found in registry")
            return False

        self.reference_points.append(
            ReferencePoint(pixel_u=pixel_u, pixel_v=pixel_v, map_point_id=map_point_id)
        )
        self._update_display()
        return True

    def _update_display(self) -> None:
        """Update the display frame with reference points and projections."""
        if cv2 is None:
            return

        self.display_frame = self.frame.copy()

        # Draw reference points (green circles with labels)
        for pt in self.reference_points:
            u, v = pt.pixel_u, pt.pixel_v
            cv2.circle(self.display_frame, (u, v), 8, (0, 255, 0), 2)
            cv2.circle(self.display_frame, (u, v), 3, (0, 255, 0), -1)
            label = pt.map_point_id
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
            K = CameraGeometry.get_intrinsics(
                self.zoom,
                Pixels(self.image_width),
                Pixels(self.image_height),
                self.sensor_width_mm,
            )
            pan_deg = Degrees(self.pan_raw + self.best_pan_offset)
            w_pos = np.array([0.0, 0.0, float(self.best_height)])

            try:
                params = CameraParameters.create(
                    image_width=Pixels(self.image_width),
                    image_height=Pixels(self.image_height),
                    intrinsic_matrix=K,
                    camera_position=w_pos,
                    pan_deg=pan_deg,
                    tilt_deg=self.tilt_deg,
                    roll_deg=Degrees(0.0),
                    map_width=Pixels(640),
                    map_height=Pixels(640),
                    pixels_per_meter=Unitless(100.0),
                )
                result = CameraGeometry.compute(params)
                H = result.homography_matrix

                for pt in self.reference_points:
                    map_point = self.registry.points[pt.map_point_id]
                    x_m, y_m = map_point.pixel_x, map_point.pixel_y
                    world_pt = np.array([[x_m], [y_m], [1.0]])
                    img_pt = H @ world_pt
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
                            (pt.pixel_u, pt.pixel_v),
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
                self.display_frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += 20

    def calibrate(self) -> bool:
        """
        Run calibration optimization using current reference points.

        Returns:
            True if calibration succeeded, False if insufficient points
        """
        if len(self.reference_points) < 1:
            print("Need at least 1 reference point to calibrate")
            return False

        print(f"\nCalibrating with {len(self.reference_points)} reference points...")

        K = CameraGeometry.get_intrinsics(
            self.zoom,
            Pixels(self.image_width),
            Pixels(self.image_height),
            self.sensor_width_mm,
        )

        best_error = float("inf")
        best_pan_offset = self.pan_offset_deg
        best_height = self.height_m

        # Joint optimization over pan offset and height
        for test_offset in np.arange(-180, 180, 1):
            test_pan = Degrees(self.pan_raw + test_offset)
            for test_height in np.arange(1.0, 20.0, 0.2):
                test_w_pos = np.array([0.0, 0.0, test_height])
                try:
                    params = CameraParameters.create(
                        image_width=Pixels(self.image_width),
                        image_height=Pixels(self.image_height),
                        intrinsic_matrix=K,
                        camera_position=test_w_pos,
                        pan_deg=test_pan,
                        tilt_deg=self.tilt_deg,
                        roll_deg=Degrees(0.0),
                        map_width=Pixels(640),
                        map_height=Pixels(640),
                        pixels_per_meter=Unitless(100.0),
                    )
                    result = CameraGeometry.compute(params)
                except ValueError:
                    continue

                H = result.homography_matrix
                total_error = 0.0
                valid_points = 0

                for pt in self.reference_points:
                    map_point = self.registry.points[pt.map_point_id]
                    x_m, y_m = map_point.pixel_x, map_point.pixel_y
                    world_pt = np.array([[x_m], [y_m], [1.0]])
                    img_pt = H @ world_pt

                    if img_pt[2, 0] > 0:
                        proj_u = float(img_pt[0, 0] / img_pt[2, 0])
                        proj_v = float(img_pt[1, 0] / img_pt[2, 0])
                        error = math.sqrt((pt.pixel_u - proj_u) ** 2 + (pt.pixel_v - proj_v) ** 2)
                        total_error += error
                        valid_points += 1

                if valid_points > 0:
                    avg_error = total_error / valid_points
                    if avg_error < best_error:
                        best_error = avg_error
                        best_pan_offset = Degrees(float(test_offset))
                        best_height = Meters(float(test_height))

        self.best_pan_offset = best_pan_offset
        self.best_height = best_height
        self.best_error = best_error

        print("\nCalibration Results:")
        print(f"  Best pan_offset: {best_pan_offset:.1f}° (was {self.pan_offset_deg:.1f}°)")
        print(f"  Best height: {best_height:.2f}m (was {self.height_m:.2f}m)")
        print(f"  Average error: {best_error:.1f} pixels")

        self._update_display()
        return True

    def save_results(self, output_path: Path | None = None) -> Path:
        """
        Save calibration results to a JSON file.

        Args:
            output_path: Optional path to save results. If None, uses default name.

        Returns:
            Path where results were saved

        Raises:
            ValueError: If no calibration results to save
        """
        if self.best_pan_offset is None or self.best_height is None:
            raise ValueError("No calibration results to save. Run calibration first.")

        if output_path is None:
            output_path = Path(f"calibration_{self.camera_name}.json")

        results = CalibrationResults(
            camera_name=self.camera_name,
            original_pan_offset=self.pan_offset_deg,
            original_height=self.height_m,
            calibrated_pan_offset=self.best_pan_offset,
            calibrated_height=self.best_height,
            calibration_error_px=self.best_error or 0.0,
            pan_raw=self.pan_raw,
            tilt_deg=self.tilt_deg,
            zoom=self.zoom,
            reference_points=self.reference_points,
        )

        with output_path.open("w") as f:
            json.dump(results.to_dict(), f, indent=2)

        print(f"\nCalibration results saved to: {output_path}")
        print("\nTo apply these settings, update camera_config.py:")
        print(f"  'height_m': {self.best_height:.1f},")
        print(f"  'pan_offset_deg': {self.best_pan_offset:.1f},")

        return output_path

    def get_results(self) -> CalibrationResults | None:
        """
        Get calibration results if available.

        Returns:
            CalibrationResults if calibration has been run, None otherwise
        """
        if self.best_pan_offset is None or self.best_height is None:
            return None

        return CalibrationResults(
            camera_name=self.camera_name,
            original_pan_offset=self.pan_offset_deg,
            original_height=self.height_m,
            calibrated_pan_offset=self.best_pan_offset,
            calibrated_height=self.best_height,
            calibration_error_px=self.best_error or 0.0,
            pan_raw=self.pan_raw,
            tilt_deg=self.tilt_deg,
            zoom=self.zoom,
            reference_points=self.reference_points,
        )


def run_interactive_session(session: CalibrationSession) -> None:
    """
    Run the interactive calibration session with OpenCV GUI.

    Args:
        session: CalibrationSession to run interactively

    Raises:
        RuntimeError: If OpenCV is not available
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required for interactive mode but is not available")

    # Re-import cv2 locally to satisfy type checker after availability check
    import select
    import sys

    import cv2 as cv2_lib

    def mouse_callback(
        event: int, x: int, y: int, _flags: int, user_session: CalibrationSession | None
    ) -> None:
        """Handle mouse events for the calibration window."""
        if user_session is None:
            return
        if event == cv2_lib.EVENT_LBUTTONDOWN:
            user_session.pending_click = (x, y)
            print(f"\nClicked at pixel ({x}, {y})")
            print("Enter Map Point ID (e.g., Z1, P5): ", end="", flush=True)

    window_name = f"Calibration - {session.camera_name}"
    cv2_lib.namedWindow(window_name, cv2_lib.WINDOW_NORMAL)
    cv2_lib.resizeWindow(
        window_name, min(1920, session.image_width), min(1080, session.image_height)
    )
    cv2_lib.setMouseCallback(window_name, mouse_callback, session)

    session._update_display()

    print("\n" + "=" * 60)
    print("INTERACTIVE CALIBRATION MODE")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Click on a point in the image that you know the Map Point ID of")
    print("2. Enter the Map Point ID when prompted (e.g., Z1, P5)")
    print("3. Repeat for multiple points (more points = better calibration)")
    print("4. Press 'C' to run calibration")
    print("5. Press 'S' to save results")
    print("6. Press 'Q' or ESC to quit")
    print("=" * 60 + "\n")

    while True:
        cv2_lib.imshow(window_name, session.display_frame)
        key = cv2_lib.waitKey(100) & 0xFF

        # Check for pending Map Point ID input
        if session.pending_click is not None:
            # This is handled in the console, check for input
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    line = sys.stdin.readline().strip()
                    if line:
                        map_point_id = line.upper()
                        if map_point_id in session.registry.points:
                            session.add_reference_point(
                                session.pending_click[0], session.pending_click[1], map_point_id
                            )
                            map_point = session.registry.points[map_point_id]
                            print(
                                f"Added reference point at ({session.pending_click[0]}, "
                                f"{session.pending_click[1]}) -> {map_point_id} "
                                f"(map coords: {map_point.pixel_x:.2f}, {map_point.pixel_y:.2f})"
                            )
                        else:
                            print(f"Map Point ID '{map_point_id}' not found in registry")
                except (ValueError, IndexError):
                    print("Invalid format. Enter a Map Point ID (e.g., Z1, P5)")
                session.pending_click = None

        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord("c"):  # Calibrate
            session.calibrate()
        elif key == ord("s"):  # Save
            try:
                session.save_results()
            except ValueError as e:
                print(f"Error: {e}")
        elif key == ord("r"):  # Reset
            session.reference_points = []
            session.best_pan_offset = None
            session.best_height = None
            session.best_error = None
            session._update_display()
            print("Reset all reference points")

    cv2_lib.destroyAllWindows()


def run_batch_calibration(
    session: CalibrationSession, reference_points: list[dict[str, Any]]
) -> CalibrationResults | None:
    """
    Run calibration in batch mode with pre-defined reference points.

    Args:
        session: CalibrationSession to use
        reference_points: List of dicts with pixel_u, pixel_v, map_point_id

    Returns:
        CalibrationResults if successful, None otherwise
    """
    print(f"\nBatch calibration with {len(reference_points)} reference points")

    for pt in reference_points:
        session.add_reference_point(pt["pixel_u"], pt["pixel_v"], pt["map_point_id"])

    if session.calibrate():
        return session.get_results()

    return None
