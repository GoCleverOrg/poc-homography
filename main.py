# main.py

import datetime
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ

# Import camera configuration
from poc_homography.camera_config import (
    PASSWORD,
    USERNAME,
    get_camera_by_name,
    get_rtsp_url,
)

# Import camera_geometry for backward compatibility
# Import GPS utilities
from poc_homography.gps_distance_calculator import dms_to_dd
from poc_homography.homography_config import HomographyConfig, get_default_config
from poc_homography.homography_factory import HomographyFactory

# Import unified homography interface
from poc_homography.homography_interface import (
    CoordinateSystemMode,
    HomographyProvider,
    HomographyProviderExtended,
)
from poc_homography.intrinsic_extrinsic_homography import IntrinsicExtrinsicHomography

# Configure logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------


class VideoAnnotator:
    """
    Main class for video/stream loading, annotation, and output with expanded canvas.
    Supports both video files and live RTSP streams.
    """

    def __init__(
        self,
        video_path: str | None = None,
        output_path: str = "output_annotated.mp4",
        side_panel_width: int = 640,
        side_panel_image: str | None = None,
        homography_config: HomographyConfig | None = None,
    ):
        """
        Initialize video annotator.

        Args:
            video_path: Path to input video file or None for stream mode.
            output_path: Path for output annotated video
            side_panel_width: Width of side panel in pixels (for top-down view)
            side_panel_image: Optional path to image for side panel (e.g., floor plan)
            homography_config: Optional configuration for homography approach (uses default if not provided)
        """
        self.video_path = video_path
        self.output_path = output_path
        self.side_panel_width = side_panel_width
        self.side_panel_image_path = side_panel_image

        # Video properties (set during load)
        self.cap = None
        self.fps = None
        self.frame_width = None
        self.frame_height = None
        self.total_frames = None

        # Output writer
        self.out = None
        self.out_expanded = None

        # Side panel image
        self.side_panel_template = None

        # Homography configuration and provider
        self.homography_config = homography_config or get_default_config()

        # Geometry Engine: Initialized after frame size is known
        # Uses HomographyProvider interface (can be any implementation)
        self.geo: HomographyProvider | None = None

    def load_video(self) -> bool:
        """Load video from file and extract properties."""
        if not self.video_path:
            print("Error: video_path is None. Use load_stream() for RTSP.")
            return False

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return False

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            logger.warning(f"Invalid FPS {self.fps}, defaulting to 30")
            self.fps = 30

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("Video loaded successfully:")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")

        # Initialize Geometry Engine using factory
        self.geo = HomographyFactory.create(
            self.homography_config.approach,
            width=self.frame_width,
            height=self.frame_height,
            **self.homography_config.get_approach_config(self.homography_config.approach),
        )

        # Load side panel
        if self.side_panel_image_path:
            self._load_side_panel_image()
        else:
            self._create_blank_side_panel()

        return True

    def load_stream(self, rtsp_url: str) -> bool:
        """
        Load RTSP stream and extract properties from the first frame.
        """
        # Use cv2.CAP_FFMPEG to ensure compatibility with RTSP
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.fps = 30  # Default FPS for recording.

        if not self.cap.isOpened():
            print(f"Error: Could not open RTSP stream at {rtsp_url}")
            return False

        # Read the first frame to get resolution
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read first frame from stream.")
            return False

        # Set properties based on the first frame
        self.frame_height, self.frame_width, _ = frame.shape
        self.total_frames = -1  # Not applicable for stream

        print("RTSP stream loaded successfully:")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  Recording FPS: {self.fps} (Actual stream speed varies)")

        # Initialize Geometry Engine using factory
        self.geo = HomographyFactory.create(
            self.homography_config.approach,
            width=self.frame_width,
            height=self.frame_height,
            **self.homography_config.get_approach_config(self.homography_config.approach),
        )

        # Load side panel
        if self.side_panel_image_path:
            self._load_side_panel_image()
        else:
            self._create_blank_side_panel()

        return True

    def _load_side_panel_image(self):
        """Load and resize side panel image (e.g., floor plan for top-down view)."""
        img = cv2.imread(self.side_panel_image_path)
        if img is None:
            print(f"Warning: Could not load side panel image {self.side_panel_image_path}")
            self._create_blank_side_panel()
            return

        # Resize to match video height and desired width
        self.side_panel_template = cv2.resize(img, (self.side_panel_width, self.frame_height))

    def _create_blank_side_panel(self):
        """Create blank side panel with grid for future plotting."""
        self.side_panel_template = (
            np.ones((self.frame_height, self.side_panel_width, 3), dtype=np.uint8) * 240
        )  # Light gray

        # Draw grid lines for reference
        grid_spacing = 50
        # Draw Grid based on real meters (if geo exists)
        if self.geo:
            try:
                # PPM is not defined in CameraGeometry yet, skipping real meter grid for now
                pass
            except AttributeError:
                pass

        for i in range(0, self.frame_height, grid_spacing):
            cv2.line(
                self.side_panel_template, (0, i), (self.side_panel_width, i), (200, 200, 200), 1
            )
        for i in range(0, self.side_panel_width, grid_spacing):
            cv2.line(self.side_panel_template, (i, 0), (i, self.frame_height), (200, 200, 200), 1)

        # Add label
        cv2.putText(
            self.side_panel_template,
            "Top-Down View (Homography)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    def setup_writers(self):
        """Setup video writers for both normal and expanded output."""
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Writer for normal annotated video
        self.out = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height)
        )

        # Writer for expanded canvas (video + side panel)
        expanded_path = self.output_path.replace(".mp4", "_expanded.mp4")
        expanded_width = self.frame_width + self.side_panel_width
        self.out_expanded = cv2.VideoWriter(
            expanded_path, fourcc, self.fps, (expanded_width, self.frame_height)
        )

        print("Output writers initialized:")
        print(f"  Normal: {self.output_path}")
        print(f"  Expanded: {expanded_path}")

    def annotate_frame(
        self,
        frame: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        labels: list[str] | None = None,
        colors: list[tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        """
        Annotate frame with bounding boxes.
        """
        annotated = frame.copy()

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box

            # Get color (default to green if not specified)
            color = colors[idx] if colors and idx < len(colors) else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Visualize the "Foot Point" used for homography (bottom-center)
            foot_x = int((x1 + x2) / 2)
            foot_y = y2
            cv2.circle(annotated, (foot_x, foot_y), 5, (0, 0, 255), -1)

            # Draw label if provided
            if labels and idx < len(labels):
                label = labels[idx]
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated, (x1, y1 - label_height - 10), (x1 + label_width + 10, y1), color, -1
                )
                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        return annotated

    def create_expanded_frame(
        self, annotated_frame: np.ndarray, tracked_points: list[tuple[int, int]] | None = None
    ) -> np.ndarray:
        """
        Create expanded canvas with annotated video and side panel.
        """
        # Create side panel for this frame (copy template)
        side_panel = self.side_panel_template.copy()

        # Draw camera location on the side panel at the bottom-center
        try:
            h_sp, w_sp = side_panel.shape[:2]

            # Place camera marker at bottom center with a small margin
            cam_px = w_sp // 2
            cam_py = h_sp - 30

            # Draw filled red circle for camera
            cv2.circle(side_panel, (cam_px, cam_py), 8, (0, 0, 255), -1)

            # Draw centered label 'cam' slightly above the marker
            (label_w, _label_h), _ = cv2.getTextSize("cam", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_x = cam_px - (label_w // 2)
            label_y = cam_py - 12
            cv2.putText(
                side_panel, "cam", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Failed to draw camera marker: {e}")
        # Plot tracked points if provided — position them relative to the camera and zoom to fit
        if tracked_points:
            try:
                h_sp, w_sp = side_panel.shape[:2]
                desired_cam_px = w_sp // 2
                desired_cam_py = h_sp - 30

                # Compute original camera position in side-panel pixels (if available)
                # Default to desired position
                cam_orig_px, cam_orig_py = desired_cam_px, desired_cam_py

                # Override if geometry provides camera position
                if self.geo and hasattr(self.geo, "world_to_map"):
                    camera_pos = (
                        self.geo.get_camera_position()
                        if hasattr(self.geo, "get_camera_position")
                        else None
                    )
                    if camera_pos is not None:
                        cam_orig_px, cam_orig_py = self.geo.world_to_map(
                            float(camera_pos[0]),
                            float(camera_pos[1]),
                            sw=self.side_panel_width,
                            sh=self.frame_height,
                        )

                # Compute deltas from camera original position
                deltas = []
                max_dx = 0.0
                max_dy = 0.0

                for px, py in tracked_points:
                    dx = px - cam_orig_px
                    dy = py - cam_orig_py
                    deltas.append((dx, dy))
                    max_dx = max(max_dx, abs(dx))
                    max_dy = max(max_dy, abs(dy))

                # Determine scale to fit points inside side-panel (zoom out if needed)
                margin = 40
                half_width_available = (w_sp // 2) - margin
                top_available = desired_cam_py - margin

                sx = (half_width_available / max_dx) if max_dx > 0 else float("inf")
                sy = (top_available / max_dy) if max_dy > 0 else float("inf")
                scale = min(1.0, sx, sy)

                # Draw transformed points
                for dx, dy in deltas:
                    new_x = int(desired_cam_px + dx * scale)
                    new_y = int(desired_cam_py + dy * scale)
                    cv2.circle(side_panel, (new_x, new_y), 5, (0, 0, 255), -1)
                    cv2.circle(side_panel, (new_x, new_y), 8, (255, 255, 255), 1)
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Failed to project tracked points: {e}")
                # Fallback: draw points directly if anything goes wrong
                for point in tracked_points:
                    cv2.circle(side_panel, point, 5, (0, 0, 255), -1)
                    cv2.circle(side_panel, point, 8, (255, 255, 255), 1)

        # Concatenate horizontally: video on left, side panel on right
        expanded = np.hstack([annotated_frame, side_panel])

        return expanded

    def process_video(self, box_generator=None):
        """
        Process entire video file with annotations.
        """
        if not self.load_video():
            return

        self.setup_writers()

        frame_number = 0

        print("\nProcessing video...")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("End of video reached.")
                break

            # Get bounding boxes for this frame
            if box_generator:
                boxes, labels, colors = box_generator(frame, frame_number)
            else:
                boxes, labels, colors = self._generate_example_boxes(frame_number)

            # Annotate frame
            annotated = self.annotate_frame(frame, boxes, labels, colors)

            # Create expanded frame with side panel
            tracked_points = self._generate_example_tracked_points(boxes, frame_number)
            expanded = self.create_expanded_frame(annotated, tracked_points)

            # Write both outputs
            self.out.write(annotated)
            self.out_expanded.write(expanded)

            # Progress indicator
            if frame_number % 30 == 0:
                if self.total_frames > 0:
                    progress = (frame_number / self.total_frames) * 100
                    print(f"Progress: {progress:.1f}% (frame {frame_number}/{self.total_frames})")
                else:
                    print(f"Processing frame {frame_number}")

            frame_number += 1

        self.cleanup()
        print("\nVideo processing complete!")

    def process_stream(self, rtsp_url: str, duration_seconds: float, box_generator=None):
        """
        Process RTSP stream live, display frames, and record to files.
        If duration_seconds <= 0, the function runs indefinitely until externally stopped.

        Args:
            rtsp_url: The full RTSP URL.
            duration_seconds: The length of video to capture in seconds (float).
            box_generator: Optional function that returns (boxes, labels, colors).
        """
        # --- Initialization and Setup ---

        is_timed_capture = duration_seconds > 0

        if not self.load_stream(rtsp_url):
            print("🛑 Initial stream load failed. Aborting process.")
            return

        try:
            self.setup_writers()
        except Exception as e:
            print(f"🛑 Error setting up video writers: {e}. Aborting.")
            self.cleanup()
            return

        frame_number = 0
        target_frames = int(duration_seconds * self.fps) if is_timed_capture else -1

        start_time = time.time()
        MAX_RECONNECT_ATTEMPTS = 5
        reconnect_attempts = 0

        if is_timed_capture:
            print(
                f"\n🎥 Starting TIMED capture for {duration_seconds:.2f} seconds ({target_frames} frames)..."
            )
        else:
            print("\n🔄 Starting CONTINUOUS stream capture (runs until stopped externally)...")

        # --- Main Processing Loop Condition ---
        while (is_timed_capture and frame_number < target_frames) or (not is_timed_capture):
            if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                print(
                    f"\n❌ Max reconnection attempts ({MAX_RECONNECT_ATTEMPTS}) reached. Stopping capture."
                )
                break

            # --- Frame Read and Reconnection Logic ---
            try:
                ret, frame = self.cap.read()
            except cv2.error as e:
                print(f"\n⚠️ OpenCV Read Error: {e}. Attempting recovery...")
                ret = False

            if not ret:
                reconnect_attempts += 1
                wait_time = min(5, reconnect_attempts * 2)

                print(
                    f"🚨 Stream lost or failed to read frame {frame_number if is_timed_capture else 'N/A'}. "
                    f"Attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}. Waiting {wait_time}s..."
                )

                self.cap.release()
                time.sleep(wait_time)

                self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

                if not self.cap.isOpened():
                    print("🛑 Reconnect failed. Continuing to next attempt.")
                    continue
                else:
                    print("✅ Reconnect successful! Resuming stream.")
                    reconnect_attempts = 0

                continue

            reconnect_attempts = 0

            # --- Annotation and Writing ---

            if box_generator:
                try:
                    boxes, labels, colors = box_generator(frame, frame_number)
                except Exception as e:
                    print(
                        f"\n⚠️ Error in box_generator at frame {frame_number}: {e}. Skipping annotation for this frame."
                    )
                    boxes, labels, colors = [], [], []
            else:
                boxes, labels, colors = self._generate_example_boxes(frame_number)

            annotated = self.annotate_frame(frame, boxes, labels, colors)

            tracked_points = self._generate_example_tracked_points(boxes, frame_number)
            expanded = self.create_expanded_frame(annotated, tracked_points)

            try:
                self.out.write(annotated)
                self.out_expanded.write(expanded)
            except Exception as e:
                print(
                    f"\n🛑 Fatal I/O Error writing frame {frame_number}: {e}. Stopping capture to prevent data corruption."
                )
                break

            # 🖼️ Display the stream (Preview Window)
            window_title = (
                "Live Stream (Auto-Stop)" if is_timed_capture else "Live Stream (Manual Stop)"
            )
            cv2.imshow(window_title, expanded)
            cv2.waitKey(1)

            # --- Progress Update ---
            if is_timed_capture:
                if frame_number % self.fps == 0 or frame_number == target_frames - 1:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    if target_frames > 0:
                        progress_percent = (frame_number / target_frames) * 100
                        print(
                            f"Progress: {progress_percent:.1f}% | Frames: {frame_number}/{target_frames} | Elapsed Time: {elapsed:.1f}s",
                            end="\r",
                        )
                    else:
                        print(
                            f"Processing frame {frame_number} | Elapsed Time: {elapsed:.1f}s",
                            end="\r",
                        )
            else:
                if frame_number % (self.fps * 10) == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    print(
                        f"Running continuously... | Frames: {frame_number} | Elapsed Time: {elapsed:.1f}s | Status: Live",
                        end="\r",
                    )

            frame_number += 1

        print("\n", end="")

        end_time = time.time()
        elapsed_time = end_time - start_time

        if frame_number > 0:
            avg_fps = frame_number / elapsed_time if elapsed_time > 0 else 0

            print(f"\n✅ Capture finished. Total frames processed: {frame_number}")
            print(
                f"Recording duration: {elapsed_time:.2f}s "
                + (f"(Target: {duration_seconds}s)" if is_timed_capture else "")
            )
            print(f"Recorded at {self.fps} FPS. Actual stream processing speed: {avg_fps:.2f} FPS.")
        else:
            print("\n✅ Capture finished with 0 frames processed.")

        self.cleanup()
        print("Stream processing complete!")

    def _generate_example_boxes(self, frame_number: int) -> tuple[list, list, list]:
        """Generate example bounding boxes for demonstration."""
        # Example: Moving box across frame
        x = 50 + (frame_number * 2) % (self.frame_width - 100)

        # Simulate depth movement
        y_close = self.frame_height - 100
        y = y_close - (frame_number % 200)

        boxes = [
            (x, int(y), x + 80, int(y + 120)),  # Moving Box
            (200, 300, 280, 380),  # Static box
        ]
        labels = ["Mover", "Static"]
        colors = [(0, 255, 0), (255, 0, 0)]  # Green, Blue

        return boxes, labels, colors

    def _generate_example_tracked_points(
        self, boxes: list, frame_number: int
    ) -> list[tuple[int, int]]:
        """Generate tracked points for side panel using Geometry projection."""
        if not boxes:
            return []

        # Check if provider is valid and ready for projection
        if self.geo and self.geo.is_valid():
            # Extract "Feet" points (bottom center of box)
            feet_points = []
            for box in boxes:
                x1, y1, x2, y2 = box
                feet_x = int((x1 + x2) / 2)
                feet_y = y2
                feet_points.append((feet_x, feet_y))

            # Check if provider has extended interface with project_points_to_map
            if isinstance(self.geo, HomographyProviderExtended):
                # Use new extended interface for batch projection
                map_coords = self.geo.project_points_to_map(feet_points)
                # Convert MapCoordinate objects to pixel coordinates
                # Get pixels_per_meter from provider, default to 100.0 if not available
                pixels_per_meter = getattr(self.geo, "pixels_per_meter", 100.0)
                map_points = []
                for coord in map_coords:
                    x_px = int((coord.x * pixels_per_meter) + (self.side_panel_width // 2))
                    y_px = int(self.frame_height - (coord.y * pixels_per_meter))
                    map_points.append((x_px, y_px))
                return map_points
            elif hasattr(self.geo, "project_image_to_map"):
                # Fall back to legacy method if available (for CameraGeometry compatibility)
                return self.geo.project_image_to_map(
                    feet_points, self.side_panel_width, self.frame_height
                )

        # Fallback: simple scaling if homography not available
        points = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            mapped_x = int((center_x / self.frame_width) * self.side_panel_width)
            mapped_y = int((center_y / self.frame_height) * self.frame_height)
            points.append((mapped_x, mapped_y))

        return points

    def cleanup(self):
        """Release video capture and writers."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        if self.out_expanded:
            self.out_expanded.release()

        cv2.waitKey(1)
        cv2.destroyAllWindows()


def annotate_stream_with_args(
    rtsp_url: str,
    duration: float,
    output_path: str,
    side_panel_width: int = 640,
    side_panel_image: str | None = None,
    # New geometry arguments
    zoom: float = 1.0,
    pan: float = 0.0,
    tilt: float = -45.0,
    height: float = 5.0,
    # Homography configuration
    homography_config: HomographyConfig | None = None,
    # Camera info for GPS coordinates
    cam_info: dict | None = None,
):
    """
    Main entry point for stream-based processing with homography.

    Args:
        rtsp_url: RTSP stream URL
        duration: Recording duration in seconds
        output_path: Path to save output video
        side_panel_width: Width of side panel in pixels
        side_panel_image: Optional path to side panel image
        zoom: Camera zoom factor (e.g., 1.0 = no zoom)
        pan: Camera pan angle in degrees (positive = right)
        tilt: Camera tilt angle in degrees (positive = down, Hikvision convention)
        height: Camera height above ground in meters
        homography_config: Optional configuration for homography approach (uses default if not provided)
        cam_info: Optional camera configuration dict containing GPS coordinates (lat, lon in DMS format)
    """

    annotator = VideoAnnotator(
        video_path=None,
        output_path=output_path,
        side_panel_width=side_panel_width,
        side_panel_image=side_panel_image,
        homography_config=homography_config,
    )

    # 1. Load the stream to set dimensions (W and H) in annotator.geo
    if not annotator.load_stream(rtsp_url):
        return

    # 2. Geometry Setup (Requires initialized annotator.geo)
    if annotator.geo:
        # Check if this is an IntrinsicExtrinsicHomography provider
        if isinstance(annotator.geo, IntrinsicExtrinsicHomography):
            # A. Calculate Intrinsics (K)
            K = annotator.geo.get_intrinsics(
                zoom_factor=zoom, width_px=annotator.frame_width, height_px=annotator.frame_height
            )

            # B. World Position Setup
            # Camera is placed at world origin [0, 0, height].
            # GPS coordinates (if available) are used for geo-referencing projected points.
            camera_position = np.array([0.0, 0.0, height])

            # Warn if GPS_BASED_ORIGIN mode was requested (not yet implemented)
            coord_mode = (
                homography_config.coordinate_system_mode
                if homography_config
                else CoordinateSystemMode.ORIGIN_AT_CAMERA
            )
            if coord_mode == CoordinateSystemMode.GPS_BASED_ORIGIN:
                logger.warning(
                    "GPS_BASED_ORIGIN mode is not yet implemented. "
                    "Using ORIGIN_AT_CAMERA mode instead."
                )

            # Parse GPS coordinates from cam_info if available
            camera_gps_lat = None
            camera_gps_lon = None
            if cam_info and "lat" in cam_info and "lon" in cam_info:
                # Validate that lat/lon values are strings before parsing
                if not isinstance(cam_info["lat"], str) or not isinstance(cam_info["lon"], str):
                    logger.warning(
                        "GPS coordinates must be strings in DMS format, got lat=%s, lon=%s",
                        type(cam_info["lat"]).__name__,
                        type(cam_info["lon"]).__name__,
                    )
                    print("Warning: Invalid GPS coordinate type - geo-referencing disabled")
                else:
                    try:
                        camera_gps_lat = dms_to_dd(cam_info["lat"])
                        camera_gps_lon = dms_to_dd(cam_info["lon"])
                        print(f"Camera GPS Position: {camera_gps_lat:.6f}°, {camera_gps_lon:.6f}°")
                    except ValueError as e:
                        logger.warning(f"Failed to parse GPS coordinates: {e}")
                        print("Warning: Invalid GPS coordinate format - geo-referencing disabled")

            print(f"Camera World Position (Xw, Yw, Zw): {camera_position} meters")
            print(f"Intrinsic Matrix K:\n{K}")

            # C. Set GPS position BEFORE compute_homography (required for geo-referencing)
            if camera_gps_lat is not None and camera_gps_lon is not None:
                try:
                    annotator.geo.set_camera_gps_position(camera_gps_lat, camera_gps_lon)
                    print("  GPS geo-referencing: enabled")
                except ValueError as e:
                    logger.warning(f"Failed to set GPS position: {e}")
                    print("  GPS geo-referencing: disabled (coordinates out of valid range)")
            else:
                print("  GPS geo-referencing: disabled (no coordinates provided)")

            # D. Compute homography using the new interface
            # NOTE: The tilt value from Hikvision cameras uses the convention where
            # positive = camera pointing down, which matches our geometry code expectations.
            # No normalization is needed.
            reference = {
                "camera_matrix": K,
                "camera_position": camera_position,
                "pan_deg": pan,
                "tilt_deg": tilt,  # Hikvision convention: positive = down
                "map_width": side_panel_width,
                "map_height": annotator.frame_height,
            }

            # Compute homography (frame not needed for intrinsic/extrinsic approach)
            result = annotator.geo.compute_homography(
                frame=np.zeros((annotator.frame_height, annotator.frame_width, 3), dtype=np.uint8),
                reference=reference,
            )

            print(f"Homography computed successfully (confidence: {result.confidence:.2f})")
            print(f"  Approach: {result.metadata.get('approach', 'unknown')}")
            print(f"  Determinant: {result.metadata.get('determinant', 'N/A')}")
        else:
            # For other homography providers, they may have different setup requirements
            print(f"Using homography provider: {type(annotator.geo).__name__}")
            print("Note: Camera parameters setup may vary for different homography approaches")

    # 3. Process the stream
    annotator.process_stream(rtsp_url, duration)


def usage_example_stream(cam_name: str, duration: float):
    """Demonstrates how to use the stream annotation functionality."""

    # 1. Get camera configuration
    cam_info = get_camera_by_name(cam_name)
    if not cam_info:
        print(f"Error: Camera '{cam_name}' not found in configuration")
        return

    RTSP_URL = get_rtsp_url(cam_name)

    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    BASE_FILENAME = "output_stream_annotated"
    FILE_EXTENSION = "mp4"

    # The format YYYYMMDD_HHMMSS is good for sorting and uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # The resulting string will look like "output/Valte_output_stream_annotated_20251121_182207.mp4"
    OUTPUT_PATH = str(output_dir / f"{cam_name}_{BASE_FILENAME}_{timestamp}.{FILE_EXTENSION}")
    SIDE_PANEL_WIDTH = 640
    SIDE_PANEL_IMAGE = None
    CAPTURE_DURATION_SECONDS = duration

    # get camera status
    camera = HikvisionPTZ(
        ip=cam_info["ip"], username=USERNAME, password=PASSWORD, name=cam_info["name"]
    )

    status = camera.get_status()
    print(f"Camera '{cam_name}' Status: {status}")

    # Get height from config (with fallback to default)
    CAMERA_HEIGHT_M = cam_info.get("height_m", 5.0)

    print("\n--- Running Stream Annotation Example ---")
    print(f"Camera: {cam_name}")
    print(f"Capture Duration: {CAPTURE_DURATION_SECONDS} seconds. The process will auto-stop.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"        {OUTPUT_PATH.replace('.mp4', '_expanded.mp4')}")

    if "192.168.1.100" in RTSP_URL:
        print(
            "\nWARNING: Placeholder IP detected. Please update RTSP_URL with your camera's actual address to run this example."
        )
        return

    annotate_stream_with_args(
        rtsp_url=RTSP_URL,
        duration=CAPTURE_DURATION_SECONDS,
        output_path=OUTPUT_PATH,
        side_panel_width=SIDE_PANEL_WIDTH,
        side_panel_image=SIDE_PANEL_IMAGE,
        zoom=status["zoom"],
        pan=status["pan"],
        tilt=status["tilt"],
        height=CAMERA_HEIGHT_M,
        cam_info=cam_info,
    )


def main():
    # Only run the stream example here
    usage_example_stream("Valte", 10.0)  # Capture for 10 seconds
    usage_example_stream("Setram", 10.0)  # Capture for 10 seconds


if __name__ == "__main__":
    main()
