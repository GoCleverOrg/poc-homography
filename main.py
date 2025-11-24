# main.py

import datetime
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import time
import math # Needed for camera_geometry
# Import the new classes
from camera_geometry import CameraGeometry
from coordinates_converter import CoordinatesConverter
from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ 

# -----------------------------------------------------------

# Camera configurations
CAMERAS = [
    {"ip": "10.207.99.178", "name": "Valte", "lat": "39°38'25.7\"N", "lon": "0°13'48.7\"W"},
    {"ip": "10.237.100.15", "name": "Setram", "lat": "41°19'46.8\"N", "lon": "2°08'31.3\"E"},
]

def get_camera_by_name(camera_name: str, camera_list: list) -> dict | None:
    """
    Finds a camera dictionary in a list by its 'name' value.

    Args:
        camera_name: The name of the camera to search for.
        camera_list: The list of camera dictionaries.

    Returns:
        The camera dictionary if found, otherwise None.
    """
    # Use a generator expression inside next().
    # It searches for the first dictionary 'camera' where its 'name' key 
    # matches the provided 'camera_name'.
    # If no match is found, 'None' (the second argument to next()) is returned.
    return next((camera for camera in camera_list if camera.get("name") == camera_name), None)

USERNAME = "admin"
PASSWORD = "CameraLab01*"

class VideoAnnotator:
    """
    Main class for video/stream loading, annotation, and output with expanded canvas.
    Supports both video files and live RTSP streams.
    """

    def __init__(self, video_path: Optional[str] = None, output_path: str = "output_annotated.mp4", 
                 side_panel_width: int = 640, side_panel_image: Optional[str] = None):
        """
        Initialize video annotator.

        Args:
            video_path: Path to input video file or None for stream mode.
            output_path: Path for output annotated video
            side_panel_width: Width of side panel in pixels (for top-down view)
            side_panel_image: Optional path to image for side panel (e.g., floor plan)
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
        
        # Geometry Engine: Initialized after frame size is known
        self.geo: Optional[CameraGeometry] = None

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
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video loaded successfully:")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
        
        # Initialize Geometry Engine
        self.geo = CameraGeometry(self.frame_width, self.frame_height)
        
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
        self.fps = 30 # Default FPS for recording.

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
        self.total_frames = -1 # Not applicable for stream

        print(f"RTSP stream loaded successfully:")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  Recording FPS: {self.fps} (Actual stream speed varies)")
        
        # Initialize Geometry Engine
        self.geo = CameraGeometry(self.frame_width, self.frame_height)

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
        self.side_panel_template = np.ones((self.frame_height, self.side_panel_width, 3), 
                                           dtype=np.uint8) * 240  # Light gray

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
            cv2.line(self.side_panel_template, (0, i), (self.side_panel_width, i), (200, 200, 200), 1)
        for i in range(0, self.side_panel_width, grid_spacing):
            cv2.line(self.side_panel_template, (i, 0), (i, self.frame_height), (200, 200, 200), 1)

        # Add label
        cv2.putText(self.side_panel_template, "Top-Down View (Homography)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def setup_writers(self):
        """Setup video writers for both normal and expanded output."""
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

        # Writer for normal annotated video
        self.out = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )

        # Writer for expanded canvas (video + side panel)
        expanded_path = self.output_path.replace('.mp4', '_expanded.mp4')
        expanded_width = self.frame_width + self.side_panel_width
        self.out_expanded = cv2.VideoWriter(
            expanded_path,
            fourcc,
            self.fps,
            (expanded_width, self.frame_height)
        )

        print(f"Output writers initialized:")
        print(f"  Normal: {self.output_path}")
        print(f"  Expanded: {expanded_path}")

    def annotate_frame(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                      labels: Optional[List[str]] = None, 
                      colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
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
            cv2.circle(annotated, (foot_x, foot_y), 5, (0,0,255), -1)

            # Draw label if provided
            if labels and idx < len(labels):
                label = labels[idx]
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(annotated, (x1, y1 - label_height - 10), 
                            (x1 + label_width + 10, y1), color, -1)
                # Draw label text
                cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def create_expanded_frame(self, annotated_frame: np.ndarray, 
                             tracked_points: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
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
            (label_w, label_h), _ = cv2.getTextSize("cam", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_x = cam_px - (label_w // 2)
            label_y = cam_py - 12
            cv2.putText(side_panel, "cam", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except Exception:
            # Fail silently if mapping/drawing fails for any reason
            pass
        # Plot tracked points if provided — position them relative to the camera and zoom to fit
        if tracked_points:
            try:
                h_sp, w_sp = side_panel.shape[:2]
                desired_cam_px = w_sp // 2
                desired_cam_py = h_sp - 30

                # Compute original camera position in side-panel pixels (if available)
                if self.geo and getattr(self.geo, 'w_pos', None) is not None:
                    cam_orig_px, cam_orig_py = self.geo.world_to_map(
                        float(self.geo.w_pos[0]), float(self.geo.w_pos[1]),
                        sw=self.side_panel_width, sh=self.frame_height
                    )
                else:
                    # If geometry is not available, assume camera was at bottom-center originally
                    cam_orig_px, cam_orig_py = desired_cam_px, desired_cam_py

                # Compute deltas from camera original position
                deltas = []
                max_dx = 0.0
                max_dy = 0.0
                map_center_x = self.side_panel_width // 2
                map_bottom_y = self.frame_height

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

                sx = (half_width_available / max_dx) if max_dx > 0 else float('inf')
                sy = (top_available / max_dy) if max_dy > 0 else float('inf')
                scale = min(1.0, sx, sy)

                # Draw transformed points
                for dx, dy in deltas:
                    new_x = int(desired_cam_px + dx * scale)
                    new_y = int(desired_cam_py + dy * scale)
                    cv2.circle(side_panel, (new_x, new_y), 5, (0, 0, 255), -1)
                    cv2.circle(side_panel, (new_x, new_y), 8, (255, 255, 255), 1)
            except Exception:
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
                progress = (frame_number / self.total_frames) * 100
                print(f"Progress: {progress:.1f}% (frame {frame_number}/{self.total_frames})")

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
            print(f"\n🎥 Starting TIMED capture for {duration_seconds:.2f} seconds ({target_frames} frames)...")
        else:
            print("\n🔄 Starting CONTINUOUS stream capture (runs until stopped externally)...")

        # --- Main Processing Loop Condition ---
        while (is_timed_capture and frame_number < target_frames) or (not is_timed_capture):
            
            if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                print(f"\n❌ Max reconnection attempts ({MAX_RECONNECT_ATTEMPTS}) reached. Stopping capture.")
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
                
                print(f"🚨 Stream lost or failed to read frame {frame_number if is_timed_capture else 'N/A'}. "
                      f"Attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}. Waiting {wait_time}s...")
                
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
                    print(f"\n⚠️ Error in box_generator at frame {frame_number}: {e}. Skipping annotation for this frame.")
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
                print(f"\n🛑 Fatal I/O Error writing frame {frame_number}: {e}. Stopping capture to prevent data corruption.")
                break
            
            # 🖼️ Display the stream (Preview Window)
            window_title = 'Live Stream (Auto-Stop)' if is_timed_capture else 'Live Stream (Manual Stop)'
            cv2.imshow(window_title, expanded)
            cv2.waitKey(1) 

            # --- Progress Update ---
            if is_timed_capture:
                if frame_number % self.fps == 0 or frame_number == target_frames - 1:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    progress_percent = (frame_number / target_frames) * 100
                    print(f"Progress: {progress_percent:.1f}% | Frames: {frame_number}/{target_frames} | Elapsed Time: {elapsed:.1f}s", end='\r')
            else:
                if frame_number % (self.fps * 10) == 0: 
                    current_time = time.time()
                    elapsed = current_time - start_time
                    print(f"Running continuously... | Frames: {frame_number} | Elapsed Time: {elapsed:.1f}s | Status: Live", end='\r')

            frame_number += 1
            
        print("\n", end='') 

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if frame_number > 0:
            avg_fps = frame_number / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n✅ Capture finished. Total frames processed: {frame_number}")
            print(f"Recording duration: {elapsed_time:.2f}s " + (f"(Target: {duration_seconds}s)" if is_timed_capture else ""))
            print(f"Recorded at {self.fps} FPS. Actual stream processing speed: {avg_fps:.2f} FPS.")
        else:
            print("\n✅ Capture finished with 0 frames processed.")

        self.cleanup()
        print("Stream processing complete!")

    def _generate_example_boxes(self, frame_number: int) -> Tuple[List, List, List]:
        """Generate example bounding boxes for demonstration."""
        # Example: Moving box across frame
        x = 50 + (frame_number * 2) % (self.frame_width - 100)
        
        # Simulate depth movement
        y_close = self.frame_height - 100
        y = y_close - (frame_number % 200)

        boxes = [
            (x, int(y), x + 80, int(y + 120)),  # Moving Box
            (200, 300, 280, 380)      # Static box 
        ]
        labels = ["Mover", "Static"]
        colors = [(0, 255, 0), (255, 0, 0)]  # Green, Blue

        return boxes, labels, colors

    def _generate_example_tracked_points(self, boxes: List, frame_number: int) -> List[Tuple[int, int]]:
        """Generate tracked points for side panel using Geometry projection."""
        if not boxes:
            return []

        if self.geo and self.geo.H is not None and not np.all(self.geo.H == np.eye(3)):
            # Extract "Feet" points (bottom center of box)
            feet_points = []
            for box in boxes:
                x1, y1, x2, y2 = box
                feet_x = int((x1 + x2) / 2)
                feet_y = y2 
                feet_points.append((feet_x, feet_y))
            
            # Project using Geometry
            return self.geo.project_image_to_map(feet_points, 
                                         self.side_panel_width, 
                                         self.frame_height)
        
        # Fallback
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


def annotate_stream_with_args(rtsp_url: str, duration: float, output_path: str, side_panel_width: int = 640,
                              side_panel_image: Optional[str] = None,
                              # New geometry arguments
                              zoom: float = 1.0, pan: float = 0.0, tilt: float = -45.0, height: float = 5.0):
    """Main entry point for stream-based processing."""
    
    annotator = VideoAnnotator(
        video_path=None, 
        output_path=output_path,
        side_panel_width=side_panel_width,
        side_panel_image=side_panel_image
    )
    
    # 1. Load the stream to set dimensions (W and H) in annotator.geo
    if not annotator.load_stream(rtsp_url):
        return

    # 2. Geometry Setup (Requires initialized annotator.geo)
    if annotator.geo:
        # A. Calculate Intrinsics (K)
        K = annotator.geo.get_intrinsics(zoom_factor=zoom, W_px=annotator.frame_width, H_px=annotator.frame_height)
        
        # B. Convert Geographic W_pos to Local World Coordinates (X, Y, Z meters)
        
        # Convert lat/lon to Decimal Degrees first
        lat_str = "41°19'46.8\"N"
        lon_str = "2°08'31.3\"E"
        lat_dd = CoordinatesConverter.dms_to_dd(lat_str)
        lon_dd = CoordinatesConverter.dms_to_dd(lon_str)
        
        # ASSUMPTION: Convert geographic coordinates (lat/lon) to a local meter-based
        # coordinate system (Xw, Yw) using a simplified projection (e.g., small area flat Earth).
        # For this example, we arbitrarily define the camera's X, Y location as 
        # a meter offset from a fictional origin, keeping the relative height (Z) correct.
        
        # Let's use 10 meters per 0.0001 DD change for a rough local projection.
        # This is a DUMMY CONVERSION for demonstration.
        X_meter = (lon_dd - 2.0) * 100000.0 * 0.01  
        Y_meter = (lat_dd - 41.0) * 100000.0 * 0.01 
        
        # World Position Vector (X, Y, Z meters). Z is the height.
        w_pos = np.array([X_meter, Y_meter, height])
        
        print(f"Geometric World Position (Xw, Yw, Zw): {w_pos}")
        print(f"Intrinsic Matrix K:\n{K}")
        
        # C. Set parameters and calculate Homography H
        annotator.geo.set_camera_parameters(
            K=K, 
            w_pos=w_pos, 
            pan_deg=pan, 
            tilt_deg=tilt, 
            map_width=side_panel_width, 
            map_height=annotator.frame_height
        )

    # 3. Process the stream
    annotator.process_stream(rtsp_url, duration)


def usage_example_stream(cam_name:str, duration: float):
    """Demonstrates how to use the stream annotation functionality."""

   # 1. Successful lookup
    cam_info = get_camera_by_name(cam_name, CAMERAS)

    RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{cam_info['ip']}:554/Streaming/Channels/101"
       
    BASE_FILENAME = "output_stream_annotated"
    FILE_EXTENSION = "mp4"

    # The format YYYYMMDD_HHMMSS is good for sorting and uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # The resulting string will look like "output_stream_annotated_20251121_182207.mp4"
    OUTPUT_PATH = f"{cam_name}_{BASE_FILENAME}_{timestamp}.{FILE_EXTENSION}"
    SIDE_PANEL_WIDTH = 640
    SIDE_PANEL_IMAGE = None
    CAPTURE_DURATION_SECONDS = duration

    # get camera status
    camera = HikvisionPTZ(
            ip=cam_info["ip"],
            username=USERNAME,
            password=PASSWORD,
            name=cam_info["name"]
    )

    status = camera.get_status()
    print(f"Camera '{cam_name}' Status: {status}")  

    # NEW GEOMETRY PARAMETERS
    CAMERA_HEIGHT_M = 5.0 # 5 meters height

    print("\n--- Running Stream Annotation Example ---")
    print(f"Capture Duration: {CAPTURE_DURATION_SECONDS} seconds. The process will auto-stop.")
    
    if "192.168.1.100" in RTSP_URL:
        print("\nWARNING: Placeholder IP detected. Please update RTSP_URL with your camera's actual address to run this example.")
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
        height=CAMERA_HEIGHT_M
    )

def main():
    # Only run the stream example here
    usage_example_stream("Valte", 10.0)  # Capture for 10 seconds
    usage_example_stream("Setram", 10.0)  # Capture for 10 seconds


if __name__ == "__main__":
    main()