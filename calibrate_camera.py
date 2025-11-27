#!/usr/bin/env python3
"""
Camera Calibration Tool using Checkerboard Pattern

This tool calibrates cameras to compute intrinsic parameters (camera matrix K)
and distortion coefficients using the checkerboard pattern detection method.

Supports two modes:
  1. Capture mode: Connect to camera RTSP stream and capture calibration images interactively
  2. Directory mode: Process existing calibration images from a directory

Usage Examples:
  # Capture mode - interactive capture from RTSP stream
  python calibrate_camera.py --camera Valte --mode capture --images ./calibration_images

  # Directory mode - process existing images
  python calibrate_camera.py --images ./calibration_images --output calibration.json

  # Custom pattern size and square size
  python calibrate_camera.py --camera Setram --mode capture --pattern 7x5 --square-size 30.0

Requirements:
  - Camera must be configured in camera_config.py for RTSP capture mode
  - Checkerboard pattern with known dimensions
  - Multiple images (default minimum: 10) with good coverage of the image area
  - Clear, well-lit checkerboard images for accurate corner detection

Output:
  - JSON file containing camera matrix, distortion coefficients, and calibration metadata
  - Reprojection error metric for quality assessment
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import cv2
import numpy as np

# Import camera configuration utilities
try:
    from camera_config import get_camera_by_name, get_rtsp_url
except ImportError:
    print("Warning: camera_config.py not found. RTSP capture mode will not be available.")
    get_camera_by_name = None
    get_rtsp_url = None


class CameraCalibrator:
    """
    Handles camera calibration using checkerboard pattern detection.

    Supports both interactive capture from RTSP streams and processing
    of existing calibration images from a directory.
    """

    def __init__(self, pattern_size: Tuple[int, int], square_size_mm: float, min_images: int = 10):
        """
        Initialize the camera calibrator.

        Args:
            pattern_size: Number of inner corners (width, height) in the checkerboard
            square_size_mm: Size of each checkerboard square in millimeters
            min_images: Minimum number of images required for calibration
        """
        self.pattern_size = pattern_size
        self.square_size_mm = square_size_mm
        self.min_images = min_images

        # Storage for calibration data
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane
        self.image_size = None

        # Calibration results
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.rms_error = None

        # Create 3D object points for the checkerboard pattern
        # Pattern is on Z=0 plane with coordinates in millimeters
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size_mm

    def capture_images_from_rtsp(self, rtsp_url: str, save_dir: Path) -> int:
        """
        Capture calibration images interactively from RTSP stream.

        Shows live feed with detected corners. User presses 'c' to capture
        when corners are detected, and 'q' to quit and proceed with calibration.

        Args:
            rtsp_url: RTSP stream URL
            save_dir: Directory to save captured images

        Returns:
            Number of images successfully captured
        """
        # Create save directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)

        # Open RTSP stream
        print(f"\nConnecting to RTSP stream: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream at {rtsp_url}")
            return 0

        print("Stream opened successfully!")
        print("\nCapture Instructions:")
        print("  - Move the checkerboard to different positions and angles")
        print("  - Press 'c' to capture when corners are detected (green overlay)")
        print("  - Press 'q' to quit and proceed with calibration")
        print(f"  - Minimum required images: {self.min_images}")

        capture_count = 0
        window_name = "Camera Calibration - Capture Mode"

        # Define criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("\nWarning: Failed to read frame from stream")
                break

            # Store image size from first frame
            if self.image_size is None:
                h, w = frame.shape[:2]
                self.image_size = (w, h)

            # Convert to grayscale for corner detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(
                gray,
                self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            # Create display frame
            display_frame = frame.copy()

            # Draw corners if detected
            if ret_corners:
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Draw the corners
                cv2.drawChessboardCorners(display_frame, self.pattern_size, corners_refined, ret_corners)

                # Add status text
                status_text = f"Corners detected! Press 'c' to capture"
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Add status text
                status_text = "No corners detected - adjust checkerboard position"
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add capture count
            count_text = f"Captured: {capture_count} / {self.min_images} minimum"
            cv2.putText(display_frame, count_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the frame
            cv2.imshow(window_name, display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting capture mode...")
                break
            elif key == ord('c') and ret_corners:
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_path = save_dir / f"calibration_{timestamp}.jpg"
                cv2.imwrite(str(image_path), frame)

                # Store corner data
                self.object_points.append(self.objp)
                self.image_points.append(corners_refined)

                capture_count += 1
                print(f"Captured image {capture_count}: {image_path.name}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        print(f"\nCapture complete! Total images captured: {capture_count}")
        return capture_count

    def process_images_from_directory(self, image_dir: Path) -> int:
        """
        Process existing calibration images from a directory.

        Args:
            image_dir: Directory containing calibration images

        Returns:
            Number of images successfully processed
        """
        if not image_dir.exists():
            print(f"Error: Image directory does not exist: {image_dir}")
            return 0

        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(ext))

        if not image_files:
            print(f"Error: No image files found in {image_dir}")
            return 0

        print(f"\nProcessing {len(image_files)} images from: {image_dir}")
        print(f"Checkerboard pattern: {self.pattern_size[0]}x{self.pattern_size[1]} inner corners")

        # Define criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        success_count = 0

        for idx, image_path in enumerate(sorted(image_files), 1):
            print(f"Processing [{idx}/{len(image_files)}]: {image_path.name}...", end=" ")

            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                print("Failed to load")
                continue

            # Store image size from first successful image
            if self.image_size is None:
                h, w = img.shape[:2]
                self.image_size = (w, h)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret:
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Store the data
                self.object_points.append(self.objp)
                self.image_points.append(corners_refined)

                success_count += 1
                print("OK")
            else:
                print("No corners detected")

        print(f"\nSuccessfully processed {success_count}/{len(image_files)} images")
        return success_count

    def calibrate(self) -> bool:
        """
        Perform camera calibration using collected image and object points.

        Returns:
            True if calibration successful, False otherwise
        """
        if len(self.object_points) < self.min_images:
            print(f"\nError: Insufficient images for calibration")
            print(f"  Required: {self.min_images}")
            print(f"  Collected: {len(self.object_points)}")
            return False

        if self.image_size is None:
            print("\nError: Image size not determined")
            return False

        print(f"\nPerforming camera calibration...")
        print(f"  Images: {len(self.object_points)}")
        print(f"  Image size: {self.image_size[0]}x{self.image_size[1]}")
        print(f"  Pattern: {self.pattern_size[0]}x{self.pattern_size[1]} inner corners")
        print(f"  Square size: {self.square_size_mm}mm")

        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None,
            None
        )

        if not ret:
            print("\nCalibration failed!")
            return False

        # Store results
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = dist_coeffs
        self.rms_error = ret

        # Calculate per-image reprojection errors for quality assessment
        mean_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(
                self.object_points[i], rvecs[i], tvecs[i],
                camera_matrix, dist_coeffs
            )
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        mean_error /= len(self.object_points)

        print("\n" + "=" * 70)
        print("CALIBRATION SUCCESSFUL")
        print("=" * 70)
        print(f"\nCalibration Quality:")
        print(f"  RMS reprojection error: {self.rms_error:.4f} pixels")
        print(f"  Mean error per image:   {mean_error:.4f} pixels")

        # Quality assessment
        if self.rms_error < 0.5:
            quality = "Excellent"
        elif self.rms_error < 1.0:
            quality = "Good"
        elif self.rms_error < 2.0:
            quality = "Acceptable"
        else:
            quality = "Poor - consider recalibrating"
        print(f"  Quality assessment:     {quality}")

        print(f"\nCamera Matrix (K):")
        print(self.camera_matrix)

        print(f"\nDistortion Coefficients [k1, k2, p1, p2, k3]:")
        print(self.distortion_coeffs.ravel())

        # Calculate field of view
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        fov_x = 2 * np.arctan(self.image_size[0] / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(self.image_size[1] / (2 * fy)) * 180 / np.pi

        print(f"\nDerived Parameters:")
        print(f"  Focal length (fx, fy): ({fx:.2f}, {fy:.2f}) pixels")
        print(f"  Field of view (x, y):  ({fov_x:.2f}, {fov_y:.2f}) degrees")
        print(f"  Principal point (cx, cy): ({camera_matrix[0, 2]:.2f}, {camera_matrix[1, 2]:.2f})")

        return True

    def save_calibration(self, output_path: Path) -> bool:
        """
        Save calibration results to JSON file.

        Args:
            output_path: Path to output JSON file

        Returns:
            True if save successful, False otherwise
        """
        if self.camera_matrix is None or self.distortion_coeffs is None:
            print("\nError: No calibration data to save. Run calibration first.")
            return False

        # Prepare calibration data
        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coeffs": self.distortion_coeffs.ravel().tolist(),
            "image_size": list(self.image_size),
            "rms_error": float(self.rms_error),
            "num_images": len(self.object_points),
            "pattern_size": list(self.pattern_size),
            "square_size_mm": float(self.square_size_mm),
            "calibration_date": datetime.now().isoformat()
        }

        # Save to file
        try:
            with open(output_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"\nCalibration data saved to: {output_path}")
            return True
        except Exception as e:
            print(f"\nError saving calibration data: {e}")
            return False

    @staticmethod
    def load_calibration(input_path: Path) -> Optional[dict]:
        """
        Load calibration data from JSON file.

        Args:
            input_path: Path to calibration JSON file

        Returns:
            Dictionary containing calibration data, or None if load fails
        """
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)

            # Convert lists back to numpy arrays
            data['camera_matrix'] = np.array(data['camera_matrix'])
            data['distortion_coeffs'] = np.array(data['distortion_coeffs'])

            return data
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return None


def parse_pattern_size(pattern_str: str) -> Tuple[int, int]:
    """
    Parse pattern size string like "9x6" into tuple (9, 6).

    Args:
        pattern_str: Pattern size string in format "WIDTHxHEIGHT"

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If pattern string is invalid
    """
    try:
        parts = pattern_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError("Pattern must be in format WIDTHxHEIGHT (e.g., 9x6)")

        width = int(parts[0])
        height = int(parts[1])

        if width < 2 or height < 2:
            raise ValueError("Pattern dimensions must be at least 2x2")

        return (width, height)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid pattern size '{pattern_str}': {e}")


def main():
    """Main entry point for camera calibration tool."""

    parser = argparse.ArgumentParser(
        description="Camera Calibration Tool using Checkerboard Pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture mode - interactive capture from RTSP stream
  %(prog)s --camera Valte --mode capture --images ./calibration_images

  # Directory mode - process existing images
  %(prog)s --images ./calibration_images --output calibration.json

  # Custom pattern and square size
  %(prog)s --camera Setram --mode capture --pattern 7x5 --square-size 30.0

Notes:
  - For capture mode, camera must be configured in camera_config.py
  - Pattern size is the number of INNER corners (not squares)
  - For a standard 10x7 square checkerboard, use pattern 9x6
  - Ensure good lighting and multiple viewing angles for best results
        """
    )

    # Camera source arguments
    parser.add_argument('--camera', type=str,
                       help='Camera name from camera_config.py (for RTSP capture mode)')
    parser.add_argument('--mode', type=str, choices=['capture', 'directory'], default='directory',
                       help='Calibration mode: capture from RTSP or process directory')

    # Image arguments
    parser.add_argument('--images', type=str, required=True,
                       help='Path to images directory (input for directory mode, output for capture mode)')

    # Checkerboard pattern arguments
    parser.add_argument('--pattern', type=str, default='9x6',
                       help='Checkerboard pattern size as WIDTHxHEIGHT inner corners (default: 9x6)')
    parser.add_argument('--square-size', type=float, default=25.0,
                       help='Size of each checkerboard square in millimeters (default: 25.0)')

    # Calibration arguments
    parser.add_argument('--min-images', type=int, default=10,
                       help='Minimum number of images required for calibration (default: 10)')

    # Output arguments
    parser.add_argument('--output', type=str,
                       help='Output file path for calibration results (JSON format)')

    args = parser.parse_args()

    # Parse pattern size
    try:
        pattern_size = parse_pattern_size(args.pattern)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Setup paths
    images_path = Path(args.images)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default output path based on mode
        if args.mode == 'capture' and args.camera:
            output_path = Path(f"calibration_{args.camera}.json")
        else:
            output_path = Path("calibration.json")

    print("=" * 70)
    print("CAMERA CALIBRATION TOOL")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Pattern size: {pattern_size[0]}x{pattern_size[1]} inner corners")
    print(f"  Square size: {args.square_size}mm")
    print(f"  Minimum images: {args.min_images}")

    # Initialize calibrator
    calibrator = CameraCalibrator(
        pattern_size=pattern_size,
        square_size_mm=args.square_size,
        min_images=args.min_images
    )

    # Execute based on mode
    if args.mode == 'capture':
        # Capture mode requires camera parameter
        if not args.camera:
            print("\nError: --camera parameter required for capture mode")
            return 1

        # Check if camera_config is available
        if get_camera_by_name is None or get_rtsp_url is None:
            print("\nError: camera_config.py not found. Cannot use capture mode.")
            return 1

        # Get camera configuration
        camera_info = get_camera_by_name(args.camera)
        if not camera_info:
            print(f"\nError: Camera '{args.camera}' not found in camera_config.py")
            return 1

        # Get RTSP URL
        rtsp_url = get_rtsp_url(args.camera)
        if not rtsp_url:
            print(f"\nError: Could not generate RTSP URL for camera '{args.camera}'")
            return 1

        print(f"  Camera: {args.camera}")
        print(f"  RTSP URL: {rtsp_url}")
        print(f"  Save directory: {images_path}")

        # Capture images
        num_captured = calibrator.capture_images_from_rtsp(rtsp_url, images_path)

        if num_captured == 0:
            print("\nError: No images captured. Calibration aborted.")
            return 1

    else:
        # Directory mode
        print(f"  Images directory: {images_path}")

        # Process images from directory
        num_processed = calibrator.process_images_from_directory(images_path)

        if num_processed == 0:
            print("\nError: No images processed. Calibration aborted.")
            return 1

    # Perform calibration
    if not calibrator.calibrate():
        print("\nCalibration failed!")
        return 1

    # Save results
    if not calibrator.save_calibration(output_path):
        print("\nFailed to save calibration results!")
        return 1

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_path}")
    print(f"\nYou can now use this calibration file with the homography pipeline.")
    print(f"To update camera_config.py, add the distortion coefficients:")
    print(f"  distortion_coeffs: {calibrator.distortion_coeffs.ravel().tolist()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
