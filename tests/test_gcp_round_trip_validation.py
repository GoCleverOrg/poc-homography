#!/usr/bin/env python3
"""
GCP Round-Trip Validation Test.

This test validates that the GCP-based homography correctly maps image points
back to their original GPS coordinates. It's a self-consistency check that
verifies the homography computation is mathematically correct.

Test Logic:
1. Load GCPs and camera capture context from config
2. Move camera to the PTZ position where GCPs were captured
3. Grab a frame and display GCP points overlaid for visual verification
4. Compute homography from those GCPs
5. For each GCP, project its image point (u, v) through the homography
6. Compare returned GPS with original GCP GPS
7. Report reprojection error in meters

Usage:
    # Use config with bundled camera context (recommended)
    python tests/test_gcp_round_trip_validation.py --config config/homography_config.yaml

    # Override camera or use saved frame
    python tests/test_gcp_round_trip_validation.py --config config.yaml --camera Valte
    python tests/test_gcp_round_trip_validation.py --config config.yaml --frame path/to/frame.jpg

    # Math-only test (no visual)
    python tests/test_gcp_round_trip_validation.py --no-visual
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from poc_homography.feature_match_homography import FeatureMatchHomography
from poc_homography.homography_config import HomographyConfig

# Camera config is optional - only needed for --camera option
try:
    from poc_homography.camera_config import get_camera_by_name, get_rtsp_url
    CAMERA_CONFIG_AVAILABLE = True
except (ValueError, ImportError) as e:
    CAMERA_CONFIG_AVAILABLE = False
    get_camera_by_name = None
    get_rtsp_url = None
    print(f"Note: Camera config not available ({e})")
    print("  Visual validation with --camera will be disabled.")
    print("  Set CAMERA_USERNAME and CAMERA_PASSWORD env vars to enable.\n")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance in meters between two GPS points using Haversine formula.

    Args:
        lat1, lon1: First point (decimal degrees)
        lat2, lon2: Second point (decimal degrees)

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# Default test GCPs - a realistic grid pattern within 2560x1440 bounds
DEFAULT_TEST_GCPS = [
    # Top-left area
    {'gps': {'latitude': 39.640600, 'longitude': -0.230200}, 'image': {'u': 400.0, 'v': 300.0}},
    # Top-right area
    {'gps': {'latitude': 39.640620, 'longitude': -0.229800}, 'image': {'u': 2100.0, 'v': 320.0}},
    # Bottom-left area
    {'gps': {'latitude': 39.640200, 'longitude': -0.230180}, 'image': {'u': 420.0, 'v': 1100.0}},
    # Bottom-right area
    {'gps': {'latitude': 39.640180, 'longitude': -0.229820}, 'image': {'u': 2080.0, 'v': 1120.0}},
    # Center point
    {'gps': {'latitude': 39.640400, 'longitude': -0.230000}, 'image': {'u': 1280.0, 'v': 720.0}},
    # Additional point for robustness
    {'gps': {'latitude': 39.640500, 'longitude': -0.229900}, 'image': {'u': 1700.0, 'v': 500.0}},
]


def run_round_trip_validation(
    gcps: list,
    width: int = 2560,
    height: int = 1440,
    tolerance_meters: float = 5.0,
    verbose: bool = True,
    inliers_only: bool = True
) -> dict:
    """
    Run round-trip validation test on GCPs.

    Args:
        gcps: List of GCP dictionaries
        width: Image width in pixels
        height: Image height in pixels
        tolerance_meters: Maximum acceptable error in meters
        verbose: Print detailed output
        inliers_only: Only count inliers for pass/fail (outliers reported but not counted)

    Returns:
        Dictionary with test results
    """
    if verbose:
        print("=" * 60)
        print("GCP Round-Trip Validation Test")
        print("=" * 60)
        print(f"\nImage dimensions: {width} x {height}")
        print(f"Number of GCPs: {len(gcps)}")
        print(f"Tolerance: {tolerance_meters}m")
        print()

    # Create provider
    provider = FeatureMatchHomography(
        width=width,
        height=height,
        min_matches=4,
        ransac_threshold=5.0
    )

    # Compute homography
    if verbose:
        print("Computing homography from GCPs...")

    result = provider.compute_homography(
        frame=None,
        reference={'ground_control_points': gcps}
    )

    if not provider.is_valid():
        if verbose:
            print("ERROR: Failed to compute valid homography!")
        return {
            'success': False,
            'error': 'Failed to compute valid homography',
            'confidence': result.confidence
        }

    # Get inlier mask if available
    inlier_mask = result.metadata.get('inlier_mask', None)

    if verbose:
        print(f"Homography computed successfully!")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Inliers: {result.metadata.get('num_inliers', 'N/A')}/{len(gcps)}")
        print()

    # Test each GCP
    errors = []
    inlier_errors = []
    outlier_errors = []
    results_detail = []

    if verbose:
        print("Round-trip validation for each GCP:")
        print("-" * 60)

    for i, gcp in enumerate(gcps):
        is_inlier = inlier_mask[i] if inlier_mask is not None and i < len(inlier_mask) else None
        original_lat = gcp['gps']['latitude']
        original_lon = gcp['gps']['longitude']
        u = gcp['image']['u']
        v = gcp['image']['v']

        # Project image point to GPS
        try:
            world_point = provider.project_point((u, v))
            projected_lat = world_point.latitude
            projected_lon = world_point.longitude

            # Calculate error in meters
            error_m = haversine_distance(
                original_lat, original_lon,
                projected_lat, projected_lon
            )
            errors.append(error_m)

            # Track separately for inliers/outliers
            if is_inlier:
                inlier_errors.append(error_m)
            elif is_inlier is not None:
                outlier_errors.append(error_m)

            passed = error_m <= tolerance_meters
            status = "PASS" if passed else "FAIL"

            result_detail = {
                'gcp_index': i,
                'original_gps': (original_lat, original_lon),
                'projected_gps': (projected_lat, projected_lon),
                'image_point': (u, v),
                'error_meters': error_m,
                'passed': passed
            }
            results_detail.append(result_detail)

            if verbose:
                desc = gcp.get('metadata', {}).get('description', f'GCP {i+1}')
                inlier_str = " (INLIER)" if is_inlier else " (OUTLIER)" if is_inlier is not None else ""
                print(f"  [{status}] {desc}{inlier_str}")
                print(f"       Image point: ({u:.1f}, {v:.1f})")
                print(f"       Original GPS:  ({original_lat:.6f}, {original_lon:.6f})")
                print(f"       Projected GPS: ({projected_lat:.6f}, {projected_lon:.6f})")
                print(f"       Error: {error_m:.2f}m")
                print()

        except Exception as e:
            if verbose:
                print(f"  [ERROR] GCP {i+1}: {e}")
            results_detail.append({
                'gcp_index': i,
                'error': str(e),
                'passed': False
            })

    # Summary statistics
    if errors:
        mean_error = sum(errors) / len(errors)
        max_error = max(errors)
        min_error = min(errors)

        # Determine pass/fail based on mode
        if inliers_only and inlier_errors:
            check_errors = inlier_errors
            inlier_passed = sum(1 for e in inlier_errors if e <= tolerance_meters)
            all_passed = inlier_passed == len(inlier_errors)
        else:
            check_errors = errors
            passed_count = sum(1 for e in errors if e <= tolerance_meters)
            all_passed = passed_count == len(errors)

        if verbose:
            print("-" * 60)
            print("SUMMARY")
            print("-" * 60)
            print(f"  GCPs tested: {len(gcps)}")
            if inlier_errors:
                inlier_passed = sum(1 for e in inlier_errors if e <= tolerance_meters)
                inlier_mean = sum(inlier_errors) / len(inlier_errors)
                print(f"  Inliers: {len(inlier_errors)} ({inlier_passed}/{len(inlier_errors)} passed)")
                print(f"  Inlier mean error: {inlier_mean:.4f}m")
            if outlier_errors:
                outlier_mean = sum(outlier_errors) / len(outlier_errors)
                print(f"  Outliers: {len(outlier_errors)} (excluded from RANSAC fit)")
                print(f"  Outlier mean error: {outlier_mean:.2f}m")
            print(f"  Overall mean error: {mean_error:.2f}m")
            print()

            if inliers_only and inlier_errors:
                if all_passed:
                    print("RESULT: ALL INLIERS PASSED (outliers excluded)")
                else:
                    failed_inliers = len(inlier_errors) - inlier_passed
                    print(f"RESULT: {failed_inliers} INLIERS FAILED")
            else:
                if all_passed:
                    print("RESULT: ALL TESTS PASSED")
                else:
                    print(f"RESULT: {len(errors) - passed_count} TESTS FAILED")
            print("=" * 60)

        return {
            'success': all_passed,
            'gcps_tested': len(gcps),
            'inliers': len(inlier_errors),
            'outliers': len(outlier_errors),
            'inlier_mean_error_m': sum(inlier_errors) / len(inlier_errors) if inlier_errors else 0,
            'mean_error_m': mean_error,
            'max_error_m': max_error,
            'min_error_m': min_error,
            'confidence': result.confidence,
            'details': results_detail
        }
    else:
        return {
            'success': False,
            'error': 'No GCPs could be tested',
            'details': results_detail
        }


def load_gcp_config(config_path: str) -> dict:
    """
    Load GCPs and camera capture context from a YAML config file.

    Returns:
        Dictionary with:
        - 'gcps': List of GCP dictionaries
        - 'camera_capture_context': Camera context dict (or None)
        - 'image_width': Image width in pixels
        - 'image_height': Image height in pixels
    """
    config = HomographyConfig.from_yaml(config_path)

    result = {
        'gcps': None,
        'camera_capture_context': None,
        'image_width': 2560,
        'image_height': 1440,
    }

    # Try to get from feature_match config
    if 'feature_match' in config.approach_specific_config:
        fm_config = config.approach_specific_config['feature_match']

        if 'ground_control_points' in fm_config:
            result['gcps'] = fm_config['ground_control_points']

        if 'camera_capture_context' in fm_config:
            ctx = fm_config['camera_capture_context']
            result['camera_capture_context'] = ctx
            result['image_width'] = ctx.get('image_width', 2560)
            result['image_height'] = ctx.get('image_height', 1440)

    if result['gcps'] is None:
        raise ValueError("No ground_control_points found in config")

    return result


def load_gcps_from_config(config_path: str) -> list:
    """Load GCPs from a YAML config file (legacy function for compatibility)."""
    return load_gcp_config(config_path)['gcps']


def move_camera_to_position(camera_name: str, ptz_position: dict, wait_time: float = 3.0) -> bool:
    """
    Move camera to specified PTZ position.

    Args:
        camera_name: Name of the camera from camera_config
        ptz_position: Dict with 'pan', 'tilt', 'zoom' keys
        wait_time: Seconds to wait for camera to reach position

    Returns:
        True if successful, False otherwise
    """
    if not CAMERA_CONFIG_AVAILABLE:
        print("Warning: Camera config not available, cannot move camera")
        return False

    cam_info = get_camera_by_name(camera_name)
    if not cam_info:
        print(f"Warning: Camera '{camera_name}' not found in config")
        return False

    try:
        # Import camera control module
        from poc_homography.camera_config import USERNAME, PASSWORD
        from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ

        print(f"Moving camera '{camera_name}' to PTZ position:")
        print(f"  Pan:  {ptz_position.get('pan', 0):.1f}°")
        print(f"  Tilt: {ptz_position.get('tilt', 0):.1f}°")
        print(f"  Zoom: {ptz_position.get('zoom', 1.0):.1f}x")

        ptz = HikvisionPTZ(cam_info['ip'], USERNAME, PASSWORD)

        # Move to absolute position using send_ptz_return
        ptz.send_ptz_return({
            'pan': ptz_position.get('pan', 0),
            'tilt': ptz_position.get('tilt', 0),
            'zoom': ptz_position.get('zoom', 1.0)
        })

        # Wait for camera to reach position
        import time
        print(f"Waiting {wait_time}s for camera to reach position...")
        time.sleep(wait_time)

        print("Camera positioned successfully")
        return True

    except ImportError as e:
        print(f"Warning: PTZ control not available ({e})")
        return False
    except Exception as e:
        print(f"Warning: Failed to move camera: {e}")
        return False


def grab_frame_from_camera(
    camera_name: str,
    ptz_position: dict = None,
    wait_time: float = 3.0
) -> np.ndarray:
    """
    Grab a single frame from a camera, optionally moving to a PTZ position first.

    Args:
        camera_name: Name of the camera from camera_config
        ptz_position: Optional dict with 'pan', 'tilt', 'zoom' to move camera first
        wait_time: Seconds to wait after moving camera

    Returns:
        Frame as numpy array (BGR)

    Raises:
        RuntimeError: If cannot connect to camera or grab frame
    """
    if not CAMERA_CONFIG_AVAILABLE:
        raise RuntimeError(
            "Camera config not available. Set CAMERA_USERNAME and CAMERA_PASSWORD "
            "environment variables to enable camera access."
        )

    # Move camera to position if specified
    if ptz_position:
        move_camera_to_position(camera_name, ptz_position, wait_time)

    rtsp_url = get_rtsp_url(camera_name)
    if not rtsp_url:
        raise RuntimeError(f"Camera '{camera_name}' not found in camera_config")

    print(f"Connecting to camera '{camera_name}'...")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to connect to camera: {rtsp_url}")

    # Grab a few frames to ensure we get a good one
    frame = None
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            break

    cap.release()

    if frame is None:
        raise RuntimeError("Failed to grab frame from camera")

    print(f"Grabbed frame: {frame.shape[1]}x{frame.shape[0]}")
    return frame


def visualize_gcps_on_frame(
    frame: np.ndarray,
    gcps: list,
    results: dict = None,
    window_name: str = "GCP Validation"
) -> None:
    """
    Display GCP points overlaid on a frame for visual verification.

    Args:
        frame: Image frame (BGR)
        gcps: List of GCP dictionaries
        results: Optional validation results with inlier/outlier info
        window_name: OpenCV window name
    """
    display = frame.copy()

    # Get inlier mask if available
    inlier_mask = None
    if results and 'details' in results:
        # Build mask from details
        inlier_mask = []
        for detail in results['details']:
            # Check if this point passed (inliers should have ~0 error)
            passed = detail.get('passed', False)
            error = detail.get('error_meters', float('inf'))
            inlier_mask.append(error < 1.0)  # Very low error = inlier

    # Draw GCP points
    for i, gcp in enumerate(gcps):
        u = int(gcp['image']['u'])
        v = int(gcp['image']['v'])
        lat = gcp['gps']['latitude']
        lon = gcp['gps']['longitude']
        desc = gcp.get('metadata', {}).get('description', f'GCP {i+1}')

        # Determine color based on inlier status
        if inlier_mask and i < len(inlier_mask):
            color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)  # Green=inlier, Red=outlier
        else:
            color = (255, 255, 0)  # Cyan if no results yet

        # Draw crosshair
        cv2.drawMarker(display, (u, v), color, cv2.MARKER_CROSS, 20, 2)

        # Draw circle
        cv2.circle(display, (u, v), 10, color, 2)

        # Draw label with GPS coordinates
        label = f"{desc}: ({lat:.6f}, {lon:.6f})"

        # Calculate label position (offset to avoid overlap)
        label_x = u + 15
        label_y = v - 10

        # Ensure label is within frame bounds
        if label_x + 300 > frame.shape[1]:
            label_x = u - 320
        if label_y < 20:
            label_y = v + 25

        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (label_x - 2, label_y - text_h - 2),
                      (label_x + text_w + 2, label_y + 2), (0, 0, 0), -1)
        cv2.putText(display, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw legend
    legend_y = 30
    cv2.putText(display, "GCP Validation - Press any key to continue, 'q' to quit",
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results:
        legend_y += 30
        status_text = "PASSED" if results.get('success', False) else "FAILED"
        status_color = (0, 255, 0) if results.get('success', False) else (0, 0, 255)
        cv2.putText(display, f"Status: {status_text}", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        legend_y += 25
        cv2.putText(display, f"Inliers: {results.get('inliers', 'N/A')}/{results.get('gcps_tested', 'N/A')}",
                    (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        legend_y += 25
        cv2.putText(display, f"Mean error: {results.get('inlier_mean_error_m', 0):.4f}m",
                    (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Color legend
    legend_y = frame.shape[0] - 60
    cv2.circle(display, (20, legend_y), 8, (0, 255, 0), -1)
    cv2.putText(display, "Inlier", (35, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.circle(display, (100, legend_y), 8, (0, 0, 255), -1)
    cv2.putText(display, "Outlier", (115, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.circle(display, (190, legend_y), 8, (255, 255, 0), -1)
    cv2.putText(display, "Pending", (205, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show frame
    cv2.imshow(window_name, display)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        cv2.destroyAllWindows()
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='GCP Round-Trip Validation Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML config file with GCPs and camera capture context'
    )
    parser.add_argument(
        '--camera',
        type=str,
        help='Camera name (overrides config camera_capture_context)'
    )
    parser.add_argument(
        '--frame', '-f',
        type=str,
        help='Path to a saved frame image (skips camera connection)'
    )
    parser.add_argument(
        '--width', '-W',
        type=int,
        default=None,
        help='Image width in pixels (default: from config or 2560)'
    )
    parser.add_argument(
        '--height', '-H',
        type=int,
        default=None,
        help='Image height in pixels (default: from config or 1440)'
    )
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=5.0,
        help='Maximum acceptable error in meters (default: 5.0)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    parser.add_argument(
        '--no-visual',
        action='store_true',
        help='Skip visual verification (run test only)'
    )
    parser.add_argument(
        '--skip-ptz',
        action='store_true',
        help='Skip moving camera to PTZ position (use current position)'
    )
    parser.add_argument(
        '--wait-time',
        type=float,
        default=3.0,
        help='Seconds to wait after moving camera (default: 3.0)'
    )

    args = parser.parse_args()

    # Initialize variables
    gcps = None
    camera_context = None
    image_width = args.width or 2560
    image_height = args.height or 1440

    # Load GCPs and camera context from config
    if args.config:
        try:
            gcp_config = load_gcp_config(args.config)
            gcps = gcp_config['gcps']
            camera_context = gcp_config['camera_capture_context']

            # Use dimensions from config unless overridden
            if args.width is None:
                image_width = gcp_config['image_width']
            if args.height is None:
                image_height = gcp_config['image_height']

            if not args.quiet:
                print(f"Loaded {len(gcps)} GCPs from {args.config}")

                if camera_context:
                    print(f"\nCamera capture context:")
                    print(f"  Camera: {camera_context.get('camera_name', 'N/A')}")
                    ptz = camera_context.get('ptz_position', {})
                    print(f"  PTZ: pan={ptz.get('pan', 0):.1f}°, "
                          f"tilt={ptz.get('tilt', 0):.1f}°, "
                          f"zoom={ptz.get('zoom', 1.0):.1f}x")
                    print(f"  Image size: {image_width}x{image_height}")
                else:
                    print("  Note: No camera_capture_context in config")
                    print("  Visual validation requires --camera or --frame")

        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default test GCPs instead.")
            gcps = DEFAULT_TEST_GCPS

    else:
        gcps = DEFAULT_TEST_GCPS
        if not args.quiet:
            print("Using default test GCPs (synthetic data)")
            print("  Hint: Use --config to load real GCPs with camera context")

    # Determine camera name (CLI overrides config)
    camera_name = args.camera
    if camera_name is None and camera_context:
        camera_name = camera_context.get('camera_name')

    # Determine PTZ position (from config, unless skipped)
    ptz_position = None
    if not args.skip_ptz and camera_context:
        ptz_position = camera_context.get('ptz_position')

    # Get frame for visualization
    frame = None
    if not args.no_visual:
        if args.frame:
            # Load saved frame
            frame = cv2.imread(args.frame)
            if frame is None:
                print(f"Warning: Could not load frame from {args.frame}")
                print("Continuing without visual verification...")
            else:
                print(f"Loaded frame from {args.frame}: {frame.shape[1]}x{frame.shape[0]}")
        elif camera_name:
            # Connect to camera
            try:
                frame = grab_frame_from_camera(
                    camera_name,
                    ptz_position=ptz_position,
                    wait_time=args.wait_time
                )
            except RuntimeError as e:
                print(f"Warning: {e}")
                print("Continuing without visual verification...")
        else:
            if not args.quiet:
                print("No camera or frame specified. Running math-only validation.")
                print("  Use --camera, --frame, or add camera_capture_context to config")

    # Check for resolution mismatch between config and grabbed frame
    # The math validation uses config dimensions (where GCPs were captured)
    # Visualization may need to scale GCPs if frame resolution differs
    scale_x, scale_y = 1.0, 1.0
    if frame is not None:
        frame_height, frame_width = frame.shape[:2]
        if not args.quiet:
            print(f"Grabbed frame dimensions: {frame_width}x{frame_height}")

        # Check for resolution mismatch
        if frame_width != image_width or frame_height != image_height:
            scale_x = frame_width / image_width
            scale_y = frame_height / image_height
            if not args.quiet:
                print(f"Resolution mismatch detected!")
                print(f"  Config dimensions: {image_width}x{image_height}")
                print(f"  Frame dimensions:  {frame_width}x{frame_height}")
                print(f"  Scaling GCPs for visualization by ({scale_x:.3f}, {scale_y:.3f})")
        else:
            if not args.quiet:
                print(f"Using frame dimensions: {frame_width}x{frame_height}")

    # Create scaled GCPs for visualization if there's a resolution mismatch
    display_gcps = gcps
    if scale_x != 1.0 or scale_y != 1.0:
        display_gcps = []
        for gcp in gcps:
            scaled_gcp = {
                'gps': gcp['gps'].copy(),
                'image': {
                    'u': gcp['image']['u'] * scale_x,
                    'v': gcp['image']['v'] * scale_y,
                },
            }
            if 'metadata' in gcp:
                scaled_gcp['metadata'] = gcp['metadata'].copy()
            display_gcps.append(scaled_gcp)

    # Show GCPs before validation (if we have a frame)
    if frame is not None:
        print("\nShowing GCP positions on frame (before validation)...")
        print("Verify that GCP markers align with the physical features.")
        print("Press any key to run validation, 'q' to quit")
        visualize_gcps_on_frame(frame, display_gcps, results=None)

    # Run validation
    results = run_round_trip_validation(
        gcps=gcps,
        width=image_width,
        height=image_height,
        tolerance_meters=args.tolerance,
        verbose=not args.quiet
    )

    # Show results visualization (if we have a frame)
    if frame is not None:
        print("\nShowing validation results on frame...")
        print("Green = inlier (passed), Red = outlier (failed)")
        print("Press any key to exit")
        visualize_gcps_on_frame(frame, display_gcps, results=results)
        cv2.destroyAllWindows()

    # Exit code based on success
    sys.exit(0 if results.get('success', False) else 1)


if __name__ == '__main__':
    main()
