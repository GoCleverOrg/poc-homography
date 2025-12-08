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
from poc_homography.kml_generator import generate_kml, create_output_directory
from poc_homography.map_debug_server import start_server, generate_html

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
    inliers_only: bool = True,
    confidence_threshold: float = 0.5,
    camera_gps: dict = None
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
        confidence_threshold: Minimum confidence for homography to be valid (default 0.5)
        camera_gps: Optional dict with 'latitude' and 'longitude' for camera position.
                    Used as reference point for local coordinate system to minimize
                    ellipsoid projection errors.

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
        print(f"Confidence threshold: {confidence_threshold}")
        if camera_gps:
            print(f"Reference point (camera GPS): ({camera_gps['latitude']:.6f}, {camera_gps['longitude']:.6f})")
        else:
            print("Reference point: GCP centroid (no camera_gps provided)")
        print()

    # Create provider
    provider = FeatureMatchHomography(
        width=width,
        height=height,
        min_matches=4,
        ransac_threshold=5.0,
        confidence_threshold=confidence_threshold
    )

    # Compute homography
    if verbose:
        print("Computing homography from GCPs...")

    # Build reference dict with GCPs and optional camera GPS
    reference = {'ground_control_points': gcps}
    if camera_gps:
        reference['camera_gps'] = camera_gps

    result = provider.compute_homography(
        frame=None,
        reference=reference
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

    # Get the homography matrix to compute pixel reprojection errors
    H = provider.H  # local metric -> image

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

        # Calculate pixel reprojection error (GPS -> local -> image vs original image)
        # Convert GPS to local metric coordinates
        x_local, y_local = provider._gps_to_local(original_lat, original_lon)
        # Project local to image using H
        pt_h = np.array([x_local, y_local, 1.0])
        proj_h = H @ pt_h
        proj_u = proj_h[0] / proj_h[2]
        proj_v = proj_h[1] / proj_h[2]
        pixel_error = math.sqrt((proj_u - u)**2 + (proj_v - v)**2)

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
                'pixel_error': pixel_error,
                'passed': passed,
                'is_inlier': is_inlier  # RANSAC inlier status
            }
            results_detail.append(result_detail)

            if verbose:
                desc = gcp.get('metadata', {}).get('description', f'GCP {i+1}')
                inlier_str = " (INLIER)" if is_inlier else " (OUTLIER)" if is_inlier is not None else ""
                print(f"  [{status}] {desc}{inlier_str}")
                print(f"       Image point: ({u:.1f}, {v:.1f})")
                print(f"       Pixel reproj error: {pixel_error:.2f} px")
                print(f"       Original GPS:  ({original_lat:.6f}, {original_lon:.6f})")
                print(f"       Estimated GPS: ({projected_lat:.6f}, {projected_lon:.6f})")
                print(f"       GPS error: {error_m:.2f}m")
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
            'details': results_detail,
            'provider': provider  # Return the provider for interactive testing
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
        - 'camera_gps': Camera GPS position dict (or None)
        - 'image_width': Image width in pixels
        - 'image_height': Image height in pixels
    """
    config = HomographyConfig.from_yaml(config_path)

    result = {
        'gcps': None,
        'camera_capture_context': None,
        'camera_gps': None,
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
            # Extract camera GPS if available
            if 'camera_gps' in ctx:
                result['camera_gps'] = ctx['camera_gps']

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

    # Get RANSAC inlier mask from results
    inlier_mask = None
    if results and 'details' in results:
        # Use the actual RANSAC inlier status, not GPS error
        inlier_mask = []
        for detail in results['details']:
            is_inlier = detail.get('is_inlier', None)
            inlier_mask.append(is_inlier)

    # Draw GCP points
    for i, gcp in enumerate(gcps):
        u = int(gcp['image']['u'])
        v = int(gcp['image']['v'])
        lat = gcp['gps']['latitude']
        lon = gcp['gps']['longitude']
        full_desc = gcp.get('metadata', {}).get('description', f'GCP {i+1}')

        # Extract just the P#XX part for the image label if present
        if full_desc.startswith('P#'):
            short_desc = full_desc.split(' - ')[0]
        else:
            short_desc = full_desc

        # Determine color based on inlier status
        if inlier_mask and i < len(inlier_mask):
            color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)  # Green=inlier, Red=outlier
        else:
            color = (255, 255, 0)  # Cyan if no results yet

        # Draw crosshair
        cv2.drawMarker(display, (u, v), color, cv2.MARKER_CROSS, 20, 2)

        # Draw circle
        cv2.circle(display, (u, v), 10, color, 2)

        # Draw label with short description only
        label = short_desc

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


def export_frame_with_markers(
    frame: np.ndarray,
    gcps: list,
    validation_results: dict,
    output_path: str
) -> str:
    """
    Export camera frame with GCP markers and validation results.

    Args:
        frame: Image frame (BGR)
        gcps: List of GCP dictionaries
        validation_results: Validation results with projected positions
        output_path: Output path for JPEG file

    Returns:
        Absolute path to exported frame
    """
    display = frame.copy()

    # Draw GCP markers and validation results
    for i, gcp in enumerate(gcps):
        # Original GCP position (green)
        u_orig = int(gcp['image']['u'])
        v_orig = int(gcp['image']['v'])
        cv2.circle(display, (u_orig, v_orig), 8, (0, 255, 0), 2)

        # Get projected position from validation results
        if validation_results and 'details' in validation_results:
            details = validation_results['details']
            if i < len(details):
                detail = details[i]
                if 'image_point' in detail:
                    u_proj, v_proj = detail['image_point']
                    u_proj = int(u_proj)
                    v_proj = int(v_proj)

                    # Projected position (blue)
                    cv2.circle(display, (u_proj, v_proj), 8, (255, 0, 0), 2)

                    # Connection line (yellow)
                    cv2.line(display, (u_orig, v_orig), (u_proj, v_proj), (0, 255, 255), 2)

        # Label with description
        desc = gcp.get('metadata', {}).get('description', f'GCP {i+1}')
        if desc.startswith('P#'):
            desc = desc.split(' - ')[0]

        label_x = u_orig + 15
        label_y = v_orig - 10
        if label_x + 100 > frame.shape[1]:
            label_x = u_orig - 120
        if label_y < 20:
            label_y = v_orig + 25

        (text_w, text_h), _ = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (label_x - 2, label_y - text_h - 2),
                      (label_x + text_w + 2, label_y + 2), (0, 0, 0), -1)
        cv2.putText(display, desc, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save frame
    cv2.imwrite(output_path, display)
    return str(Path(output_path).absolute())


def run_map_debug_visualization(
    frame: np.ndarray,
    gcps: list,
    validation_results: dict,
    camera_gps: dict,
    output_dir: str = None
) -> None:
    """
    Launch web-based map visualization with GCP overlays and round-trip error analysis.

    Args:
        frame: Image frame (BGR)
        gcps: List of GCP dictionaries
        validation_results: Validation results from run_round_trip_validation
        camera_gps: Camera GPS position dict with 'latitude' and 'longitude'
        output_dir: Optional output directory (creates one if not specified)
    """
    import datetime

    # Create output directory if not specified
    if output_dir is None:
        output_dir = create_output_directory()
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export camera frame with GCP markers
    frame_filename = f"annotated_frame_{timestamp}.jpg"
    frame_path = os.path.join(output_dir, frame_filename)
    export_frame_with_markers(frame, gcps, validation_results, frame_path)
    print(f"Exported annotated frame: {frame_path}")

    # Generate KML file
    kml_filename = f"gcp_validation_{timestamp}.kml"
    kml_path = os.path.join(output_dir, kml_filename)
    generate_kml(gcps, validation_results, camera_gps, kml_path)
    print(f"Generated KML file: {kml_path}")

    # Start web server
    print("\nLaunching map debug visualization...")
    start_server(
        output_dir=output_dir,
        camera_frame_path=frame_path,
        kml_path=kml_path,
        camera_gps=camera_gps,
        gcps=gcps,
        validation_results=validation_results,
        auto_open=True
    )


def interactive_gps_projection(
    frame: np.ndarray,
    provider: 'FeatureMatchHomography',
    gcps: list = None,
    results: dict = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    window_name: str = "Interactive GPS Projection"
) -> None:
    """
    Interactive mode: click on the frame to get GPS coordinates.

    Args:
        frame: Image frame (BGR)
        provider: The homography provider with computed homography
        gcps: Optional list of GCPs to display
        results: Optional validation results
        scale_x: Scale factor for x coordinates (display -> original)
        scale_y: Scale factor for y coordinates (display -> original)
        window_name: OpenCV window name
    """
    # State for mouse callback
    state = {
        'click_point': None,
        'projected_points': [],  # List of (pixel, gps) tuples
    }

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinates to original image coordinates
            orig_x = x / scale_x
            orig_y = y / scale_y
            state['click_point'] = (x, y, orig_x, orig_y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "=" * 60)
    print("INTERACTIVE GPS PROJECTION MODE")
    print("=" * 60)
    print("Click anywhere on the image to get GPS coordinates")
    print("Press 'c' to clear markers, 'q' to quit")
    print("=" * 60)

    while True:
        # Create fresh display
        display = frame.copy()

        # Draw GCP points if provided
        if gcps:
            inlier_mask = None
            if results and 'details' in results:
                inlier_mask = [d.get('is_inlier') for d in results['details']]

            for i, gcp in enumerate(gcps):
                u = int(gcp['image']['u'])
                v = int(gcp['image']['v'])

                if inlier_mask and i < len(inlier_mask):
                    color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
                else:
                    color = (255, 255, 0)

                cv2.drawMarker(display, (u, v), color, cv2.MARKER_CROSS, 15, 1)
                cv2.circle(display, (u, v), 8, color, 1)

        # Draw previously projected points
        for i, (pixel, gps_coord) in enumerate(state['projected_points']):
            px, py = int(pixel[0]), int(pixel[1])
            # Draw marker
            cv2.drawMarker(display, (px, py), (255, 0, 255), cv2.MARKER_DIAMOND, 20, 2)
            cv2.circle(display, (px, py), 12, (255, 0, 255), 2)

            # Draw label
            label = f"#{i+1}: ({gps_coord[0]:.6f}, {gps_coord[1]:.6f})"
            label_x = px + 15
            label_y = py - 10
            if label_x + 350 > frame.shape[1]:
                label_x = px - 360
            if label_y < 20:
                label_y = py + 25

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (label_x - 2, label_y - text_h - 2),
                          (label_x + text_w + 2, label_y + 2), (0, 0, 0), -1)
            cv2.putText(display, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Draw instructions
        cv2.putText(display, "Click to project GPS | 'c' clear | 'q' quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw point count
        if state['projected_points']:
            cv2.putText(display, f"Points: {len(state['projected_points'])}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Handle new click
        if state['click_point'] is not None:
            disp_x, disp_y, orig_x, orig_y = state['click_point']
            state['click_point'] = None

            try:
                # Project the point using original coordinates
                world_point = provider.project_point((orig_x, orig_y))
                lat, lon = world_point.latitude, world_point.longitude

                # Store for display (using display coordinates)
                state['projected_points'].append(((disp_x, disp_y), (lat, lon)))

                # Print to console
                print(f"\n  Point #{len(state['projected_points'])}:")
                print(f"    Pixel (display): ({disp_x:.1f}, {disp_y:.1f})")
                if scale_x != 1.0 or scale_y != 1.0:
                    print(f"    Pixel (original): ({orig_x:.1f}, {orig_y:.1f})")
                print(f"    GPS: ({lat:.6f}, {lon:.6f})")

            except Exception as e:
                print(f"\n  Error projecting point ({orig_x:.1f}, {orig_y:.1f}): {e}")

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            state['projected_points'].clear()
            print("\n  Cleared all projected points")

    cv2.destroyAllWindows()


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
        '--confidence', '-C',
        type=float,
        default=0.5,
        help='Minimum confidence threshold for valid homography (default: 0.5)'
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
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Enable interactive mode: click on frame to get GPS coordinates'
    )
    parser.add_argument(
        '--map-debug',
        action='store_true',
        help='Launch web-based map visualization with GCP overlays and round-trip error analysis'
    )

    args = parser.parse_args()

    # Initialize variables
    gcps = None
    camera_context = None
    camera_gps = None
    image_width = args.width or 2560
    image_height = args.height or 1440

    # Load GCPs and camera context from config
    if args.config:
        try:
            gcp_config = load_gcp_config(args.config)
            gcps = gcp_config['gcps']
            camera_context = gcp_config['camera_capture_context']
            camera_gps = gcp_config['camera_gps']

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
                    if camera_gps:
                        print(f"  Camera GPS: ({camera_gps['latitude']:.6f}, {camera_gps['longitude']:.6f})")
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

    # Show GCPs before validation (if we have a frame and not in map-debug mode)
    if frame is not None and not args.map_debug:
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
        verbose=not args.quiet,
        confidence_threshold=args.confidence,
        camera_gps=camera_gps
    )

    # Show results visualization based on mode
    if frame is not None:
        if args.map_debug:
            # Map debug mode: Launch web-based visualization
            if camera_gps is None:
                print("Warning: --map-debug requires camera GPS coordinates")
                print("  Add camera_gps to your config or use standard visualization")
                args.map_debug = False
            else:
                run_map_debug_visualization(
                    frame=frame,
                    gcps=gcps,
                    validation_results=results,
                    camera_gps=camera_gps
                )

        if not args.map_debug:
            # Standard OpenCV visualization
            print("\nShowing validation results on frame...")
            print("Green = inlier (passed), Red = outlier (failed)")
            print("Press any key to continue" + (" to interactive mode" if args.interactive else ""))
            visualize_gcps_on_frame(frame, display_gcps, results=results)

            # Interactive mode: click to get GPS coordinates
            if args.interactive and results.get('provider'):
                interactive_gps_projection(
                    frame=frame,
                    provider=results['provider'],
                    gcps=display_gcps,
                    results=results,
                    scale_x=scale_x,
                    scale_y=scale_y
                )

            cv2.destroyAllWindows()

    # Exit code based on success
    sys.exit(0 if results.get('success', False) else 1)


if __name__ == '__main__':
    main()
