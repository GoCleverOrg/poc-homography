#!/usr/bin/env python3
"""
Multi-frame PTZ calibration capture tool.

Command-line interface for managing multi-frame calibration sessions:
- Initialize new capture sessions
- Capture frames at different PTZ positions
- Add Ground Control Points (GCPs) with GPS coordinates
- Mark GCP observations in frames (pixel coordinates)
- Validate session status
- Run multi-frame calibration
- List frames and GCPs

This tool orchestrates the complete workflow from frame capture through
calibration, providing a simple CLI interface for field operations.

Usage Examples:
    # Initialize a new session
    python tools/multi_frame_capture.py init --camera Valte --output data/session_001

    # Capture frames (move camera between captures)
    python tools/multi_frame_capture.py capture data/session_001
    python tools/multi_frame_capture.py capture data/session_001
    python tools/multi_frame_capture.py capture data/session_001

    # Add GCPs
    python tools/multi_frame_capture.py add-gcp data/session_001 --id gcp_001 --lat 39.640583 --lon -0.230194

    # Mark GCP observations in frames
    python tools/multi_frame_capture.py mark data/session_001 --gcp gcp_001 --frame frame_001 --u 1250.5 --v 680.0

    # Check status and validation
    python tools/multi_frame_capture.py status data/session_001

    # Run calibration
    python tools/multi_frame_capture.py calibrate data/session_001 --output calibration_result.yaml

    # List frames and GCPs
    python tools/multi_frame_capture.py list-frames data/session_001
    python tools/multi_frame_capture.py list-gcps data/session_001
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Color output for terminal (simple, no external dependencies)
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'

    @staticmethod
    def enabled():
        """Check if color output is supported."""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

def cprint(text: str, color: str = '', bold: bool = False):
    """Print colored text if terminal supports it."""
    if Colors.enabled():
        prefix = Colors.BOLD if bold else ''
        suffix = Colors.RESET
        print(f"{prefix}{color}{text}{suffix}")
    else:
        print(text)

def print_header(text: str):
    """Print section header."""
    cprint(f"\n{'='*70}", Colors.CYAN)
    cprint(text, Colors.CYAN, bold=True)
    cprint('='*70, Colors.CYAN)

def print_success(text: str):
    """Print success message."""
    cprint(f"✓ {text}", Colors.GREEN, bold=True)

def print_error(text: str):
    """Print error message."""
    cprint(f"✗ {text}", Colors.RED, bold=True)

def print_warning(text: str):
    """Print warning message."""
    cprint(f"⚠ {text}", Colors.YELLOW)

def print_info(text: str):
    """Print info message."""
    cprint(f"ℹ {text}", Colors.BLUE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Command Handlers
# ============================================================================

def cmd_init(args):
    """Initialize a new multi-frame capture session."""
    from poc_homography.multi_frame_session import MultiFrameCaptureSession
    from poc_homography.camera_config import get_camera_by_name_safe

    print_header("Initializing Multi-Frame Capture Session")

    # Validate camera exists
    camera_config = get_camera_by_name_safe(args.camera)
    if camera_config is None:
        print_error(f"Camera '{args.camera}' not found in camera_config.py")
        return 1

    print_info(f"Camera: {args.camera}")
    print_info(f"Output directory: {args.output}")

    # Check if output directory already exists
    output_path = Path(args.output)
    session_file = output_path / "session.yaml"
    if session_file.exists():
        print_warning(f"Session already exists at {session_file}")
        response = input("Overwrite existing session? (y/N): ").strip().lower()
        if response != 'y':
            print_info("Initialization cancelled")
            return 0

    try:
        # Create session
        session = MultiFrameCaptureSession(
            camera_name=args.camera,
            output_dir=args.output
        )

        # Save session
        session.save_session()

        print_success("Session initialized successfully")
        print_info(f"Session file: {session_file}")
        print_info("Next steps:")
        print("  1. Capture frames: python tools/multi_frame_capture.py capture " + args.output)
        print("  2. Add GCPs: python tools/multi_frame_capture.py add-gcp " + args.output + " --id gcp_001 --lat LAT --lon LON")
        print("  3. Mark observations: python tools/multi_frame_capture.py mark " + args.output + " --gcp gcp_001 --frame FRAME_ID --u U --v V")

        return 0

    except Exception as e:
        print_error(f"Failed to initialize session: {e}")
        logger.exception("Initialization failed")
        return 1


def cmd_capture(args):
    """Capture a frame from the camera."""
    from poc_homography.multi_frame_session import MultiFrameCaptureSession

    print_header("Capturing Frame")

    session_path = Path(args.session_dir)
    session_file = session_path / "session.yaml"

    if not session_file.exists():
        print_error(f"Session not found: {session_file}")
        print_info("Initialize a session first with: python tools/multi_frame_capture.py init")
        return 1

    try:
        # Load session
        print_info("Loading session...")
        session = MultiFrameCaptureSession.load_session(str(session_file))

        print_info(f"Camera: {session.camera_name}")
        print_info(f"Existing frames: {len(session.frames)}")

        # Capture frame
        print_info(f"Capturing frame (waiting {args.wait}s for camera stabilization)...")
        frame = session.capture_frame(wait_time=args.wait)

        print_success(f"Frame captured: {frame.frame_id}")
        print_info(f"PTZ position: pan={frame.ptz_position.pan:.2f}°, tilt={frame.ptz_position.tilt:.2f}°, zoom={frame.ptz_position.zoom:.1f}")
        print_info(f"Image saved: {frame.image_path}")

        # Save session
        session.save_session()
        print_success("Session saved")

        print_info("Next steps:")
        print("  - Capture more frames (move camera to different PTZ position first)")
        print("  - Add GCPs and mark observations")

        return 0

    except ImportError as e:
        print_error("Failed to import camera capture module")
        print_error(str(e))
        print_info("Ensure tools.capture_gcps_web is available")
        return 1
    except Exception as e:
        print_error(f"Failed to capture frame: {e}")
        logger.exception("Frame capture failed")
        return 1


def cmd_add_gcp(args):
    """Add a Ground Control Point to the session."""
    from poc_homography.multi_frame_session import MultiFrameCaptureSession

    print_header("Adding Ground Control Point")

    session_path = Path(args.session_dir)
    session_file = session_path / "session.yaml"

    if not session_file.exists():
        print_error(f"Session not found: {session_file}")
        return 1

    try:
        # Load session
        session = MultiFrameCaptureSession.load_session(str(session_file))

        print_info(f"GCP ID: {args.id}")
        print_info(f"GPS coordinates: ({args.lat:.6f}, {args.lon:.6f})")

        if args.utm_easting and args.utm_northing:
            print_info(f"UTM coordinates: ({args.utm_easting:.2f}, {args.utm_northing:.2f})")

        # Add GCP
        gcp = session.add_gcp(
            gcp_id=args.id,
            gps_lat=args.lat,
            gps_lon=args.lon,
            utm_easting=args.utm_easting,
            utm_northing=args.utm_northing
        )

        print_success(f"GCP '{args.id}' added successfully")

        # Save session
        session.save_session()
        print_success("Session saved")

        print_info("Next steps:")
        print(f"  - Mark this GCP in frames: python tools/multi_frame_capture.py mark {args.session_dir} --gcp {args.id} --frame FRAME_ID --u U --v V")

        return 0

    except ValueError as e:
        print_error(f"Invalid GCP data: {e}")
        return 1
    except Exception as e:
        print_error(f"Failed to add GCP: {e}")
        logger.exception("Add GCP failed")
        return 1


def cmd_mark(args):
    """Mark a GCP observation in a frame."""
    from poc_homography.multi_frame_session import MultiFrameCaptureSession

    print_header("Marking GCP Observation")

    session_path = Path(args.session_dir)
    session_file = session_path / "session.yaml"

    if not session_file.exists():
        print_error(f"Session not found: {session_file}")
        return 1

    try:
        # Load session
        session = MultiFrameCaptureSession.load_session(str(session_file))

        print_info(f"GCP: {args.gcp}")
        print_info(f"Frame: {args.frame}")
        print_info(f"Pixel coordinates: ({args.u:.1f}, {args.v:.1f})")

        # Add observation
        session.add_gcp_observation(
            gcp_id=args.gcp,
            frame_id=args.frame,
            u=args.u,
            v=args.v
        )

        print_success(f"Observation marked: GCP '{args.gcp}' in frame '{args.frame}'")

        # Save session
        session.save_session()
        print_success("Session saved")

        # Show GCP status
        gcp = session.gcps[args.gcp]
        num_observations = len(gcp.frame_observations)
        print_info(f"GCP '{args.gcp}' now has {num_observations} observation(s)")

        return 0

    except ValueError as e:
        print_error(f"Invalid observation: {e}")
        return 1
    except Exception as e:
        print_error(f"Failed to mark observation: {e}")
        logger.exception("Mark observation failed")
        return 1


def cmd_status(args):
    """Show session status and validation."""
    from poc_homography.multi_frame_session import MultiFrameCaptureSession

    print_header("Session Status")

    session_path = Path(args.session_dir)
    session_file = session_path / "session.yaml"

    if not session_file.exists():
        print_error(f"Session not found: {session_file}")
        return 1

    try:
        # Load session
        session = MultiFrameCaptureSession.load_session(str(session_file))

        # Basic info
        print_info(f"Camera: {session.camera_name}")
        print_info(f"Output directory: {session.output_dir}")
        print_info(f"Session file: {session_file}")
        print()

        # Statistics
        stats = session.get_session_stats()

        cprint("Statistics:", Colors.CYAN, bold=True)
        print(f"  Frames: {stats['num_frames']}")
        print(f"  GCPs: {stats['num_gcps']}")
        print(f"  Total observations: {stats['total_observations']}")
        print(f"  Avg observations per GCP: {stats['avg_observations_per_gcp']:.1f}")
        print(f"  Avg GCPs per frame: {stats['avg_gcps_per_frame']:.1f}")
        print()

        # PTZ range
        if stats['ptz_range']:
            ptz_range = stats['ptz_range']
            cprint("PTZ Range:", Colors.CYAN, bold=True)
            print(f"  Pan: {ptz_range['pan'][0]:.1f}° to {ptz_range['pan'][1]:.1f}°")
            print(f"  Tilt: {ptz_range['tilt'][0]:.1f}° to {ptz_range['tilt'][1]:.1f}°")
            print(f"  Zoom: {ptz_range['zoom'][0]:.1f} to {ptz_range['zoom'][1]:.1f}")
            print()

        # Validation
        validation = session.validate_session()

        cprint("Validation:", Colors.CYAN, bold=True)
        if validation['is_valid']:
            print_success("Session is valid and ready for calibration")
        else:
            print_error("Session is NOT valid for calibration")
        print()

        # Errors
        if validation['errors']:
            cprint("Errors (must be fixed):", Colors.RED, bold=True)
            for error in validation['errors']:
                print(f"  ✗ {error}")
            print()

        # Warnings
        if validation['warnings']:
            cprint("Warnings (recommended to address):", Colors.YELLOW, bold=True)
            for warning in validation['warnings']:
                print(f"  ⚠ {warning}")
            print()

        # Show frame details if verbose
        if args.verbose and stats['frame_details']:
            cprint("Frame Details:", Colors.CYAN, bold=True)
            for frame in stats['frame_details']:
                print(f"  {frame['frame_id']}:")
                print(f"    PTZ: pan={frame['pan']:.1f}°, tilt={frame['tilt']:.1f}°, zoom={frame['zoom']:.1f}")
                print(f"    GCP observations: {frame['num_gcps']}")
                print(f"    Timestamp: {frame['timestamp']}")
            print()

        # Show GCP details if verbose
        if args.verbose and stats['gcp_details']:
            cprint("GCP Details:", Colors.CYAN, bold=True)
            for gcp in stats['gcp_details']:
                print(f"  {gcp['gcp_id']}:")
                print(f"    GPS: ({gcp['gps_lat']:.6f}, {gcp['gps_lon']:.6f})")
                print(f"    Observations: {gcp['num_observations']} frame(s)")
                print(f"    Frames: {', '.join(gcp['frame_ids'])}")
            print()

        # Next steps
        if not validation['is_valid']:
            print_info("Next steps:")
            if stats['num_frames'] < 3:
                print("  - Capture more frames (minimum 3 required)")
            if stats['num_gcps'] < 6:
                print("  - Add more GCPs (minimum 6 required)")
            if validation['errors']:
                print("  - Address validation errors listed above")
        else:
            print_info("Next steps:")
            print(f"  - Run calibration: python tools/multi_frame_capture.py calibrate {args.session_dir} --output result.yaml")

        return 0

    except Exception as e:
        print_error(f"Failed to get status: {e}")
        logger.exception("Status check failed")
        return 1


def cmd_calibrate(args):
    """Run multi-frame calibration."""
    from poc_homography.multi_frame_session import MultiFrameCaptureSession
    from poc_homography.multi_frame_calibrator import MultiFrameCalibrator
    from poc_homography.multi_frame_io import save_multi_frame_calibration_result
    from poc_homography.camera_geometry import CameraGeometry
    from poc_homography.camera_config import get_camera_by_name_safe
    import numpy as np

    print_header("Multi-Frame Calibration")

    session_path = Path(args.session_dir)
    session_file = session_path / "session.yaml"

    if not session_file.exists():
        print_error(f"Session not found: {session_file}")
        return 1

    try:
        # Load session
        print_info("Loading session...")
        session = MultiFrameCaptureSession.load_session(str(session_file))

        # Validate session
        print_info("Validating session...")
        validation = session.validate_session()

        if not validation['is_valid']:
            print_error("Session validation failed:")
            for error in validation['errors']:
                print(f"  ✗ {error}")
            print_info("Fix validation errors before running calibration")
            return 1

        print_success("Session is valid")
        print()

        # Get camera configuration
        camera_config = get_camera_by_name_safe(session.camera_name)
        if camera_config is None:
            print_error(f"Camera '{session.camera_name}' not found in camera_config.py")
            return 1

        # Get intrinsics from first frame's zoom
        first_frame = session.frames[0]
        K = CameraGeometry.get_intrinsics(zoom_factor=first_frame.ptz_position.zoom)

        # Camera position (use from camera_config if available, else default)
        w_pos = [0.0, 0.0, camera_config.get('height_m', 5.0)]

        print_info("Camera configuration:")
        print(f"  Camera: {session.camera_name}")
        print(f"  Height: {w_pos[2]:.2f}m")
        print(f"  Intrinsic matrix K (from zoom={first_frame.ptz_position.zoom:.1f}):")
        print(f"    fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        print(f"    cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        print()

        # Create camera geometry
        geo = CameraGeometry(1920, 1080)

        # Export calibration data
        print_info("Preparing calibration data...")

        # Build camera_config dict
        cam_config = {
            'K': K,
            'w_pos': w_pos,
            'camera_name': session.camera_name
        }

        # Add reference coordinates if available
        if 'lat' in camera_config and 'lon' in camera_config:
            # Parse GPS coordinates from format like "39°38'25.72\"N"
            def parse_gps_coord(coord_str):
                """Parse GPS coordinate string to decimal degrees."""
                import re
                match = re.match(r'(\d+)°(\d+)\'([\d.]+)"([NSEW])', coord_str)
                if not match:
                    return None
                deg, min, sec, dir = match.groups()
                decimal = float(deg) + float(min)/60 + float(sec)/3600
                if dir in ['S', 'W']:
                    decimal = -decimal
                return decimal

            lat = parse_gps_coord(camera_config['lat'])
            lon = parse_gps_coord(camera_config['lon'])

            if lat and lon:
                cam_config['reference_lat'] = lat
                cam_config['reference_lon'] = lon
                print_info(f"Using camera GPS as reference: ({lat:.6f}, {lon:.6f})")

        calib_data = session.export_calibration_data(camera_config=cam_config)

        print_info(f"Frames: {len(calib_data.frames)}")
        print_info(f"GCPs: {len(calib_data.gcps)}")
        print_info(f"Total observations: {sum(len(gcp.frame_observations) for gcp in calib_data.gcps)}")
        print()

        # Create calibrator
        print_info("Initializing calibrator...")
        calibrator = MultiFrameCalibrator(
            camera_geometry=geo,
            calibration_data=calib_data,
            loss_function=args.loss,
            loss_scale=args.loss_scale,
            regularization_weight=args.regularization_weight
        )

        print_info(f"Loss function: {args.loss}")
        print_info(f"Loss scale: {args.loss_scale}px")
        print_info(f"Regularization weight: {args.regularization_weight}")
        print()

        # Run calibration
        print_info("Running calibration optimization...")
        print_info("(This may take a few seconds)")
        print()

        result = calibrator.calibrate()

        # Display results
        print_header("Calibration Results")

        if result.convergence_info.get('success', False):
            print_success("Calibration converged successfully")
        else:
            print_warning("Calibration did not fully converge")
            print_info(f"Reason: {result.convergence_info.get('message', 'Unknown')}")
        print()

        cprint("Optimized Parameters:", Colors.CYAN, bold=True)
        print(f"  Δpan:   {result.optimized_params[0]:+.3f}°")
        print(f"  Δtilt:  {result.optimized_params[1]:+.3f}°")
        print(f"  Δroll:  {result.optimized_params[2]:+.3f}°")
        print(f"  ΔX:     {result.optimized_params[3]:+.3f}m")
        print(f"  ΔY:     {result.optimized_params[4]:+.3f}m")
        print(f"  ΔZ:     {result.optimized_params[5]:+.3f}m")
        print()

        cprint("Error Metrics:", Colors.CYAN, bold=True)
        print(f"  Initial RMS error:  {result.initial_error:.2f}px")
        print(f"  Final RMS error:    {result.final_error:.2f}px")
        print(f"  Improvement:        {result.initial_error - result.final_error:.2f}px ({(1 - result.final_error/result.initial_error)*100:.1f}%)")
        print()

        cprint("Inlier/Outlier Analysis:", Colors.CYAN, bold=True)
        print(f"  Inliers:  {result.num_inliers}/{result.total_observations} ({result.inlier_ratio*100:.1f}%)")
        print(f"  Outliers: {result.num_outliers}/{result.total_observations} ({(1-result.inlier_ratio)*100:.1f}%)")
        print()

        cprint("Per-Frame Errors:", Colors.CYAN, bold=True)
        for frame_id, rms_error in sorted(result.per_frame_errors.items()):
            inliers = result.per_frame_inliers.get(frame_id, 0)
            outliers = result.per_frame_outliers.get(frame_id, 0)
            total = inliers + outliers
            print(f"  {frame_id}: {rms_error:.2f}px ({inliers}/{total} inliers)")
        print()

        cprint("Convergence Info:", Colors.CYAN, bold=True)
        print(f"  Iterations: {result.convergence_info.get('iterations', 'N/A')}")
        print(f"  Function evals: {result.convergence_info.get('function_evals', 'N/A')}")
        print()

        # Save result if output specified
        if args.output:
            output_path = Path(args.output)
            print_info(f"Saving results to {output_path}...")
            save_multi_frame_calibration_result(result, str(output_path))
            print_success(f"Results saved: {output_path}")

        print_info("Next steps:")
        print("  - Review per-frame errors to identify problematic frames")
        print("  - Check optimized parameters are within reasonable ranges")
        print("  - Apply parameter corrections to camera_config.py")

        return 0

    except Exception as e:
        print_error(f"Calibration failed: {e}")
        logger.exception("Calibration failed")
        return 1


def cmd_list_frames(args):
    """List all frames in the session."""
    from poc_homography.multi_frame_session import MultiFrameCaptureSession

    print_header("Frame List")

    session_path = Path(args.session_dir)
    session_file = session_path / "session.yaml"

    if not session_file.exists():
        print_error(f"Session not found: {session_file}")
        return 1

    try:
        # Load session
        session = MultiFrameCaptureSession.load_session(str(session_file))

        if not session.frames:
            print_warning("No frames captured yet")
            print_info("Capture frames with: python tools/multi_frame_capture.py capture " + args.session_dir)
            return 0

        print_info(f"Total frames: {len(session.frames)}")
        print()

        # List frames
        for frame in session.frames:
            # Count GCP observations in this frame
            num_gcps = sum(
                1 for gcp in session.gcps.values()
                if frame.frame_id in gcp.frame_observations
            )

            cprint(f"{frame.frame_id}:", Colors.CYAN, bold=True)
            print(f"  Timestamp: {frame.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  PTZ: pan={frame.ptz_position.pan:.2f}°, tilt={frame.ptz_position.tilt:.2f}°, zoom={frame.ptz_position.zoom:.1f}")
            print(f"  Image: {frame.image_path}")
            print(f"  GCP observations: {num_gcps}")

            # Show GCPs if verbose
            if args.verbose and num_gcps > 0:
                gcps_in_frame = [
                    (gcp_id, gcp.frame_observations[frame.frame_id])
                    for gcp_id, gcp in session.gcps.items()
                    if frame.frame_id in gcp.frame_observations
                ]
                print("  GCPs:")
                for gcp_id, obs in gcps_in_frame:
                    print(f"    - {gcp_id}: ({obs['u']:.1f}, {obs['v']:.1f})")
            print()

        return 0

    except Exception as e:
        print_error(f"Failed to list frames: {e}")
        logger.exception("List frames failed")
        return 1


def cmd_list_gcps(args):
    """List all GCPs in the session."""
    from poc_homography.multi_frame_session import MultiFrameCaptureSession

    print_header("GCP List")

    session_path = Path(args.session_dir)
    session_file = session_path / "session.yaml"

    if not session_file.exists():
        print_error(f"Session not found: {session_file}")
        return 1

    try:
        # Load session
        session = MultiFrameCaptureSession.load_session(str(session_file))

        if not session.gcps:
            print_warning("No GCPs added yet")
            print_info("Add GCPs with: python tools/multi_frame_capture.py add-gcp " + args.session_dir + " --id gcp_001 --lat LAT --lon LON")
            return 0

        print_info(f"Total GCPs: {len(session.gcps)}")
        print()

        # List GCPs
        for gcp_id, gcp in session.gcps.items():
            num_observations = len(gcp.frame_observations)

            cprint(f"{gcp_id}:", Colors.CYAN, bold=True)
            print(f"  GPS: ({gcp.gps_lat:.6f}, {gcp.gps_lon:.6f})")

            if gcp.utm_easting and gcp.utm_northing:
                print(f"  UTM: ({gcp.utm_easting:.2f}, {gcp.utm_northing:.2f})")

            print(f"  Observations: {num_observations} frame(s)")

            # Show observations if verbose
            if args.verbose and num_observations > 0:
                print("  Frames:")
                for frame_id, obs in gcp.frame_observations.items():
                    print(f"    - {frame_id}: ({obs['u']:.1f}, {obs['v']:.1f})")

            print()

        return 0

    except Exception as e:
        print_error(f"Failed to list GCPs: {e}")
        logger.exception("List GCPs failed")
        return 1


# ============================================================================
# Main CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-frame PTZ calibration capture tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize session
  %(prog)s init --camera Valte --output data/session_001

  # Capture frames
  %(prog)s capture data/session_001

  # Add GCP
  %(prog)s add-gcp data/session_001 --id gcp_001 --lat 39.640583 --lon -0.230194

  # Mark observation
  %(prog)s mark data/session_001 --gcp gcp_001 --frame frame_001 --u 1250.5 --v 680.0

  # Check status
  %(prog)s status data/session_001

  # Run calibration
  %(prog)s calibrate data/session_001 --output result.yaml

  # List frames and GCPs
  %(prog)s list-frames data/session_001
  %(prog)s list-gcps data/session_001
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.required = True

    # ---- init command ----
    parser_init = subparsers.add_parser('init', help='Initialize a new capture session')
    parser_init.add_argument('--camera', required=True, help='Camera name (e.g., Valte, Setram)')
    parser_init.add_argument('--output', required=True, help='Output directory for session data')
    parser_init.set_defaults(func=cmd_init)

    # ---- capture command ----
    parser_capture = subparsers.add_parser('capture', help='Capture a frame from camera')
    parser_capture.add_argument('session_dir', help='Session directory')
    parser_capture.add_argument('--wait', type=float, default=2.0, help='Wait time for camera stabilization (seconds, default: 2.0)')
    parser_capture.set_defaults(func=cmd_capture)

    # ---- add-gcp command ----
    parser_add_gcp = subparsers.add_parser('add-gcp', help='Add a Ground Control Point')
    parser_add_gcp.add_argument('session_dir', help='Session directory')
    parser_add_gcp.add_argument('--id', required=True, help='GCP identifier (e.g., gcp_001)')
    parser_add_gcp.add_argument('--lat', type=float, required=True, help='GPS latitude (decimal degrees)')
    parser_add_gcp.add_argument('--lon', type=float, required=True, help='GPS longitude (decimal degrees)')
    parser_add_gcp.add_argument('--utm-easting', type=float, help='UTM easting (meters, optional)')
    parser_add_gcp.add_argument('--utm-northing', type=float, help='UTM northing (meters, optional)')
    parser_add_gcp.set_defaults(func=cmd_add_gcp)

    # ---- mark command ----
    parser_mark = subparsers.add_parser('mark', help='Mark a GCP observation in a frame')
    parser_mark.add_argument('session_dir', help='Session directory')
    parser_mark.add_argument('--gcp', required=True, help='GCP identifier')
    parser_mark.add_argument('--frame', required=True, help='Frame identifier')
    parser_mark.add_argument('--u', type=float, required=True, help='Horizontal pixel coordinate (0 = left edge)')
    parser_mark.add_argument('--v', type=float, required=True, help='Vertical pixel coordinate (0 = top edge)')
    parser_mark.set_defaults(func=cmd_mark)

    # ---- status command ----
    parser_status = subparsers.add_parser('status', help='Show session status and validation')
    parser_status.add_argument('session_dir', help='Session directory')
    parser_status.add_argument('--verbose', '-v', action='store_true', help='Show detailed frame and GCP information')
    parser_status.set_defaults(func=cmd_status)

    # ---- calibrate command ----
    parser_calibrate = subparsers.add_parser('calibrate', help='Run multi-frame calibration')
    parser_calibrate.add_argument('session_dir', help='Session directory')
    parser_calibrate.add_argument('--output', '-o', help='Output YAML file for calibration results')
    parser_calibrate.add_argument('--loss', choices=['huber', 'cauchy'], default='huber', help='Robust loss function (default: huber)')
    parser_calibrate.add_argument('--loss-scale', type=float, default=1.0, help='Loss function scale parameter in pixels (default: 1.0)')
    parser_calibrate.add_argument('--regularization-weight', type=float, default=1.0, help='Regularization weight (default: 1.0)')
    parser_calibrate.set_defaults(func=cmd_calibrate)

    # ---- list-frames command ----
    parser_list_frames = subparsers.add_parser('list-frames', help='List all frames in session')
    parser_list_frames.add_argument('session_dir', help='Session directory')
    parser_list_frames.add_argument('--verbose', '-v', action='store_true', help='Show GCP observations in each frame')
    parser_list_frames.set_defaults(func=cmd_list_frames)

    # ---- list-gcps command ----
    parser_list_gcps = subparsers.add_parser('list-gcps', help='List all GCPs in session')
    parser_list_gcps.add_argument('session_dir', help='Session directory')
    parser_list_gcps.add_argument('--verbose', '-v', action='store_true', help='Show frame observations for each GCP')
    parser_list_gcps.set_defaults(func=cmd_list_gcps)

    # Parse arguments and execute command
    args = parser.parse_args()

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        logger.exception("Unexpected error")
        return 1


if __name__ == '__main__':
    sys.exit(main())
