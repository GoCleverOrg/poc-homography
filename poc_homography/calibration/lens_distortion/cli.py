"""Command-line interface for lens distortion calibration.

Usage:
    # Detect lines in images
    python -m poc_homography.calibration.lens_distortion.cli detect --images /path/to/images/

    # Run full calibration
    python -m poc_homography.calibration.lens_distortion.cli calibrate \
        --images /path/to/images/ \
        --fx 1000 --fy 1000 --cx 960 --cy 540

    # Calibrate from survey session
    python -m poc_homography.calibration.lens_distortion.cli calibrate \
        --survey-session webapp/survey/20240115/session_abc123/

    # Save calibration results
    python -m poc_homography.calibration.lens_distortion.cli calibrate \
        --images /path/to/images/ \
        --fx 1000 --fy 1000 --cx 960 --cy 540 \
        --output calibration.yaml \
        --camera-id my_camera
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from poc_homography.calibration.lens_distortion.calibration_table import (
    CameraCalibrationTable,
    ZoomCalibrationEntry,
)
from poc_homography.calibration.lens_distortion.distortion_solver import (
    DistortionSolver,
    SolverConfig,
)
from poc_homography.calibration.lens_distortion.line_detection import (
    LineDetectionConfig,
    LineDetector,
)
from poc_homography.calibration.lens_distortion.models import CameraLine, PTZPosition
from poc_homography.types import Degrees

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def find_images(path: Path) -> list[Path]:
    """Find all image files in a directory or return single file."""
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(path.glob(f"*{ext}"))
        images.extend(path.glob(f"*{ext.upper()}"))
    return sorted(images)


def parse_ptz_from_filename(filename: str) -> PTZPosition:
    """Try to extract PTZ from filename, return defaults if not possible.

    Expected format: prefix_pan_tilt_zoom_suffix.jpg
    Example: survey_45.5_30.0_2.0_20240115.jpg
    """
    try:
        parts = filename.replace(".jpg", "").replace(".png", "").split("_")
        # Look for numeric triplet that could be pan, tilt, zoom
        for i in range(len(parts) - 2):
            try:
                pan = float(parts[i])
                tilt = float(parts[i + 1])
                zoom = float(parts[i + 2])
                if -180 <= pan <= 180 and -90 <= tilt <= 90 and zoom > 0:
                    return PTZPosition(
                        pan_deg=Degrees(pan), tilt_deg=Degrees(tilt), zoom_factor=zoom
                    )
            except (ValueError, IndexError):
                continue
    except Exception:
        pass

    # Default values
    return PTZPosition(pan_deg=Degrees(0.0), tilt_deg=Degrees(30.0), zoom_factor=1.0)


def cmd_detect(args: argparse.Namespace) -> int:
    """Run line detection on images."""
    images_path = Path(args.images)
    if not images_path.exists():
        logger.error(f"Path not found: {images_path}")
        return 1

    images = find_images(images_path)
    if not images:
        logger.error(f"No images found in {images_path}")
        return 1

    logger.info(f"Found {len(images)} images")

    # Configure detector
    config = LineDetectionConfig(
        min_line_length=args.min_length,
        min_confidence=args.min_confidence,
    )
    detector = LineDetector(config=config)

    all_results = []
    total_lines = 0

    for img_path in images:
        try:
            candidates = detector.detect_from_file(img_path)
            total_lines += len(candidates)

            result = {
                "image": str(img_path),
                "num_lines": len(candidates),
                "lines": [
                    {
                        "start": list(c.start),
                        "end": list(c.end),
                        "confidence": round(c.confidence, 3),
                        "length": round(c.length or 0.0, 1),
                        "angle_deg": round(c.angle_deg or 0.0, 1),
                    }
                    for c in candidates[: args.max_lines]
                ],
            }
            all_results.append(result)

            if args.verbose:
                logger.info(f"{img_path.name}: {len(candidates)} lines detected")

        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")

    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved detection results to {output_path}")
    else:
        print(json.dumps(all_results, indent=2))

    logger.info(f"Total: {total_lines} lines detected across {len(images)} images")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run full calibration pipeline."""
    # Determine image source
    if args.survey_session:
        images_path = Path(args.survey_session)
    elif args.images:
        images_path = Path(args.images)
    else:
        logger.error("Must specify --images or --survey-session")
        return 1

    if not images_path.exists():
        logger.error(f"Path not found: {images_path}")
        return 1

    images = find_images(images_path)
    if not images:
        logger.error(f"No images found in {images_path}")
        return 1

    logger.info(f"Found {len(images)} images for calibration")

    # Build intrinsic matrix
    intrinsic_matrix = np.array(
        [
            [args.fx, 0.0, args.cx],
            [0.0, args.fy, args.cy],
            [0.0, 0.0, 1.0],
        ]
    )
    logger.info(f"Intrinsic matrix: fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")

    # Configure detection
    detection_config = LineDetectionConfig(
        min_line_length=args.min_length,
        min_confidence=args.min_confidence,
    )
    detector = LineDetector(config=detection_config)

    # Detect lines in all images
    all_lines: list[CameraLine] = []
    line_counter = 0

    for img_path in images:
        try:
            candidates = detector.detect_from_file(img_path)
            ptz = parse_ptz_from_filename(img_path.name)

            # Take top N candidates per image
            for c in candidates[: args.max_lines_per_image]:
                camera_line = c.to_camera_line(
                    line_id=f"line_{line_counter:04d}",
                    image_path=str(img_path),
                    ptz_position=ptz,
                )
                all_lines.append(camera_line)
                line_counter += 1

            if args.verbose:
                logger.info(
                    f"{img_path.name}: {len(candidates)} detected, "
                    f"using top {min(len(candidates), args.max_lines_per_image)}"
                )

        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")

    if len(all_lines) < args.min_total_lines:
        logger.error(f"Only {len(all_lines)} lines detected, need at least {args.min_total_lines}")
        return 1

    logger.info(f"Total lines for calibration: {len(all_lines)}")

    # Configure solver
    solver_config = SolverConfig(
        use_radial_only=args.radial_only,
        max_iterations=args.max_iterations,
    )
    solver = DistortionSolver(config=solver_config)

    # Run optimization
    logger.info("Running distortion coefficient optimization...")
    result = solver.solve(all_lines, intrinsic_matrix)

    # Report results
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Optimization: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Iterations: {result.iterations}")
    print(f"Message: {result.message}")
    print()
    print(f"Initial straightness error: {result.initial_error:.4f}")
    print(f"Final straightness error:   {result.final_error:.4f}")
    print(f"Improvement: {(1 - result.improvement_ratio()) * 100:.1f}%")
    print(f"Overall RMSE: {result.overall_rmse:.3f} pixels")
    print()
    print("Distortion Coefficients:")
    print(f"  k1 = {float(result.distortion.k1):+.6f}  (radial)")
    print(f"  k2 = {float(result.distortion.k2):+.6f}  (radial)")
    print(f"  k3 = {float(result.distortion.k3):+.6f}  (radial)")
    print(f"  p1 = {float(result.distortion.p1):+.6f}  (tangential)")
    print(f"  p2 = {float(result.distortion.p2):+.6f}  (tangential)")
    print()

    # Quality assessment
    if result.overall_rmse < 2.0:
        print("✓ RMSE < 2 pixels - GOOD calibration quality")
    elif result.overall_rmse < 5.0:
        print("⚠ RMSE 2-5 pixels - ACCEPTABLE calibration quality")
    else:
        print("✗ RMSE > 5 pixels - POOR calibration quality, consider more/better lines")

    if result.is_improved():
        print("✓ Calibration improved straightness")
    else:
        print("⚠ Calibration did not improve straightness")

    print("=" * 60)

    # Per-line details if verbose
    if args.verbose:
        print("\nPer-line RMSE (pixels):")
        for err in sorted(result.line_errors, key=lambda x: x["rmse_pixels"], reverse=True)[:10]:
            print(f"  {err['line_id']}: {err['rmse_pixels']:.3f}")
        if len(result.line_errors) > 10:
            print(f"  ... and {len(result.line_errors) - 10} more")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        camera_id = args.camera_id or "unknown_camera"
        zoom = args.zoom or 1.0

        table = CameraCalibrationTable(camera_id=camera_id)
        entry = ZoomCalibrationEntry.from_solver_result(
            zoom_factor=zoom,
            distortion=result.distortion,
            validation_rmse=result.overall_rmse,
            source_images=[str(p) for p in images[:10]],  # First 10 as reference
            num_lines_used=len(all_lines),
        )
        table.add_entry(entry)
        table.save(output_path)
        logger.info(f"Saved calibration to {output_path}")

    # Also output JSON for programmatic use
    if args.json:
        output = {
            "success": result.success,
            "iterations": result.iterations,
            "initial_error": result.initial_error,
            "final_error": result.final_error,
            "overall_rmse": result.overall_rmse,
            "num_lines": len(all_lines),
            "coefficients": {
                "k1": float(result.distortion.k1),
                "k2": float(result.distortion.k2),
                "k3": float(result.distortion.k3),
                "p1": float(result.distortion.p1),
                "p2": float(result.distortion.p2),
            },
        }
        print("\nJSON output:")
        print(json.dumps(output, indent=2))

    return 0 if result.success and result.is_improved() else 1


def cmd_visualize(args: argparse.Namespace) -> int:
    """Visualize detected lines on an image."""
    import cv2

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return 1

    config = LineDetectionConfig(
        min_line_length=args.min_length,
        min_confidence=args.min_confidence,
    )
    detector = LineDetector(config=config)

    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return 1

    candidates = detector.detect(img)
    logger.info(f"Detected {len(candidates)} lines")

    output = detector.visualize(img, candidates, show_confidence=True)

    output_path = Path(args.output) if args.output else image_path.with_suffix(".detected.jpg")
    cv2.imwrite(str(output_path), output)
    logger.info(f"Saved visualization to {output_path}")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Lens distortion calibration using parking spot lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # detect command
    detect_parser = subparsers.add_parser("detect", help="Detect lines in images")
    detect_parser.add_argument("--images", "-i", required=True, help="Image or directory path")
    detect_parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    detect_parser.add_argument(
        "--min-length", type=int, default=100, help="Minimum line length in pixels"
    )
    detect_parser.add_argument(
        "--min-confidence", type=float, default=0.3, help="Minimum confidence threshold"
    )
    detect_parser.add_argument(
        "--max-lines", type=int, default=50, help="Max lines per image to report"
    )

    # calibrate command
    cal_parser = subparsers.add_parser("calibrate", help="Run full calibration")
    cal_parser.add_argument("--images", "-i", help="Image or directory path")
    cal_parser.add_argument("--survey-session", "-s", help="Survey session directory")
    cal_parser.add_argument("--fx", type=float, default=1000.0, help="Focal length X")
    cal_parser.add_argument("--fy", type=float, default=1000.0, help="Focal length Y")
    cal_parser.add_argument("--cx", type=float, default=960.0, help="Principal point X")
    cal_parser.add_argument("--cy", type=float, default=540.0, help="Principal point Y")
    cal_parser.add_argument("--output", "-o", help="Output YAML calibration file")
    cal_parser.add_argument("--camera-id", help="Camera identifier for output file")
    cal_parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor for calibration")
    cal_parser.add_argument(
        "--min-length", type=int, default=100, help="Minimum line length in pixels"
    )
    cal_parser.add_argument(
        "--min-confidence", type=float, default=0.3, help="Minimum confidence threshold"
    )
    cal_parser.add_argument(
        "--max-lines-per-image", type=int, default=10, help="Max lines per image"
    )
    cal_parser.add_argument(
        "--min-total-lines", type=int, default=5, help="Minimum total lines required"
    )
    cal_parser.add_argument(
        "--radial-only", action="store_true", help="Only optimize radial coefficients (k1,k2,k3)"
    )
    cal_parser.add_argument(
        "--max-iterations", type=int, default=1000, help="Maximum optimizer iterations"
    )
    cal_parser.add_argument("--json", action="store_true", help="Also output JSON results")

    # visualize command
    vis_parser = subparsers.add_parser("visualize", help="Visualize detected lines")
    vis_parser.add_argument("--image", "-i", required=True, help="Image path")
    vis_parser.add_argument("--output", "-o", help="Output image path")
    vis_parser.add_argument(
        "--min-length", type=int, default=100, help="Minimum line length in pixels"
    )
    vis_parser.add_argument(
        "--min-confidence", type=float, default=0.3, help="Minimum confidence threshold"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == "detect":
        return cmd_detect(args)
    elif args.command == "calibrate":
        return cmd_calibrate(args)
    elif args.command == "visualize":
        return cmd_visualize(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
