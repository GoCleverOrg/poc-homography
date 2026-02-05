"""Command-line interface for lens distortion calibration.

Usage:
    # Detect lines in images
    python -m poc_homography.calibration.lens_distortion.cli detect --images /path/to/images/

    # Run full calibration (fx/fy default to factory-derived 1670.8 px at zoom=1)
    python -m poc_homography.calibration.lens_distortion.cli calibrate \
        --images /path/to/images/

    # Calibrate from survey session
    python -m poc_homography.calibration.lens_distortion.cli calibrate \
        --survey-session webapp/survey/20240115/session_abc123/

    # Save calibration results
    python -m poc_homography.calibration.lens_distortion.cli calibrate \
        --images /path/to/images/ \
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
from poc_homography.camera_config import (
    DEFAULT_BASE_FOCAL_LENGTH_MM,
    DEFAULT_SENSOR_WIDTH_MM,
)
from poc_homography.types import Degrees

# Factory-derived focal length in pixels at zoom=1 for 1920x1080
# f_px = base_focal_length_mm * (image_width / sensor_width_mm)
_DEFAULT_FX = DEFAULT_BASE_FOCAL_LENGTH_MM * (1920 / DEFAULT_SENSOR_WIDTH_MM)
_DEFAULT_FY = _DEFAULT_FX

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


def group_images_by_zoom(
    images: list[Path],
    tolerance: float = 0.1,
) -> dict[float, list[Path]]:
    """Group images by zoom factor extracted from filenames.

    Args:
        images: List of image paths.
        tolerance: Rounding tolerance for zoom values (default 0.1).

    Returns:
        Dictionary mapping rounded zoom factor to list of image paths.
    """
    zoom_groups: dict[float, list[Path]] = {}
    for img_path in images:
        ptz = parse_ptz_from_filename(img_path.name)
        # Round to nearest tolerance to group similar zoom values
        zoom = round(ptz.zoom_factor / tolerance) * tolerance
        zoom_groups.setdefault(zoom, []).append(img_path)
    return zoom_groups


def cmd_calibrate_batch(args: argparse.Namespace) -> int:
    """Run batch calibration across multiple zoom levels.

    This command automatically groups images by zoom level (parsed from filenames),
    runs calibration independently for each zoom, and outputs a single calibration
    table with all results.
    """
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

    logger.info(f"Found {len(images)} total images")

    # Group images by zoom factor
    zoom_groups = group_images_by_zoom(images)
    sorted_zooms = sorted(zoom_groups.keys())

    # Report zoom groups found
    zoom_summary = ", ".join(
        f"{zoom:.1f} ({len(zoom_groups[zoom])} images)" for zoom in sorted_zooms
    )
    print(f"\nFound {len(sorted_zooms)} zoom levels: {zoom_summary}")

    # Check for default zoom (1.0) which might indicate parsing failures
    default_zoom_count = len(zoom_groups.get(1.0, []))
    total_images = len(images)
    if default_zoom_count > 0 and default_zoom_count == total_images:
        logger.warning(
            "All images have default zoom (1.0). "
            "Check that filenames follow pattern: prefix_pan_tilt_zoom_suffix.jpg"
        )

    # Validate zoom groups before calibration
    min_images = args.min_images_per_zoom
    insufficient_zooms = []
    for zoom, group_images in zoom_groups.items():
        if len(group_images) < min_images:
            insufficient_zooms.append((zoom, len(group_images)))

    if insufficient_zooms:
        for zoom, count in insufficient_zooms:
            logger.error(
                f"ERROR: Insufficient data for zoom {zoom:.1f}\n"
                f"  Found: {count} images (need {min_images})"
            )
        print("\nCalibration aborted.")
        return 1

    # Configure detection
    detection_config = LineDetectionConfig(
        min_line_length=args.min_length,
        min_confidence=args.min_confidence,
    )
    detector = LineDetector(config=detection_config)

    # Configure solver
    solver_config = SolverConfig(
        use_radial_only=args.radial_only,
        max_iterations=args.max_iterations,
    )
    solver = DistortionSolver(config=solver_config)

    # Initialize calibration table
    camera_id = args.camera_id or "unknown_camera"
    table = CameraCalibrationTable(camera_id=camera_id)

    # Track results for summary
    results_summary: list[dict] = []
    failed_zooms: list[tuple[float, str]] = []

    print("\n" + "=" * 60)
    print("BATCH CALIBRATION")
    print("=" * 60)

    for zoom in sorted_zooms:
        group_images = zoom_groups[zoom]
        print(f"\nProcessing zoom {zoom:.1f} ({len(group_images)} images)...")

        # Detect lines in all images for this zoom
        all_lines: list[CameraLine] = []
        line_counter = 0

        for img_path in group_images:
            try:
                candidates = detector.detect_from_file(img_path)
                ptz = parse_ptz_from_filename(img_path.name)

                # Take top N candidates per image
                for c in candidates[: args.max_lines_per_image]:
                    camera_line = c.to_camera_line(
                        line_id=f"zoom{zoom:.1f}_line_{line_counter:04d}",
                        image_path=str(img_path),
                        ptz_position=ptz,
                    )
                    all_lines.append(camera_line)
                    line_counter += 1

                if args.verbose:
                    logger.info(
                        f"  {img_path.name}: {len(candidates)} detected, "
                        f"using top {min(len(candidates), args.max_lines_per_image)}"
                    )

            except Exception as e:
                logger.warning(f"  Failed to process {img_path}: {e}")

        print(f"  Detected {len(all_lines)} lines")

        # Validate minimum lines
        min_lines = args.min_lines_per_zoom
        if len(all_lines) < min_lines:
            error_msg = f"Only {len(all_lines)} lines detected (need {min_lines})"
            failed_zooms.append((zoom, error_msg))
            logger.error(f"ERROR: Insufficient lines for zoom {zoom:.1f}")
            logger.error(f"  Found: {len(all_lines)} lines (need {min_lines})")
            continue

        # Build intrinsic matrix with zoom-scaled focal length
        # fx_zoom = base_fx * zoom_factor
        fx_zoom = args.base_fx * zoom
        fy_zoom = args.base_fy * zoom
        intrinsic_matrix = np.array(
            [
                [fx_zoom, 0.0, args.cx],
                [0.0, fy_zoom, args.cy],
                [0.0, 0.0, 1.0],
            ]
        )

        if args.verbose:
            logger.info(f"  Intrinsics: fx={fx_zoom:.1f}, fy={fy_zoom:.1f}")

        # Run optimization
        print("  Running calibration...")
        result = solver.solve(all_lines, intrinsic_matrix)

        if not result.success:
            error_msg = f"Optimization failed: {result.message}"
            failed_zooms.append((zoom, error_msg))
            logger.warning(f"  {error_msg}")
            continue

        # Create calibration entry
        entry = ZoomCalibrationEntry.from_solver_result(
            zoom_factor=zoom,
            distortion=result.distortion,
            validation_rmse=result.overall_rmse,
            source_images=[str(p) for p in group_images[:10]],  # First 10 as reference
            num_lines_used=len(all_lines),
            fx=fx_zoom,
            fy=fy_zoom,
            cx=args.cx,
            cy=args.cy,
        )
        table.add_entry(entry)

        # Track result
        results_summary.append(
            {
                "zoom": zoom,
                "images": len(group_images),
                "lines": len(all_lines),
                "rmse": result.overall_rmse,
                "fx": fx_zoom,
                "fy": fy_zoom,
                "k1": float(result.distortion.k1),
                "k2": float(result.distortion.k2),
            }
        )

        # Quality assessment
        if result.overall_rmse < 2.0:
            print(f"  RMSE: {result.overall_rmse:.2f} pixels ✓")
        elif result.overall_rmse < 5.0:
            print(f"  RMSE: {result.overall_rmse:.2f} pixels ⚠ (acceptable)")
        else:
            print(f"  RMSE: {result.overall_rmse:.2f} pixels ✗ (poor quality)")
            logger.warning(f"  RMSE > 5 pixels for zoom {zoom:.1f}, consider more/better lines")

    # Check for failures
    if failed_zooms:
        print("\n" + "=" * 60)
        print("CALIBRATION FAILED")
        print("=" * 60)
        for zoom, error in failed_zooms:
            print(f"  Zoom {zoom:.1f}: {error}")
        print("\nCalibration aborted. No output file written.")
        return 1

    if not table.entries:
        logger.error("No successful calibrations. Check image quality and line detection.")
        return 1

    # Save results
    if args.output:
        output_path = Path(args.output)
        table.save(output_path)

    # Print summary
    print("\n" + "=" * 60)
    print(table.summary())
    print("=" * 60)

    if args.output:
        print(f"\nSaved calibration table to {args.output}")

    # Print JSON if requested
    if args.json:
        output = {
            "camera_id": camera_id,
            "zoom_levels": len(results_summary),
            "entries": results_summary,
        }
        print("\nJSON output:")
        import json

        print(json.dumps(output, indent=2))

    return 0


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
    cal_parser.add_argument(
        "--fx",
        type=float,
        default=_DEFAULT_FX,
        help=f"Focal length X in pixels (default: {_DEFAULT_FX:.1f} from factory specs)",
    )
    cal_parser.add_argument(
        "--fy",
        type=float,
        default=_DEFAULT_FY,
        help=f"Focal length Y in pixels (default: {_DEFAULT_FY:.1f} from factory specs)",
    )
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

    # calibrate-batch command
    batch_parser = subparsers.add_parser(
        "calibrate-batch",
        help="Run batch calibration across multiple zoom levels",
        description="""
Run lens distortion calibration across multiple zoom levels automatically.

This command groups images by zoom level (parsed from filenames), runs
calibration independently for each zoom, and outputs a single calibration
table with all results.

Expected filename format: prefix_pan_tilt_zoom_suffix.jpg
Example: valte_cam01_20260126_095620_30.0_15.3_1.0.jpg (zoom=1.0)

The output YAML file can be used with CameraCalibrationTable.load() for
zoom-interpolated distortion correction.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic batch calibration
  python -m poc_homography.calibration.lens_distortion.cli calibrate-batch \\
      --images /path/to/multi_zoom_images/ \\
      --output camera_calibration.yaml \\
      --camera-id valte_cam01

  # With custom thresholds
  python -m poc_homography.calibration.lens_distortion.cli calibrate-batch \\
      --images survey/valte_cam01/ \\
      --output calibration_results/valte_cam01_calibration.yaml \\
      --camera-id valte_cam01 \\
      --min-images-per-zoom 5 \\
      --min-lines-per-zoom 15 \\
      --verbose
        """,
    )
    batch_parser.add_argument("--images", "-i", help="Image or directory path")
    batch_parser.add_argument("--survey-session", "-s", help="Survey session directory")
    batch_parser.add_argument(
        "--base-fx",
        type=float,
        default=_DEFAULT_FX,
        help=f"Base focal length X at zoom=1 (default: {_DEFAULT_FX:.1f}). Scaled by zoom factor.",
    )
    batch_parser.add_argument(
        "--base-fy",
        type=float,
        default=_DEFAULT_FY,
        help=f"Base focal length Y at zoom=1 (default: {_DEFAULT_FY:.1f}). Scaled by zoom factor.",
    )
    batch_parser.add_argument(
        "--cx", type=float, default=960.0, help="Principal point X (default: 960)"
    )
    batch_parser.add_argument(
        "--cy", type=float, default=540.0, help="Principal point Y (default: 540)"
    )
    batch_parser.add_argument("--output", "-o", help="Output YAML calibration file")
    batch_parser.add_argument(
        "--camera-id", help="Camera identifier for output file", required=True
    )
    batch_parser.add_argument(
        "--min-images-per-zoom",
        type=int,
        default=3,
        help="Minimum images per zoom group (default: 3)",
    )
    batch_parser.add_argument(
        "--min-lines-per-zoom",
        type=int,
        default=10,
        help="Minimum lines per zoom group (default: 10)",
    )
    batch_parser.add_argument(
        "--min-length", type=int, default=100, help="Minimum line length in pixels"
    )
    batch_parser.add_argument(
        "--min-confidence", type=float, default=0.3, help="Minimum confidence threshold"
    )
    batch_parser.add_argument(
        "--max-lines-per-image", type=int, default=10, help="Max lines per image"
    )
    batch_parser.add_argument(
        "--radial-only",
        action="store_true",
        help="Only optimize radial coefficients (k1,k2,k3)",
    )
    batch_parser.add_argument(
        "--max-iterations", type=int, default=1000, help="Maximum optimizer iterations"
    )
    batch_parser.add_argument("--json", action="store_true", help="Also output JSON results")

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
    elif args.command == "calibrate-batch":
        return cmd_calibrate_batch(args)
    elif args.command == "visualize":
        return cmd_visualize(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
