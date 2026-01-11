"""Testing CLI commands."""

import json
import os
from datetime import datetime
from pathlib import Path

import typer

from poc_homography.camera_config import CAMERAS
from poc_homography.cli.main import test_app
from poc_homography.testing.data_generator import run_data_generator
from poc_homography.testing.sam3 import PROMPTS_TO_TEST, PromptTestResult, test_prompts


@test_app.command("sam3")
def sam3_command(
    image: Path = typer.Argument(..., help="Path to test image"),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save mask/overlay images (default: ./prompt_test_results/)",
    ),
    save_json: bool = typer.Option(
        False,
        "--save-json",
        help="Save results to JSON file in output directory",
    ),
) -> None:
    """
    Test alternative prompts for SAM3 road marking detection.

    Tests multiple prompts on the given image and compares results.
    Produces a comparison table showing detection quality metrics.

    Requires ROBOFLOW_API_KEY environment variable to be set.

    Example:
        hom test sam3 test_image.jpg
        hom test sam3 test_image.jpg --output-dir ./results --save-json
    """
    # Check for API key
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        typer.echo(
            "Error: ROBOFLOW_API_KEY environment variable not set",
            err=True,
        )
        raise typer.Exit(1)

    # Validate image exists
    if not image.exists():
        typer.echo(f"Error: Image not found: {image}", err=True)
        raise typer.Exit(1)

    # Set default output directory if not specified
    if output_dir is None:
        output_dir = Path("./prompt_test_results")

    # Run tests
    try:
        results = test_prompts(
            image_path=image,
            api_key=api_key,
            output_dir=output_dir,
            verbose=True,
        )
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Print results table
    _print_results_table(results)

    # Save JSON if requested
    if save_json:
        json_path = output_dir / "results.json"
        _save_results_json(results, json_path)


def _print_results_table(results: list[PromptTestResult]) -> None:
    """Print results as a markdown table."""
    print("\n" + "=" * 80)
    print("SAM3 PROMPT COMPARISON RESULTS")
    print("=" * 80)

    # Markdown table header
    print("\n| Prompt | Detections | Polygons | Coverage % | Avg Confidence | Quality Assessment |")
    print("|--------|-----------|----------|------------|----------------|-------------------|")

    # Find best result for comparison
    valid_results = [r for r in results if not r.error]
    best_coverage = max((r.coverage for r in valid_results), default=0)
    best_detections = max((r.detections for r in valid_results), default=0)

    for result in results:
        if result.error:
            print(f"| {result.prompt} | ERROR | - | - | - | {result.error[:30]} |")
            continue

        # Quality assessment
        quality: list[str] = []
        if result.coverage >= best_coverage * 0.9:
            quality.append("High coverage")
        if result.detections >= best_detections * 0.9:
            quality.append("Good detection count")
        if result.avg_confidence >= 0.7:
            quality.append("High confidence")
        elif result.avg_confidence >= 0.5:
            quality.append("Medium confidence")
        else:
            quality.append("Low confidence")

        quality_str = "; ".join(quality) if quality else "N/A"

        print(
            f"| {result.prompt} | {result.detections} | {result.polygons} | "
            f"{result.coverage:.2f} | {result.avg_confidence:.3f} | {quality_str} |"
        )

    print("\n" + "=" * 80)

    # Recommendation
    if valid_results:
        # Sort by a combined score (coverage * avg_confidence)
        best = max(valid_results, key=lambda r: r.coverage * r.avg_confidence)
        print(f"\nRECOMMENDATION: Best performing prompt is '{best.prompt}'")
        print(f"  - Coverage: {best.coverage:.2f}%")
        print(f"  - Detections: {best.detections}")
        print(f"  - Avg Confidence: {best.avg_confidence:.3f}")

        if best.prompt != "road markings":
            print(f"\nSuggestion: Consider updating DEFAULT_SAM3_PROMPT to '{best.prompt}'")
        else:
            print("\nThe current default 'road markings' appears to be optimal.")


def _save_results_json(results: list[PromptTestResult], output_path: Path) -> None:
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "prompts_tested": PROMPTS_TO_TEST,
        "results": [
            {
                "prompt": r.prompt,
                "detections": r.detections,
                "polygons": r.polygons,
                "coverage": r.coverage,
                "avg_confidence": float(r.avg_confidence),
                "min_confidence": float(r.min_confidence),
                "max_confidence": float(r.max_confidence),
                "error": r.error,
            }
            for r in results
        ],
    }
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


@test_app.command("data-generator")
def data_generator_command(
    camera_name: str | None = typer.Argument(None, help="Camera name (e.g., Valte, Setram)"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file path (default: test_data_{camera}_{timestamp}.json)",
    ),
    map_points: Path | None = typer.Option(
        None,
        "--map-points",
        "-m",
        help="Path to map points JSON file",
    ),
    list_cameras: bool = typer.Option(
        False,
        "--list-cameras",
        help="List available cameras and exit",
    ),
) -> None:
    """
    Generate test data for camera calibration with interactive GCP marking.

    This tool captures a full-resolution frame from the specified camera,
    fetches current PTZ parameters, and launches a web interface for
    interactive Ground Control Point (GCP) marking.

    The web interface allows you to:
    - Click on the image to mark GCP locations
    - Search and select map points from a registry
    - Adjust camera parameters (lat/lon/height/pan/tilt/zoom)
    - Export marked GCPs as JSON with the captured frame

    Example:
        hom test data-generator Valte
        hom test data-generator Setram --output my_test.json --map-points map_points.json
        hom test data-generator --list-cameras
    """
    # Handle --list-cameras
    if list_cameras:
        typer.echo("Available cameras:")
        for cam in CAMERAS:
            typer.echo(f"  - {cam['name']} ({cam['ip']})")
        raise typer.Exit(0)

    # Validate camera_name is provided
    if not camera_name:
        typer.echo(
            "Error: CAMERA_NAME is required unless --list-cameras is specified",
            err=True,
        )
        raise typer.Exit(1)

    # Validate camera exists
    available_names = [cam["name"] for cam in CAMERAS]
    if camera_name not in available_names:
        typer.echo(
            f"Error: Camera '{camera_name}' not found. Available: {', '.join(available_names)}",
            err=True,
        )
        raise typer.Exit(1)

    # Run data generator
    try:
        run_data_generator(
            camera_name=camera_name,
            output_path=str(output) if output else None,
            map_points_path=map_points,
        )
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
