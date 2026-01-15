"""Testing CLI commands."""

from pathlib import Path

import typer

from poc_homography.application import ApplicationContext
from poc_homography.cli.main import test_app
from poc_homography.testing.data_generator import run_data_generator


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
        hom test data-generator Setram --output my_test.json --map-points valte_map_points.yaml
        hom test data-generator --list-cameras
    """
    # Get camera configurations from repository
    ctx = ApplicationContext.default()
    all_configs = ctx.repo_camera_config.get_all()

    # Handle --list-cameras
    if list_cameras:
        typer.echo("Available cameras:")
        for cam in all_configs:
            ip = cam.ip_address or "no IP"
            typer.echo(f"  - {cam.name} ({ip})")
        raise typer.Exit(0)

    # Validate camera_name is provided
    if not camera_name:
        typer.echo(
            "Error: CAMERA_NAME is required unless --list-cameras is specified",
            err=True,
        )
        raise typer.Exit(1)

    # Validate camera exists
    available_names = [cam.name for cam in all_configs]
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
