"""GCP (Ground Control Point) CLI commands."""

import webbrowser
from pathlib import Path

import typer

from poc_homography.camera_config import get_camera_configs
from poc_homography.cli.main import gcp_app
from poc_homography.coordinates import dms_to_dd
from poc_homography.gcp.verify import (
    CameraLocation,
    generate_verification_map,
    load_gcps_from_yaml,
)
from poc_homography.types import Degrees, Meters


def _get_camera_config_decimal(camera_name: str) -> dict[str, float]:
    """
    Get camera configuration and convert DMS to decimal degrees.

    Args:
        camera_name: Name of the camera

    Returns:
        Dictionary with lat, lon, height_m, pan_offset_deg in decimal degrees
    """
    configs = {cam["name"]: cam for cam in get_camera_configs()}
    if camera_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown camera: {camera_name}. Available: {available}")

    cam = configs[camera_name]

    return {
        "lat": dms_to_dd(cam["lat"]),
        "lon": dms_to_dd(cam["lon"]),
        "height_m": cam["height_m"],
        "pan_offset_deg": cam["pan_offset_deg"],
    }


@gcp_app.command("verify")
def verify_command(
    gcps_file: Path = typer.Option(..., help="Path to GCPs YAML file"),
    camera: str | None = typer.Option(
        None, help="Camera name (e.g., 'Valte') for distance/bearing calculations"
    ),
    output: Path | None = typer.Option(
        None, help="Output HTML file path (default: gcps_file.html)"
    ),
    open_browser: bool = typer.Option(True, help="Open map in browser after generation"),
    image_height: int = typer.Option(1080, help="Image height for legacy coordinate conversion"),
) -> None:
    """
    Generate verification map for GCP GPS coordinates.

    Creates an interactive HTML map with GCPs plotted on satellite imagery
    for visual inspection. Optionally includes camera location and FOV cone
    if camera parameter is provided.

    The GCPs YAML file should contain either:

    Simple format:
        gcps:
          - lat: 39.64
            lon: -0.23
            name: "GCP1"
            pixel_u: 100
            pixel_v: 200

    Complex format (from capture tool):
        homography:
          feature_match:
            camera_capture_context:
              camera_name: "Valte"
              ptz_position:
                pan: 45.0
                tilt: 30.0
                zoom: 5.0
            ground_control_points:
              - gps:
                  latitude: 39.64
                  longitude: -0.23
                image:
                  u: 100
                  v: 200
                metadata:
                  description: "GCP1"

    Example:
        hom gcp verify --gcps-file gcps.yaml
        hom gcp verify --gcps-file gcps.yaml --camera Valte --output verification.html
        hom gcp verify --gcps-file gcps.yaml --camera Valte --no-open-browser
    """
    # Load GCPs from YAML file
    try:
        gcps, ptz_info, metadata = load_gcps_from_yaml(gcps_file, image_height)
    except FileNotFoundError:
        typer.echo(f"Error: GCPs file not found: {gcps_file}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: Failed to load GCPs: {e}", err=True)
        raise typer.Exit(1)

    if not gcps:
        typer.echo("Error: No GCPs found in YAML file", err=True)
        raise typer.Exit(1)

    # Get camera configuration if specified
    camera_config: CameraLocation | None = None
    if camera:
        try:
            config = _get_camera_config_decimal(camera)
            camera_config = CameraLocation(
                lat=Degrees(config["lat"]),
                lon=Degrees(config["lon"]),
                height_m=Meters(config["height_m"]),
                pan_offset_deg=Degrees(config["pan_offset_deg"]),
            )
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

    # Determine output file
    if output is None:
        output = gcps_file.with_suffix(".html")

    # Generate HTML map
    typer.echo(f"Generating verification map with {len(gcps)} GCPs...")
    html = generate_verification_map(
        gcps=gcps,
        camera_config=camera_config,
        ptz_info=ptz_info,
        metadata=metadata,
        title=f"GCP Verification - {gcps_file.name}",
    )

    # Write to file
    output.write_text(html, encoding="utf-8")
    typer.echo(f"Map saved to: {output}")

    # Open in browser if requested
    if open_browser:
        typer.echo("Opening map in browser...")
        webbrowser.open(f"file://{output.resolve()}")
