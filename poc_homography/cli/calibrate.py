"""Calibration CLI commands."""

from pathlib import Path

import typer
import yaml

from poc_homography.calibration import (
    GCP,
    TARGET_ERROR_THRESHOLD_PX,
    analyze_projection_error,
    print_results,
    run_calibration,
)
from poc_homography.camera_config import get_camera_configs
from poc_homography.cli.main import calibrate_app
from poc_homography.coordinates import dms_to_dd
from poc_homography.map_points import MapPointRegistry
from poc_homography.types import Degrees, Meters, Pixels, PixelsFloat, Unitless


def _get_camera_config(camera_name: str) -> dict[str, float]:
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


@calibrate_app.command("projection")
def projection_command(
    camera: str = typer.Option(..., help="Camera name (e.g., 'camera1')"),
    map_x: float = typer.Option(..., help="Map point X coordinate (meters, East)"),
    map_y: float = typer.Option(..., help="Map point Y coordinate (meters, North)"),
    pixel_u: float = typer.Option(..., help="Actual pixel U coordinate (clicked by user)"),
    pixel_v: float = typer.Option(..., help="Actual pixel V coordinate (clicked by user)"),
    pan_raw: float = typer.Option(..., help="Raw pan value from camera PTZ (degrees)"),
    tilt: float = typer.Option(..., help="Tilt angle in degrees"),
    zoom: float = typer.Option(..., help="Zoom factor"),
    width: int = typer.Option(1920, help="Image width in pixels"),
    height: int = typer.Option(1080, help="Image height in pixels"),
) -> None:
    """
    Analyze projection error to identify calibration issues.

    This command helps identify which parameter is causing projection misalignment:
    - Pan offset
    - Camera height
    - Focal length / intrinsics

    Example:
        hom calibrate projection --camera camera1 --map-x 10.5 --map-y 20.3
            --pixel-u 960 --pixel-v 540 --pan-raw 45.0 --tilt 30.0 --zoom 5.0
    """
    # Get camera configuration
    try:
        config = _get_camera_config(camera)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Convert CLI inputs to typed units at the boundary
    result = analyze_projection_error(
        camera_lat=Degrees(config["lat"]),
        camera_lon=Degrees(config["lon"]),
        height_m=Meters(config["height_m"]),
        pan_offset_deg=Degrees(config["pan_offset_deg"]),
        map_point_x=Meters(map_x),
        map_point_y=Meters(map_y),
        actual_u=PixelsFloat(pixel_u),
        actual_v=PixelsFloat(pixel_v),
        pan_raw=Degrees(pan_raw),
        tilt_deg=Degrees(tilt),
        zoom=Unitless(zoom),
        image_width=Pixels(width),
        image_height=Pixels(height),
    )

    # Exit with code 1 if adjustments are needed (for scripting)
    if result.needs_pan_adjustment or result.needs_height_adjustment:
        raise typer.Exit(1)


@calibrate_app.command("comprehensive")
def comprehensive_command(
    camera: str = typer.Option(..., help="Camera name (e.g., 'Valte')"),
    gcps_file: Path = typer.Option(..., help="Path to GCPs YAML file"),
    registry_file: Path = typer.Option(..., help="Path to map point registry JSON file"),
    width: int = typer.Option(1920, help="Image width in pixels"),
    height: int = typer.Option(1080, help="Image height in pixels"),
    optimize_position: bool = typer.Option(True, help="Optimize camera X/Y position"),
    optimize_focal: bool = typer.Option(True, help="Optimize focal length multiplier"),
    optimize_pan: bool = typer.Option(True, help="Optimize pan offset"),
    optimize_tilt: bool = typer.Option(True, help="Optimize tilt offset"),
    optimize_distortion: bool = typer.Option(True, help="Optimize lens distortion (k1, k2)"),
) -> None:
    """
    Comprehensive calibration to optimize all camera parameters.

    This command optimizes ALL parameters that affect projection accuracy:
    - Pan offset (camera home position bearing)
    - Camera height
    - Camera position offset (X/Y in map coordinates)
    - Focal length multiplier
    - Tilt offset
    - Lens distortion (k1, k2)

    The GCPs YAML file should contain:
        gcps:
          - map_point_id: Z1
            pixel_u: 960.0
            pixel_v: 540.0
            pan_raw: 0.0
            tilt_deg: 30.0
            zoom: 1.0
          - map_point_id: Z2
            ...

    Example:
        hom calibrate comprehensive --camera Valte --gcps-file gcps.yaml
            --registry-file map_points.json
    """
    # Get camera configuration
    configs = {cam["name"]: cam for cam in get_camera_configs()}
    if camera not in configs:
        available = ", ".join(configs.keys())
        typer.echo(f"Error: Unknown camera: {camera}. Available: {available}", err=True)
        raise typer.Exit(1)

    camera_config = configs[camera]

    # Load GCPs from YAML file
    try:
        with gcps_file.open() as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        typer.echo(f"Error: GCPs file not found: {gcps_file}", err=True)
        raise typer.Exit(1)
    except yaml.YAMLError as e:
        typer.echo(f"Error: Invalid YAML in {gcps_file}: {e}", err=True)
        raise typer.Exit(1)

    gcps: list[GCP] = []
    for gcp_data in data.get("gcps", []):
        gcps.append(
            GCP(
                map_point_id=gcp_data["map_point_id"],
                pixel_u=PixelsFloat(gcp_data["pixel_u"]),
                pixel_v=PixelsFloat(gcp_data["pixel_v"]),
                pan_raw=Degrees(gcp_data.get("pan_raw", 0.0)),
                tilt_deg=Degrees(gcp_data.get("tilt_deg", 30.0)),
                zoom=Unitless(gcp_data.get("zoom", 1.0)),
            )
        )

    if not gcps:
        typer.echo("Error: No GCPs found in YAML file", err=True)
        raise typer.Exit(1)

    # Load map point registry
    try:
        registry = MapPointRegistry.load(registry_file)
    except FileNotFoundError:
        typer.echo(f"Error: Registry file not found: {registry_file}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: Failed to load registry: {e}", err=True)
        raise typer.Exit(1)

    # Run calibration
    optimized_params, mean_error, individual_errors = run_calibration(
        camera_config=camera_config,
        gcps=gcps,
        registry=registry,
        optimize_position=optimize_position,
        optimize_focal=optimize_focal,
        optimize_pan=optimize_pan,
        optimize_tilt=optimize_tilt,
        optimize_distortion=optimize_distortion,
        image_width=Pixels(width),
        image_height=Pixels(height),
        verbose=True,
    )

    # Print results
    print_results(camera_config, optimized_params, mean_error, individual_errors, gcps)

    # Exit with code 1 if target accuracy not achieved
    if mean_error >= TARGET_ERROR_THRESHOLD_PX:
        raise typer.Exit(1)
