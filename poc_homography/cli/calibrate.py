"""Calibration CLI commands."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
import yaml

from poc_homography.application import ApplicationContext
from poc_homography.calibration import (
    TARGET_ERROR_THRESHOLD_PX,
    print_results,
    run_calibration,
)
from poc_homography.cli.main import calibrate_app
from poc_homography.domain import CameraCalibration, CameraConfig
from poc_homography.domain.entities.ground_control_point import GroundControlPoint
from poc_homography.domain.vo.map_point import MapPoint
from poc_homography.domain.vo.pixel_point import PixelPoint
from poc_homography.types import Degrees, Pixels, PixelsFloat, Unitless


def _build_legacy_camera_dict(
    config: CameraConfig,
    calibration: CameraCalibration | None,
) -> dict[str, Any]:
    """Build a legacy camera config dict from domain entities.

    This adapter function creates the dict format expected by legacy calibration code.
    """
    result: dict[str, Any] = {
        "name": config.name,
        "ip": config.ip_address,
    }

    if calibration:
        result["height_m"] = float(calibration.height)
        result["pan_offset_deg"] = calibration.base_orientation.yaw
        result["tilt_offset_deg"] = calibration.base_orientation.pitch
        result["k1"] = calibration.distortion.k1
        result["k2"] = calibration.distortion.k2
        # camera_x and camera_y from position (for comprehensive calibration)
        result["camera_x"] = float(calibration.position.x)
        result["camera_y"] = float(calibration.position.y)

    return result


def _load_gcps_from_registry_file(file_path: Path) -> dict[str, GroundControlPoint]:
    """Load GCPs from a registry YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary mapping GCP name to GroundControlPoint entity.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Registry file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return {}

    map_id = data.get("map_id", file_path.stem)
    points_data = data.get("points", [])

    gcps: dict[str, GroundControlPoint] = {}
    for point_data in points_data:
        name = str(point_data["id"])
        pixel_x = float(point_data["pixel_x"])
        pixel_y = float(point_data["pixel_y"])

        pixel_point = PixelPoint(_x=pixel_x, _y=pixel_y)
        map_point = MapPoint(map_id=map_id, pixel_point=pixel_point)
        gcp = GroundControlPoint(id=name, name=name, map_point=map_point)
        gcps[name] = gcp

    return gcps


@dataclass
class GCPObservationData:
    """GCP observation data for calibration.

    This represents an observation of a GCP in a camera image, including
    the camera pose (pan/tilt/zoom) when the observation was captured.
    """

    map_point_id: str
    pixel_u: PixelsFloat
    pixel_v: PixelsFloat
    pan_raw: Degrees
    tilt_deg: Degrees
    zoom: Unitless


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
        hom calibrate comprehensive --camera Valte --gcps-file valte_gcps.yaml
            --registry-file valte_map_points.yaml
    """
    # Get camera configuration and calibration from repositories
    ctx = ApplicationContext.default()

    all_configs = ctx.repo_camera_config.get_all()
    cam_config = next((c for c in all_configs if c.name == camera), None)

    if not cam_config:
        available = ", ".join(c.name for c in all_configs)
        typer.echo(f"Error: Unknown camera: {camera}. Available: {available}", err=True)
        raise typer.Exit(1)

    # Get calibration data
    cam_calibration = ctx.repo_camera_calibration.get(cam_config.id)

    # Build legacy dict for run_calibration compatibility
    camera_config = _build_legacy_camera_dict(cam_config, cam_calibration)

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

    gcps: list[GCPObservationData] = []
    for gcp_data in data.get("gcps", []):
        gcps.append(
            GCPObservationData(
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
        registry = _load_gcps_from_registry_file(registry_file)
    except FileNotFoundError:
        typer.echo(f"Error: Registry file not found: {registry_file}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: Failed to load registry: {e}", err=True)
        raise typer.Exit(1)

    # Run calibration (GCPObservationData satisfies GCPObservation protocol)
    optimized_params, mean_error, individual_errors = run_calibration(
        camera_config=camera_config,
        gcps=gcps,  # type: ignore[arg-type]
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
    print_results(camera_config, optimized_params, mean_error, individual_errors, gcps)  # type: ignore[arg-type]

    # Exit with code 1 if target accuracy not achieved
    if mean_error >= TARGET_ERROR_THRESHOLD_PX:
        raise typer.Exit(1)
