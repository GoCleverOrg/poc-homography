"""Calibration CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import typer

from poc_homography.application import ApplicationContext
from poc_homography.calibration import (
    TARGET_ERROR_THRESHOLD_PX,
    print_results,
    run_calibration,
)
from poc_homography.cli.main import calibrate_app
from poc_homography.types import Degrees, Pixels, PixelsFloat, Unitless

if TYPE_CHECKING:
    from poc_homography.domain import CameraCalibration, CameraConfig
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint


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
    frame_id: str = typer.Option(
        ..., help="Frame ID with annotations (e.g., 'valte/Valte/2024-01-15T10-30-45')"
    ),
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

    GCP observations (annotations) are loaded from the captured frame repository.
    Use `hom frame annotate <frame_id>` to create annotations interactively first.

    Example:
        hom frame annotate valte/Valte/2024-01-15T10-30-45
        hom calibrate comprehensive --frame-id valte/Valte/2024-01-15T10-30-45
    """
    ctx = ApplicationContext.default()

    # Load captured frame
    captured_frame = ctx.repo_captured_frame.get(frame_id)
    if not captured_frame:
        typer.echo(f"Error: Frame '{frame_id}' not found", err=True)
        raise typer.Exit(1)

    # Load annotations for the frame
    annotations = ctx.repo_captured_frame.get_annotations(frame_id)
    if not annotations:
        typer.echo(f"Error: No annotations found for frame '{frame_id}'", err=True)
        typer.echo("Use 'hom frame annotate <frame_id>' to create annotations first", err=True)
        raise typer.Exit(1)

    # Get camera configuration and calibration from repositories
    camera_name = captured_frame.camera_name
    map_id = captured_frame.map_id

    all_configs = ctx.repo_camera_config.get_all()
    cam_config = next((c for c in all_configs if c.name == camera_name), None)

    if not cam_config:
        available = ", ".join(c.name for c in all_configs)
        typer.echo(f"Error: Unknown camera: {camera_name}. Available: {available}", err=True)
        raise typer.Exit(1)

    # Get calibration data
    cam_calibration = ctx.repo_camera_calibration.get(cam_config.id)

    # Build legacy dict for run_calibration compatibility
    camera_config = _build_legacy_camera_dict(cam_config, cam_calibration)

    # Convert annotations to GCPObservationData
    gcps: list[GCPObservationData] = []
    for ann in annotations:
        gcps.append(
            GCPObservationData(
                map_point_id=ann.gcp_id,
                pixel_u=ann.pixel.x,
                pixel_v=ann.pixel.y,
                pan_raw=Degrees(ann.camera_pose.pan_raw),
                tilt_deg=Degrees(ann.camera_pose.tilt_deg),
                zoom=Unitless(ann.camera_pose.zoom),
            )
        )

    typer.echo(f"Loaded {len(gcps)} annotations from frame '{frame_id}'")

    # Load GCPs from repository based on frame's map_id
    registry: dict[str, GroundControlPoint] = ctx.repo_gcp.get_by_map(map_id)  # type: ignore[assignment]

    if not registry:
        typer.echo(
            f"Error: No GCPs found in repository for map '{map_id}'",
            err=True,
        )
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
