"""Camera intrinsics and validation CLI commands."""

from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING, Any

import typer

from poc_homography.application import ApplicationContext
from poc_homography.calibration import TARGET_ERROR_THRESHOLD_PX
from poc_homography.camera import PTZStatus, get_camera_intrinsics
from poc_homography.cli.main import camera_app
from poc_homography.domain import CameraCalibration, CameraConfig, CameraSpec
from poc_homography.types import Degrees, Millimeters, Pixels, Unitless
from poc_homography.validation import GCPData, validate_model

if TYPE_CHECKING:
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint
    from poc_homography.domain.vo.camera_intrinsics import CameraIntrinsics


def _build_legacy_camera_dict(
    config: CameraConfig,
    calibration: CameraCalibration | None,
) -> dict[str, Any]:
    """Build a legacy camera config dict from domain entities.

    This adapter function creates the dict format expected by legacy validation code.
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

    return result


class OutputFormat(str, Enum):
    """Output format options."""

    HUMAN = "human"
    JSON = "json"
    YAML = "yaml"


@camera_app.command("intrinsics")
def intrinsics_command(
    camera: str = typer.Option(..., help="Camera name (e.g., 'Valte')"),
    image_width: int = typer.Option(1920, help="Image width in pixels"),
    image_height: int = typer.Option(1080, help="Image height in pixels"),
    sensor_width_mm: float = typer.Option(
        float(CameraSpec.HIKVISION_DS_2DF8425IX.sensor_width),
        help="Sensor width in millimeters",
    ),
    base_focal_length_mm: float = typer.Option(
        float(CameraSpec.HIKVISION_DS_2DF8425IX.base_focal_length),
        help="Base focal length in millimeters (at 1x zoom)",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.HUMAN,
        "--format",
        "-f",
        help="Output format",
    ),
) -> None:
    """
    Get current camera intrinsics from PTZ status.

    Queries a camera's current PTZ position and computes the intrinsic matrix
    based on the zoom level and sensor parameters.

    Example:
        hom camera intrinsics --camera Valte
        hom camera intrinsics --camera Valte --format json
        hom camera intrinsics --camera Valte --format yaml
    """
    # Get camera configuration from repository
    ctx = ApplicationContext.default()
    all_configs = ctx.repo_camera_config.get_all()
    cam_config = next((c for c in all_configs if c.name == camera), None)

    if not cam_config:
        available = [c.name for c in all_configs]
        typer.echo(
            f"Error: Camera '{camera}' not found. Available cameras: {', '.join(available)}",
            err=True,
        )
        raise typer.Exit(1)

    if not cam_config.ip_address:
        typer.echo(f"Error: Camera '{camera}' has no IP address configured", err=True)
        raise typer.Exit(1)

    # Get PTZ status and intrinsics using credentials from camera config
    try:
        ptz_status, intrinsics = get_camera_intrinsics(
            camera_ip=cam_config.ip_address,
            username=cam_config.credential.username,
            password=cam_config.credential.password,
            image_width=Pixels(image_width),
            image_height=Pixels(image_height),
            sensor_width_mm=Millimeters(sensor_width_mm),
            base_focal_length_mm=Millimeters(base_focal_length_mm),
        )
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Format output
    if output_format == OutputFormat.HUMAN:
        output = _format_human_readable(camera, cam_config.ip_address, ptz_status, intrinsics)
    elif output_format == OutputFormat.JSON:
        output = _format_json(camera, cam_config.ip_address, ptz_status, intrinsics)
    else:  # YAML
        output = _format_yaml(camera, ptz_status, intrinsics)

    typer.echo(output)


def _format_human_readable(
    camera_name: str,
    camera_ip: str,
    ptz: PTZStatus,
    intr: CameraIntrinsics,
) -> str:
    """Format result for human-readable output."""
    K = intr.K

    lines = [
        "=" * 60,
        f"{camera_name.upper()} CAMERA - Current Status & Intrinsics",
        "=" * 60,
        "",
        "PTZ Position:",
        f"  Pan (azimuth):    {ptz.pan:.1f}°",
        f"  Tilt (elevation): {ptz.tilt:.1f}° (Hikvision: positive = down)",
        f"  Zoom factor:      {ptz.zoom:.1f}x",
        "",
        "Camera Intrinsics:",
        f"  Sensor width:       {intr.sensor_width} mm",
        f"  Base focal length:  {intr.base_focal_length} mm",
        f"  Focal length (mm):  {intr.focal_length:.2f} mm (at {ptz.zoom:.1f}x zoom)",
        f"  Focal length (px):  {intr.focal_length_px:.2f} px",
        f"  Principal point:    ({intr.cx:.1f}, {intr.cy:.1f})",
        "",
        "Intrinsic Matrix K:",
        f"  [{K[0, 0]:10.2f}  {K[0, 1]:10.2f}  {K[0, 2]:10.2f}]",
        f"  [{K[1, 0]:10.2f}  {K[1, 1]:10.2f}  {K[1, 2]:10.2f}]",
        f"  [{K[2, 0]:10.2f}  {K[2, 1]:10.2f}  {K[2, 2]:10.2f}]",
        "",
        "For homography_config.yaml camera_capture_context:",
        "  ptz_position:",
        f"    pan: {ptz.pan:.1f}",
        f"    tilt: {ptz.tilt:.1f}",
        f"    zoom: {ptz.zoom:.1f}",
        "  intrinsics:",
        f"    focal_length_px: {intr.focal_length_px:.2f}",
        "    principal_point:",
        f"      cx: {intr.cx:.1f}",
        f"      cy: {intr.cy:.1f}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def _format_json(
    camera_name: str,
    camera_ip: str,
    ptz: PTZStatus,
    intr: CameraIntrinsics,
) -> str:
    """Format result as JSON."""
    output = {
        "camera_name": camera_name,
        "camera_ip": camera_ip,
        "ptz": {
            "pan": ptz.pan,
            "tilt": ptz.tilt,
            "zoom": ptz.zoom,
        },
        "intrinsics": {
            "focal_length_mm": intr.focal_length,
            "focal_length_px": intr.focal_length_px,
            "principal_point": {"cx": intr.cx, "cy": intr.cy},
            "sensor_width_mm": intr.sensor_width,
            "base_focal_length_mm": intr.base_focal_length,
            "image_width": intr.image_width,
            "image_height": intr.image_height,
        },
        "K": intr.K.tolist(),
    }
    return json.dumps(output, indent=2)


def _format_yaml(
    camera_name: str,
    ptz: PTZStatus,
    intr: CameraIntrinsics,
) -> str:
    """Format result as YAML snippet for config file."""
    lines = [
        "# Camera capture context for homography_config.yaml",
        "camera_capture_context:",
        f'  camera_name: "{camera_name}"',
        f"  image_width: {intr.image_width}",
        f"  image_height: {intr.image_height}",
        "  ptz_position:",
        f"    pan: {ptz.pan:.1f}",
        f"    tilt: {ptz.tilt:.1f}",
        f"    zoom: {ptz.zoom:.1f}",
        "  intrinsics:",
        f"    focal_length_px: {intr.focal_length_px:.2f}",
        "    principal_point:",
        f"      cx: {intr.cx:.1f}",
        f"      cy: {intr.cy:.1f}",
    ]
    return "\n".join(lines)


@camera_app.command("validate")
def validate_command(
    frame_id: str = typer.Option(
        ..., help="Frame ID with annotations (e.g., 'valte/Valte/2024-01-15T10-30-45')"
    ),
) -> None:
    """
    Validate camera model with Ground Control Points.

    Tests projection accuracy with known GCPs to verify that the camera
    geometry model works correctly before running full calibration.

    GCP observations (annotations) are loaded from the captured frame repository.
    Use `hom frame annotate <frame_id>` to create annotations interactively first.

    Example:
        hom frame annotate valte/Valte/2024-01-15T10-30-45
        hom camera validate --frame-id valte/Valte/2024-01-15T10-30-45
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

    # Build legacy dict for validate_model compatibility
    camera_config = _build_legacy_camera_dict(cam_config, cam_calibration)

    # Convert annotations to GCPData
    gcps: list[GCPData] = []
    for ann in annotations:
        gcps.append(
            GCPData(
                map_point_id=ann.gcp_id,
                pixel_u=ann.pixel.x,
                pixel_v=ann.pixel.y,
                pan=Degrees(ann.camera_pose.pan_raw),
                tilt=Degrees(ann.camera_pose.tilt_deg),
                zoom=Unitless(ann.camera_pose.zoom),
                name=ann.gcp_id,
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

    # Run validation
    mean_error, _results = validate_model(camera_config, gcps, registry, verbose=True)

    # Exit with code 1 if validation failed or mean error is too high
    if mean_error is None:
        typer.echo("\nError: All GCPs failed validation", err=True)
        raise typer.Exit(1)

    if mean_error >= TARGET_ERROR_THRESHOLD_PX:
        typer.echo(
            f"\nValidation failed: Mean error {mean_error:.2f}px exceeds "
            f"{TARGET_ERROR_THRESHOLD_PX}px threshold",
            err=True,
        )
        raise typer.Exit(1)
