"""Camera intrinsics and validation CLI commands."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path  # noqa: TC003 (runtime type used by typer.Option)
from typing import TYPE_CHECKING

import typer

from poc_homography.calibration import TARGET_ERROR_THRESHOLD_PX
from poc_homography.camera import CameraIntrinsics, PTZStatus, compute_intrinsics
from poc_homography.camera_config import (
    DEFAULT_BASE_FOCAL_LENGTH_MM,
    DEFAULT_SENSOR_WIDTH_MM,
    get_camera_by_name,
    get_camera_configs,
    get_tenant_credentials,
)
from poc_homography.cli.main import camera_app
from poc_homography.domain.protocols.camera_controller import CameraControllerError
from poc_homography.infrastructure.clients.hikvision.isapi_client import HikvisionISAPIClient
from poc_homography.map_points import GCPRegistry
from poc_homography.types import Degrees, Millimeters, Pixels, Unitless
from poc_homography.validation import load_gcps_from_yaml, validate_model

if TYPE_CHECKING:
    from poc_homography.domain.vo.camera_capabilities import CameraCapabilities
    from poc_homography.domain.vo.ptz_state import PTZState


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
        DEFAULT_SENSOR_WIDTH_MM,
        help="Sensor width in millimeters",
    ),
    base_focal_length_mm: float = typer.Option(
        DEFAULT_BASE_FOCAL_LENGTH_MM,
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
    # Get camera configuration
    cam_info = get_camera_by_name(camera)
    if not cam_info:
        available = [c["name"] for c in get_camera_configs()]
        typer.echo(
            f"Error: Camera '{camera}' not found. Available cameras: {', '.join(available)}",
            err=True,
        )
        raise typer.Exit(1)

    # Resolve tenant-specific credentials (falls back to global env vars)
    tenant_id = cam_info.get("tenant_id") or ""
    username, password = get_tenant_credentials(tenant_id)
    if not username or not password:
        typer.echo(
            f"Error: Camera credentials not set for tenant '{tenant_id}'. "
            f"Set {tenant_id.upper()}_CAMERA_USERNAME / {tenant_id.upper()}_CAMERA_PASSWORD "
            "(or global CAMERA_USERNAME / CAMERA_PASSWORD) environment variables.",
            err=True,
        )
        raise typer.Exit(1)

    # Query live PTZ state (and optics/capabilities) via the ISAPI adapter
    client = HikvisionISAPIClient(cam_info["ip"], username, password)
    try:
        ptz_state = client.get_ptz_status()
    except CameraControllerError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Capabilities are best-effort: some firmware does not expose them.
    capabilities: CameraCapabilities | None = None
    try:
        capabilities = client.get_capabilities()
    except CameraControllerError:
        capabilities = None

    # Compute intrinsics from the live zoom level
    intrinsics = compute_intrinsics(
        zoom=ptz_state.zoom,
        image_width=Pixels(image_width),
        image_height=Pixels(image_height),
        sensor_width_mm=Millimeters(sensor_width_mm),
        base_focal_length_mm=Millimeters(base_focal_length_mm),
    )

    ptz_status = _ptz_status_from_state(ptz_state)

    # Format output
    if output_format == OutputFormat.HUMAN:
        output = _format_human_readable(
            camera, cam_info["ip"], ptz_status, intrinsics, ptz_state.focus, capabilities
        )
    elif output_format == OutputFormat.JSON:
        output = _format_json(camera, cam_info["ip"], ptz_status, intrinsics)
    else:  # YAML
        output = _format_yaml(camera, ptz_status, intrinsics)

    typer.echo(output)


def _ptz_status_from_state(state: PTZState) -> PTZStatus:
    """Adapt the domain :class:`PTZState` to the local ``PTZStatus`` view."""
    return PTZStatus(
        pan=Degrees(state.pan_raw),
        tilt=Degrees(state.tilt_deg),
        zoom=Unitless(state.zoom),
    )


def _format_human_readable(
    camera_name: str,
    camera_ip: str,
    ptz: PTZStatus,
    intr: CameraIntrinsics,
    focus: int | None = None,
    capabilities: CameraCapabilities | None = None,
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
    ]
    if focus is not None:
        lines.append(f"  Focus position:   {focus} steps")
    if capabilities is not None:
        lines += [
            "",
            "Camera Capabilities:",
            f"  Pan range:  {capabilities.pan_min:.1f}° .. {capabilities.pan_max:.1f}°",
            f"  Tilt range: {capabilities.tilt_min:.1f}° .. {capabilities.tilt_max:.1f}°",
            f"  Zoom range: {capabilities.zoom_min:.1f}x .. {capabilities.zoom_max:.1f}x",
        ]
    lines += [
        "",
        "Camera Intrinsics:",
        f"  Sensor width:       {intr.sensor_width_mm} mm",
        f"  Base focal length:  {intr.base_focal_length_mm} mm",
        f"  Focal length (mm):  {intr.focal_length_mm:.2f} mm (at {ptz.zoom:.1f}x zoom)",
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
            "focal_length_mm": intr.focal_length_mm,
            "focal_length_px": intr.focal_length_px,
            "principal_point": {"cx": intr.cx, "cy": intr.cy},
            "sensor_width_mm": intr.sensor_width_mm,
            "base_focal_length_mm": intr.base_focal_length_mm,
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
    camera: str = typer.Option(..., help="Camera name (e.g., 'Valte')"),
    gcps_file: Path = typer.Option(..., help="Path to GCPs YAML file"),
    registry_file: Path = typer.Option(..., help="Path to map point registry JSON file"),
) -> None:
    """
    Validate camera model with Ground Control Points.

    Tests projection accuracy with known GCPs to verify that the camera
    geometry model works correctly before running full calibration.

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
        hom camera validate --camera Valte --gcps-file valte_gcps.yaml
            --registry-file valte_map_points.yaml
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
        gcps = load_gcps_from_yaml(gcps_file)
    except FileNotFoundError:
        typer.echo(f"Error: GCPs file not found: {gcps_file}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: Invalid GCPs file: {e}", err=True)
        raise typer.Exit(1)

    if not gcps:
        typer.echo("Error: No GCPs found in YAML file", err=True)
        raise typer.Exit(1)

    # Load map point registry
    try:
        registry = GCPRegistry.load(registry_file)
    except FileNotFoundError:
        typer.echo(f"Error: Registry file not found: {registry_file}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: Failed to load registry: {e}", err=True)
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
