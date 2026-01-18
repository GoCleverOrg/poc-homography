"""Interactive calibration CLI command."""

from __future__ import annotations

import importlib.util
from pathlib import Path  # noqa: TC003 - Typer needs Path at runtime for CLI arg parsing
from typing import TYPE_CHECKING

import typer

from poc_homography.application import ApplicationContext
from poc_homography.cli.main import app

if TYPE_CHECKING:
    from poc_homography.domain.entities.ground_control_point import GroundControlPoint

# Check for OpenCV availability without importing (avoids unused import warning)
CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


@app.command("interactive")
def interactive_command(
    frame_id: str | None = typer.Option(
        None,
        "--frame-id",
        help="Frame ID from repository (e.g., 'valte/Valte/2024-01-15T10-30-45')",
    ),
    frame_path: Path | None = typer.Option(
        None,
        "--frame-path",
        help="Path to saved camera frame image (PNG, JPG, etc.)",
    ),
    camera: str | None = typer.Option(
        None,
        "--camera",
        help="Camera name (required with --frame-path)",
    ),
    pan_raw: float | None = typer.Option(
        None,
        "--pan-raw",
        help="Raw pan value from camera PTZ (required with --frame-path)",
    ),
    tilt: float | None = typer.Option(
        None,
        "--tilt",
        help="Tilt angle in degrees (required with --frame-path)",
    ),
    zoom: float | None = typer.Option(
        None,
        "--zoom",
        help="Zoom factor (required with --frame-path)",
    ),
    height: float = typer.Option(10.0, help="Initial camera height estimate in meters"),
    pan_offset: float = typer.Option(0.0, help="Initial pan offset estimate in degrees"),
) -> None:
    """
    Interactive calibration tool for map point-to-image projection.

    This tool provides a GUI interface to calibrate camera projection parameters
    by clicking on known reference points in a camera frame image.

    You can provide a frame in two ways:

    1. --frame-id: Load from the captured frames repository (recommended).
       Camera name, PTZ values, and GCPs are loaded automatically from the
       repository based on the frame's map_id.

    2. --frame-path with --camera, --pan-raw, --tilt, --zoom: Load from file
       with manual PTZ values. GCPs are loaded from the repository based on
       the camera's map_id.

    The tool will:
    1. Display the camera frame
    2. Let you click on points you know the Map Point ID of
    3. Enter the Map Point ID for each clicked point
    4. Calculate the optimal parameters (pan_offset, height)
    5. Show you what to update in camera calibration

    Example (from repository - recommended):
        hom interactive --frame-id valte/Valte/2024-01-15T10-30-45

    Example (from file):
        hom interactive --frame-path frame.png --camera Valte
            --pan-raw 45.0 --tilt 30.0 --zoom 5.0
    """
    if not CV2_AVAILABLE:
        typer.echo(
            "Error: OpenCV is required for interactive mode but is not installed",
            err=True,
        )
        typer.echo("Install with: uv pip install opencv-python", err=True)
        raise typer.Exit(1)

    # Validate arguments
    if frame_id is None and frame_path is None:
        typer.echo(
            "Error: Either --frame-id or --frame-path is required",
            err=True,
        )
        raise typer.Exit(1)

    if frame_id is not None and frame_path is not None:
        typer.echo(
            "Error: Cannot use both --frame-id and --frame-path",
            err=True,
        )
        raise typer.Exit(1)

    # Import here to avoid issues when CV2 is not available
    import cv2 as cv2_lib
    import numpy as np

    from poc_homography.calibration.interactive import (
        CalibrationSession,
        run_interactive_session,
    )
    from poc_homography.types import Degrees, Meters, Millimeters, Unitless

    ctx = ApplicationContext.default()

    # Load frame and PTZ state
    if frame_id is not None:
        # Load from CapturedFrame repository
        captured_frame = ctx.repo_captured_frame.get(frame_id)
        if captured_frame is None:
            typer.echo(f"Error: Frame '{frame_id}' not found in repository", err=True)
            raise typer.Exit(1)

        # Get image path and load
        image_path = ctx.repo_captured_frame.get_image_path(captured_frame)
        if not image_path.exists():
            typer.echo(f"Error: Image file not found: {image_path}", err=True)
            raise typer.Exit(1)

        frame_data = cv2_lib.imread(str(image_path))
        if frame_data is None:
            typer.echo(f"Error: Failed to load frame image: {image_path}", err=True)
            raise typer.Exit(1)

        # Use values from captured frame
        camera_name = captured_frame.camera_name
        map_id = captured_frame.map_id
        pan_raw_value = Degrees(captured_frame.ptz_state.pan_raw)
        tilt_value = Degrees(captured_frame.ptz_state.tilt_deg)
        zoom_value = Unitless(captured_frame.ptz_state.zoom)

        typer.echo(f"Loaded frame: {frame_id}")
        typer.echo(f"  Camera: {camera_name}, Map: {map_id}")
        typer.echo(f"  PTZ: pan={pan_raw_value:.1f}, tilt={tilt_value:.1f}, zoom={zoom_value:.1f}")

    else:
        # Load from file path
        assert frame_path is not None

        # Validate required arguments for file mode
        if camera is None:
            typer.echo("Error: --camera is required with --frame-path", err=True)
            raise typer.Exit(1)
        if pan_raw is None:
            typer.echo("Error: --pan-raw is required with --frame-path", err=True)
            raise typer.Exit(1)
        if tilt is None:
            typer.echo("Error: --tilt is required with --frame-path", err=True)
            raise typer.Exit(1)
        if zoom is None:
            typer.echo("Error: --zoom is required with --frame-path", err=True)
            raise typer.Exit(1)

        if not frame_path.exists():
            typer.echo(f"Error: Frame image not found: {frame_path}", err=True)
            raise typer.Exit(1)

        frame_data = cv2_lib.imread(str(frame_path))
        if frame_data is None:
            typer.echo(f"Error: Failed to load frame image: {frame_path}", err=True)
            raise typer.Exit(1)

        # Get camera config to find map_id
        all_configs = ctx.repo_camera_config.get_all()
        cam_config = next((c for c in all_configs if c.name == camera), None)
        if cam_config is None:
            available = [c.name for c in all_configs]
            typer.echo(
                f"Error: Camera '{camera}' not found. Available: {', '.join(available)}",
                err=True,
            )
            raise typer.Exit(1)

        camera_name = camera
        map_id = cam_config.map_id
        pan_raw_value = Degrees(pan_raw)
        tilt_value = Degrees(tilt)
        zoom_value = Unitless(zoom)

    # Ensure the frame is uint8
    frame = np.asarray(frame_data, dtype=np.uint8)

    # Load GCPs from repository based on map_id
    # Cast to proper type (mixin returns Entity, but we know they're GroundControlPoints)
    map_registry: dict[str, GroundControlPoint] = ctx.repo_gcp.get_by_map(map_id)  # type: ignore[assignment]

    if not map_registry:
        typer.echo(
            f"Error: No GCPs found in repository for map '{map_id}'",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(map_registry)} GCPs from repository for map '{map_id}'")

    # Create calibration session
    session = CalibrationSession(
        camera_name=camera_name,
        frame=frame,
        registry=map_registry,
        height_m=Meters(height),
        pan_offset_deg=Degrees(pan_offset),
        pan_raw=pan_raw_value,
        tilt_deg=tilt_value,
        zoom=zoom_value,
        sensor_width_mm=Millimeters(7.18),
    )

    # Run interactive session
    try:
        run_interactive_session(session)
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
