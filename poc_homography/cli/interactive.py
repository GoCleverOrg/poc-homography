"""Interactive calibration CLI command."""

import importlib.util
from pathlib import Path

import typer

from poc_homography.cli.main import app

# Check for OpenCV availability without importing (avoids unused import warning)
CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


@app.command("interactive")
def interactive_command(
    camera: str = typer.Option(..., help="Camera name (e.g., 'Valte')"),
    frame_path: Path = typer.Option(..., help="Path to saved camera frame image (PNG, JPG, etc.)"),
    registry: Path = typer.Option(..., help="Path to map point registry JSON file"),
    pan_raw: float = typer.Option(..., help="Raw pan value from camera PTZ (degrees)"),
    tilt: float = typer.Option(..., help="Tilt angle in degrees"),
    zoom: float = typer.Option(..., help="Zoom factor"),
    height: float = typer.Option(10.0, help="Initial camera height estimate in meters"),
    pan_offset: float = typer.Option(0.0, help="Initial pan offset estimate in degrees"),
) -> None:
    """
    Interactive calibration tool for map point-to-image projection.

    This tool provides a GUI interface to calibrate camera projection parameters
    by clicking on known reference points in a camera frame image.

    The tool will:
    1. Display the camera frame
    2. Let you click on points you know the Map Point ID of
    3. Enter the Map Point ID for each clicked point
    4. Calculate the optimal parameters (pan_offset, height)
    5. Show you what to update in camera_config.py

    Example:
        hom interactive --camera Valte --frame-path frame.png
            --registry map_points.json --pan-raw 45.0 --tilt 30.0 --zoom 5.0
    """
    if not CV2_AVAILABLE:
        typer.echo("Error: OpenCV is required for interactive mode but is not installed", err=True)
        typer.echo("Install with: uv pip install opencv-python", err=True)
        raise typer.Exit(1)

    # Import here to avoid issues when CV2 is not available
    import cv2 as cv2_lib
    import numpy as np

    from poc_homography.calibration.interactive import CalibrationSession, run_interactive_session
    from poc_homography.map_points import MapPointRegistry
    from poc_homography.types import Degrees, Meters, Millimeters, Unitless

    # Load frame image
    if not frame_path.exists():
        typer.echo(f"Error: Frame image not found: {frame_path}", err=True)
        raise typer.Exit(1)

    frame_data = cv2_lib.imread(str(frame_path))
    if frame_data is None:
        typer.echo(f"Error: Failed to load frame image: {frame_path}", err=True)
        raise typer.Exit(1)

    # Ensure the frame is uint8
    frame = np.asarray(frame_data, dtype=np.uint8)

    # Load map point registry
    try:
        map_registry = MapPointRegistry.load(registry)
    except FileNotFoundError:
        typer.echo(f"Error: Registry file not found: {registry}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: Failed to load registry: {e}", err=True)
        raise typer.Exit(1)

    # Create calibration session
    session = CalibrationSession(
        camera_name=camera,
        frame=frame,
        registry=map_registry,
        height_m=Meters(height),
        pan_offset_deg=Degrees(pan_offset),
        pan_raw=Degrees(pan_raw),
        tilt_deg=Degrees(tilt),
        zoom=Unitless(zoom),
        sensor_width_mm=Millimeters(7.18),
    )

    # Run interactive session
    try:
        run_interactive_session(session)
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
