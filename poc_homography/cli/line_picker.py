"""CLI command for the GeoTIFF line picker web application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

from poc_homography.cli.main import app

line_picker_app = typer.Typer(help="GeoTIFF line picker commands")
app.add_typer(line_picker_app, name="line-picker")


@line_picker_app.command("serve")
def serve(
    image_path: Path = typer.Argument(
        ...,
        help="Path to the GeoTIFF map image file",
        exists=True,
        readable=True,
    ),
    map_id: str = typer.Option(
        "valte",
        "--map-id",
        "-m",
        help="Map identifier for tagging saved line files",
    ),
    camera: str | None = typer.Option(
        None,
        "--camera",
        "-c",
        help="Camera name to load geotransform from config (e.g., 'Valte')",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        8001,  # Different default port to avoid conflict with point picker
        "--port",
        "-p",
        help="Port to bind to",
    ),
) -> None:
    """Launch the line picker web application.

    Opens a web browser to create lines on a map using pixel coordinate endpoints.

    Example:
        hom line-picker serve path/to/Cartografia_valencia.tif
        hom line-picker serve map.tif --map-id valte --port 8001
    """
    # Resolve to absolute path
    image_path = image_path.resolve()

    project_root = Path(__file__).parent.parent.parent

    geotiff = None

    # Load geotransform from camera config if specified
    if camera:
        from poc_homography.camera_config import get_camera_by_name

        cam_config = get_camera_by_name(camera)
        if cam_config is None:
            typer.echo(f"Error: Camera '{camera}' not found in config", err=True)
            raise typer.Exit(1)

        geotiff_params = cam_config.get("geotiff_params")
        if geotiff_params:
            gt = geotiff_params.get("geotransform")
            crs = geotiff_params.get("utm_crs")
            if gt and crs:
                from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
                from poc_homography.types import Easting, Meters, Northing, Unitless

                geotiff = GeoTiff(
                    geotransform=GeoTransform(
                        origin_easting=Easting(gt[0]),
                        pixel_width=Meters(gt[1]),
                        row_rotation=Unitless(gt[2]),
                        origin_northing=Northing(gt[3]),
                        col_rotation=Unitless(gt[4]),
                        pixel_height=Meters(gt[5]),
                    ),
                    crs=crs,
                )
                typer.echo(f"Loaded geotransform from camera '{camera}': {gt}")
                typer.echo(f"CRS: {crs}")
        else:
            typer.echo(f"Warning: Camera '{camera}' has no geotiff_params", err=True)

    typer.echo(f"Loading image: {image_path}")

    # Find the webapp directory
    webapp_dir = project_root / "webapp"

    if not webapp_dir.exists():
        typer.echo(f"Error: webapp directory not found at {webapp_dir}", err=True)
        raise typer.Exit(1)

    # Add webapp to sys.path so Django can find the apps
    if str(webapp_dir) not in sys.path:
        sys.path.insert(0, str(webapp_dir))

    # Set Django settings module
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "homography_web.settings")

    # Initialize Django
    import django

    django.setup()

    # Initialize the line picker state
    from line_picker.state import initialize_state

    initialize_state(
        image_path,
        map_id=map_id,
        geotiff=geotiff,
    )

    typer.echo(f"Starting server at http://{host}:{port}/line-picker/")
    typer.echo("Press Ctrl+C to stop")

    # Open browser after a short delay
    import threading
    import webbrowser

    def open_browser() -> None:
        import time

        time.sleep(1)
        webbrowser.open(f"http://{host}:{port}/line-picker/")

    threading.Thread(target=open_browser, daemon=True).start()

    # Run Django development server
    from django.core.management import execute_from_command_line

    # Change to webapp directory for Django to find templates/static
    original_cwd = os.getcwd()
    os.chdir(webapp_dir)

    try:
        execute_from_command_line(["manage.py", "runserver", f"{host}:{port}", "--noreload"])
    finally:
        os.chdir(original_cwd)
