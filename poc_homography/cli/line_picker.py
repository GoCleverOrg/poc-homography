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
    gcps_dir: Path | None = typer.Option(
        None,
        "--gcps-dir",
        "-g",
        help="Directory containing per-GCP YAML files (defaults to data/gcps/)",
    ),
    map_id: str | None = typer.Option(
        None,
        "--map-id",
        "-m",
        help="Map identifier to load GCPs for (defaults to first available)",
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
    GCPs from the repository are displayed as clickable markers on the map.

    Example:
        hom line-picker serve path/to/Cartografia_valencia.tif
        hom line-picker serve map.tif --gcps-dir data/gcps --map-id valte --port 8001
    """
    from poc_homography.map_points.gcp_registry import list_map_ids

    # Resolve to absolute path
    image_path = image_path.resolve()

    # Determine GCPs directory
    project_root = Path(__file__).parent.parent.parent
    if gcps_dir is None:
        gcps_dir = project_root / "data" / "gcps"
    else:
        gcps_dir = gcps_dir.resolve()

    if not gcps_dir.exists():
        typer.echo(f"Error: GCPs directory not found at {gcps_dir}", err=True)
        typer.echo("Please specify with --gcps-dir option", err=True)
        raise typer.Exit(1)

    # Determine map_id
    available = list_map_ids(gcps_dir)
    if not available:
        typer.echo(f"Error: No GCPs found in {gcps_dir}", err=True)
        raise typer.Exit(1)

    if map_id is None:
        map_id = available[0]
    elif map_id not in available:
        typer.echo(f"Error: Map '{map_id}' not found. Available: {available}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading GCPs for map '{map_id}' from: {gcps_dir}")

    geotransform = None
    crs = None

    # Load geotransform from camera config if specified
    if camera:
        from poc_homography.camera_config import get_camera_by_name

        cam_config = get_camera_by_name(camera)
        if cam_config is None:
            typer.echo(f"Error: Camera '{camera}' not found in config", err=True)
            raise typer.Exit(1)

        geotiff_params = cam_config.get("geotiff_params")
        if geotiff_params:
            geotransform = geotiff_params.get("geotransform")
            crs = geotiff_params.get("utm_crs")
            typer.echo(f"Loaded geotransform from camera '{camera}': {geotransform}")
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
        gcps_dir=gcps_dir,
        map_id=map_id,
        geotransform=geotransform,
        crs=crs,
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
