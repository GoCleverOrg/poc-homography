"""CLI command for the GeoTIFF point picker web application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

from poc_homography.cli.main import app

point_picker_app = typer.Typer(help="GeoTIFF point picker commands")
app.add_typer(point_picker_app, name="picker")


@point_picker_app.command("serve")
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind to",
    ),
) -> None:
    """Launch the point picker web application.

    Opens a web browser to pick and manage points on an image.
    Points can be tagged as parking_spot (PS), arrows (AR), crosswalk (CW),
    or extra (EX) with auto-incrementing IDs.

    State is lazily initialized per-tenant when requests arrive.

    Example:
        hom picker serve
        hom picker serve --port 8080
    """
    # Find the webapp directory
    # Path: cli/point_picker.py -> cli/ -> poc_homography/ -> project_root/ -> webapp/
    project_root = Path(__file__).parent.parent.parent
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

    typer.echo(f"Starting server at http://{host}:{port}/point-picker/")
    typer.echo("Press Ctrl+C to stop")

    # Open browser after a short delay
    import threading
    import webbrowser

    def open_browser() -> None:
        import time

        time.sleep(1)
        webbrowser.open(f"http://{host}:{port}/point-picker/")

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
