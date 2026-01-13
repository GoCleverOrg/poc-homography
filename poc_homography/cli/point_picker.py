"""CLI command for the GeoTIFF point picker web application."""

from pathlib import Path

import typer
import uvicorn

from poc_homography.cli.main import app

point_picker_app = typer.Typer(help="GeoTIFF point picker commands")
app.add_typer(point_picker_app, name="picker")


@point_picker_app.command("serve")
def serve(
    geotiff_path: Path = typer.Argument(
        ...,
        help="Path to the GeoTIFF file to annotate",
        exists=True,
        readable=True,
    ),
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
    reload: bool = typer.Option(
        False,
        "--reload",
        "-r",
        help="Enable auto-reload for development",
    ),
) -> None:
    """Launch the GeoTIFF point picker web application.

    Opens a web browser to pick and manage points on a GeoTIFF image.
    Points can be tagged as parking_spot (PS), arrows (AR), crosswalk (CW),
    or extra (EX) with auto-incrementing IDs.

    Example:
        hom picker serve path/to/image.tif
        hom picker serve image.tif --port 8080
    """
    from poc_homography.point_picker import create_app

    # Resolve to absolute path
    geotiff_path = geotiff_path.resolve()

    typer.echo(f"Loading GeoTIFF: {geotiff_path}")

    # Create the FastAPI app
    fastapi_app = create_app(geotiff_path)

    typer.echo(f"Starting server at http://{host}:{port}")
    typer.echo("Press Ctrl+C to stop")

    # Open browser after a short delay
    import threading
    import webbrowser

    def open_browser() -> None:
        import time

        time.sleep(1)
        webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    # Run the server
    uvicorn.run(fastapi_app, host=host, port=port, reload=reload)
