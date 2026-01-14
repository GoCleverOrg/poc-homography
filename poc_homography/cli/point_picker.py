"""CLI command for the GeoTIFF point picker web application."""

from pathlib import Path

import typer
import uvicorn

from poc_homography.cli.main import app

point_picker_app = typer.Typer(help="GeoTIFF point picker commands")
app.add_typer(point_picker_app, name="picker")


@point_picker_app.command("serve")
def serve(
    image_path: Path = typer.Argument(
        ...,
        help="Path to the image file to annotate (PNG, TIFF, etc.)",
        exists=True,
        readable=True,
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
    """Launch the point picker web application.

    Opens a web browser to pick and manage points on an image.
    Points can be tagged as parking_spot (PS), arrows (AR), crosswalk (CW),
    or extra (EX) with auto-incrementing IDs.

    For PNG/JPG images, use --camera to load geotransform from camera config.

    Example:
        hom picker serve path/to/image.png --camera Valte
        hom picker serve image.tif --port 8080
    """
    from poc_homography.point_picker import create_app

    # Resolve to absolute path
    image_path = image_path.resolve()

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

    # Create the FastAPI app
    fastapi_app = create_app(image_path, geotransform=geotransform, crs=crs)

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
