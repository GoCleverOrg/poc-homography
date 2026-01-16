"""CLI command for the camera frame annotator."""

from pathlib import Path

import typer

from poc_homography.cli.main import app

annotate_app = typer.Typer(help="Annotation commands")
app.add_typer(annotate_app, name="annotate")


@annotate_app.command("frame")
def annotate_frame(
    image_path: Path | None = typer.Argument(
        None,
        help="Path to the camera frame image to annotate (optional, uses selector mode if not provided)",
        exists=True,
        readable=True,
    ),
    gcps_file: Path = typer.Option(
        None,
        "--gcps",
        "-g",
        help="Path to GCPs YAML file (default: tests/homography/test_data/Cartografia_valencia_gcps.yaml)",
    ),
    port: int = typer.Option(
        8888,
        "--port",
        "-p",
        help="Port to run the server on",
    ),
) -> None:
    """
    Annotate a camera frame with GCP references.

    Opens a web interface to:
    - Click on the image to mark annotation points
    - Filter and select GCPs from the registry
    - Generate YAML annotations for copy-paste into test cases

    Example:
        hom annotate frame tests/homography/test_data/valte_102.5_20.7_1_20260115_112639.jpg
        hom annotate frame image.jpg --gcps my_gcps.yaml --port 9000
    """
    from poc_homography.tools.camera_annotator import run_annotator

    run_annotator(image_path, gcps_file=gcps_file, port=port)
