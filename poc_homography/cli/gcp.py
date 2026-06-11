"""Ground Control Point (GCP) analysis CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from poc_homography.cli.main import gcp_app
from poc_homography.validation.gcp_distribution import (
    calculate_distribution,
    load_image_points,
    render_full_text_report,
    render_html_report,
)


@gcp_app.command("distribution")
def distribution_command(
    gcps: Path = typer.Option(..., "--gcps", "-g", help="Path to GCP YAML file"),
    width: int | None = typer.Option(
        None, "--width", "-w", help="Image width in pixels (overrides YAML metadata)"
    ),
    # No -h short flag for --height to avoid ambiguity with the conventional -h/--help.
    height: int | None = typer.Option(
        None, "--height", help="Image height in pixels (overrides YAML metadata)"
    ),
    html: bool = typer.Option(False, "--html", help="Generate HTML visualization instead of text"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (default: stdout for text, gcp_distribution.html for HTML)",
    ),
) -> None:
    """Analyze the spatial distribution of GCPs in image space.

    Reads an existing GCP YAML file and reports how well the ground control
    points are distributed across the image plane (image_u/image_v). Requires
    no camera connection and no frame image.

    Examples:
        hom gcp distribution --gcps gcps.yaml
        hom gcp distribution -g gcps.yaml --width 1920 --height 1080
        hom gcp distribution -g gcps.yaml --html -o report.html
    """
    try:
        points, yaml_width, yaml_height = load_image_points(gcps)
    except FileNotFoundError:
        typer.echo(f"Error: File not found: {gcps}", err=True)
        raise typer.Exit(1) from None
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    # Resolve dimensions: CLI overrides YAML metadata.
    resolved_width = width if width is not None else yaml_width
    resolved_height = height if height is not None else yaml_height

    if resolved_width is None or resolved_height is None:
        typer.echo(
            "Error: Image dimensions required. Provide --width and --height or "
            "use a YAML with feature_match image dimensions.",
            err=True,
        )
        raise typer.Exit(1)

    metrics = calculate_distribution(points, resolved_width, resolved_height)

    if html:
        out_path = output if output is not None else Path("gcp_distribution.html")
        content = render_html_report(gcps.name, metrics, points)
        out_path.write_text(content, encoding="utf-8")
        typer.echo(f"HTML report written to {out_path}")
        return

    report = render_full_text_report(gcps.name, metrics, points)
    if output is not None:
        output.write_text(report, encoding="utf-8")
        typer.echo(f"Report written to {output}")
    else:
        typer.echo(report)
