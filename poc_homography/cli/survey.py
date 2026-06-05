"""Multi-camera survey operator CLI commands (C5, issue #262)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer
import yaml

from poc_homography.cli.main import survey_app
from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig
from poc_homography.survey.run_service import build_survey_run_service

# Module-level service accessor; tests monkeypatch this attribute.
_survey_run_service = build_survey_run_service()


@survey_app.command("run")
def run_command(
    plan: Path = typer.Option(..., help="Path to a YAML survey plan config file"),
    cameras: str = typer.Option(..., help="Comma-separated camera ids"),
) -> None:
    """
    Launch a multi-camera survey run from a plan config and stream progress.

    Reads the YAML plan, builds a SurveyPlanConfig, starts a run for the given
    cameras, and prints per-camera session ids followed by streamed progress.
    """
    try:
        with Path(plan).open() as f:
            data = yaml.safe_load(f)
        cfg = SurveyPlanConfig.from_dict(data)
    except (OSError, yaml.YAMLError, ValueError, TypeError) as e:
        typer.echo(f"Error: Failed to load plan config: {e}", err=True)
        raise typer.Exit(1)

    camera_ids = [c.strip() for c in cameras.split(",") if c.strip()]
    if not camera_ids:
        typer.echo("Error: No camera ids provided", err=True)
        raise typer.Exit(1)

    try:
        result = _survey_run_service.start_run(cfg, camera_ids)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    run_id = str(result["run_id"])
    session_ids = cast("dict[str, str]", result["session_ids"])
    typer.echo(f"run_id: {run_id}")
    for camera_id, session_id in session_ids.items():
        typer.echo(f"  {camera_id}: {session_id}")

    for event in _survey_run_service.iter_progress(run_id):
        typer.echo(
            f"[{event['camera_id']}] phase={event['phase']} "
            f"frames={event['frame_count']} status={event['status']}"
        )


@survey_app.command("status")
def status_command(
    run_id: str = typer.Option(..., help="Run id to query"),
) -> None:
    """Print per-camera status for a survey run."""
    status = _survey_run_service.get_status(run_id)
    if status is None:
        typer.echo(f"Error: Unknown run_id: {run_id}", err=True)
        raise typer.Exit(1)

    typer.echo(f"run_id: {status['run_id']}")
    cameras = cast("dict[str, dict[str, object]]", status["cameras"])
    for camera_id, cam in cameras.items():
        typer.echo(
            f"  {camera_id}: phase={cam['phase']} "
            f"frames={cam['frame_count']} status={cam['status']}"
        )


@survey_app.command("abort")
def abort_command(
    run_id: str = typer.Option(..., help="Run id to abort"),
) -> None:
    """Request a graceful abort of a survey run."""
    result = _survey_run_service.abort_run(run_id)
    if result is None:
        typer.echo(f"Error: Unknown run_id: {run_id}", err=True)
        raise typer.Exit(1)

    typer.echo(result["message"])


@survey_app.command("list")
def list_command(
    limit: int = typer.Option(20, help="Maximum number of runs to list"),
) -> None:
    """List recent survey runs as a table."""
    runs = _survey_run_service.list_runs(limit)
    typer.echo("run_id\tstatus\tstart_time\tcamera_count\ttotal_frame_count")
    for run in runs:
        typer.echo(
            f"{run['run_id']}\t{run['status']}\t{run['start_time']}\t"
            f"{run['camera_count']}\t{run['total_frame_count']}"
        )


@survey_app.command("browse")
def browse_command(
    run_id: str = typer.Option(..., help="Run id to browse"),
    phase: int | None = typer.Option(None, help="Filter by phase number (1..9)"),
    camera: str | None = typer.Option(None, help="Filter by camera id"),
    zoom: float | None = typer.Option(None, help="Filter by reported zoom factor"),
) -> None:
    """Browse dataset groupings (phase, camera, zoom) with frame counts."""
    try:
        groups = _survey_run_service.browse_groups(run_id, phase=phase, camera=camera, zoom=zoom)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not groups:
        typer.echo("No groupings found")
        return

    typer.echo("phase\tcamera\tzoom\tframe_count")
    for group in groups:
        typer.echo(f"{group['phase']}\t{group['camera']}\t{group['zoom']}\t{group['frame_count']}")
