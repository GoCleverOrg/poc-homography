"""Per-camera calibration-confidence report CLI command (#366).

A thin, read-only CLI over the pure aggregation in
:mod:`poc_homography.calibration.confidence`. For one camera (``--camera``) or
every camera of a tenant (``--tenant``), it reads the already-persisted intrinsic
lens table and extrinsic registration records, JOINS them into one per-camera
:class:`~poc_homography.calibration.confidence.CameraConfidenceReport`, and emits
a human-readable or JSON report. It recomputes NO calibration.

The data-access seam (:func:`_run_confidence_report`) references both
repositories and the session factory as module-level names so tests can swap them
for offline doubles and exercise the intrinsic⋈extrinsic join with NO live Neon
access. The pure join itself lives in :func:`_collect_reports`.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import TYPE_CHECKING

import typer

from poc_homography.calibration.confidence import build_camera_confidence
from poc_homography.camera_config import (
    get_camera_by_name,
    get_camera_configs,
    get_cameras_for_tenant,
)
from poc_homography.cli.main import calibrate_app
from poc_homography.infrastructure.database import get_session
from poc_homography.infrastructure.repositories.repo_postgres_lens_calibration_table import (
    RepoPostgresLensCalibrationTable,
)
from poc_homography.infrastructure.repositories.repo_postgres_ptz_registration import (
    RepoPostgresPtzRegistration,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session

    from poc_homography.calibration.confidence import CameraConfidenceReport


class OutputFormat(str, Enum):
    """Output format options."""

    HUMAN = "human"
    JSON = "json"


def _collect_reports(
    camera_specs: Sequence[tuple[str, str]],
    *,
    lens_repo: RepoPostgresLensCalibrationTable,
    ptz_repo: RepoPostgresPtzRegistration,
) -> list[CameraConfidenceReport]:
    """Build one confidence report per ``(camera_id, camera_name)`` spec.

    This is the read-only join: for each camera it reads the persisted lens
    table and registration records and delegates to
    :func:`build_camera_confidence`. It is the primary offline test target.

    Args:
        camera_specs: ``(camera_id, camera_name)`` pairs to report on.
        lens_repo: Intrinsic lens-table repository (its ``id`` is the camera id).
        ptz_repo: Extrinsic PTZ-registration repository.

    Returns:
        One :class:`CameraConfidenceReport` per spec, in input order.
    """
    reports: list[CameraConfidenceReport] = []
    for camera_id, camera_name in camera_specs:
        lens_table = lens_repo.get(camera_id)
        registrations = ptz_repo.get_by_camera(camera_id)
        reports.append(
            build_camera_confidence(
                camera_id=camera_id,
                camera_name=camera_name,
                lens_table=lens_table,
                registrations=registrations,
            )
        )
    return reports


def _run_confidence_report(
    *,
    camera_specs: Sequence[tuple[str, str]],
    session_factory: Callable[[], AbstractContextManager[Session]],
) -> list[CameraConfidenceReport]:
    """Open a session, construct both repositories, and collect the reports.

    Extracted from the command body so tests can inject a fake session factory
    and monkeypatch the module-level repository classes for fully-offline runs.

    Args:
        camera_specs: ``(camera_id, camera_name)`` pairs to report on.
        session_factory: Database session context-manager factory.

    Returns:
        One :class:`CameraConfidenceReport` per spec.
    """
    with session_factory() as session:
        lens_repo = RepoPostgresLensCalibrationTable(session)
        ptz_repo = RepoPostgresPtzRegistration(session)
        return _collect_reports(camera_specs, lens_repo=lens_repo, ptz_repo=ptz_repo)


def _report_to_dict(report: CameraConfidenceReport) -> dict:
    """Serialize a :class:`CameraConfidenceReport` to a JSON-friendly dict."""
    intrinsic = (
        {
            "num_zoom_entries": report.intrinsic.num_zoom_entries,
            "worst_rmse_px": report.intrinsic.worst_rmse_px,
            "mean_rmse_px": report.intrinsic.mean_rmse_px,
            "per_zoom": [
                {
                    "zoom_factor": point.zoom_factor,
                    "validation_rmse": point.validation_rmse,
                    "reprojection_error_px": point.reprojection_error_px,
                }
                for point in report.intrinsic.per_zoom
            ],
        }
        if report.intrinsic is not None
        else None
    )
    extrinsic = (
        {
            "num_registrations": report.extrinsic.num_registrations,
            "num_validated": report.extrinsic.num_validated,
            "worst_rms_m": report.extrinsic.worst_rms_m,
            "worst_p90_m": report.extrinsic.worst_p90_m,
            "worst_max_m": report.extrinsic.worst_max_m,
            "mean_rms_m": report.extrinsic.mean_rms_m,
            "min_inlier_ratio": report.extrinsic.min_inlier_ratio,
        }
        if report.extrinsic is not None
        else None
    )
    return {
        "camera_id": report.camera_id,
        "camera_name": report.camera_name,
        "calibrated": report.calibrated,
        "label": report.label,
        "score": report.score,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "notes": list(report.notes),
    }


def _format_json(reports: Sequence[CameraConfidenceReport], *, tenant: str | None) -> str:
    """Render the reports as JSON (single dict for ``--camera``, else tenant block)."""
    if tenant is None:
        return json.dumps(_report_to_dict(reports[0]), indent=2)
    return json.dumps(
        {"tenant": tenant, "cameras": [_report_to_dict(r) for r in reports]},
        indent=2,
    )


def _format_one_human(report: CameraConfidenceReport) -> list[str]:
    """Render a single camera's confidence block as a list of text lines."""
    status = "CALIBRATED" if report.calibrated else "UNCALIBRATED"
    lines = [
        "=" * 60,
        f"{report.camera_name} ({report.camera_id})",
        "=" * 60,
        f"Status:     {status}",
        f"Confidence: {report.label}  (score {report.score:.2f})",
        "",
    ]

    lines.append("Intrinsic calibration:")
    if report.intrinsic is not None:
        intr = report.intrinsic
        lines += [
            f"  Zoom entries:  {intr.num_zoom_entries}",
            f"  Worst RMSE:    {intr.worst_rmse_px:.2f} px",
            f"  Mean RMSE:     {intr.mean_rmse_px:.2f} px",
            "  Per-zoom error (max of validation_rmse, reprojection_error_px):",
        ]
        for point in intr.per_zoom:
            error = max(point.validation_rmse, point.reprojection_error_px)
            lines.append(f"    zoom {point.zoom_factor:.1f}x: {error:.2f} px")
    else:
        lines.append("  (none persisted)")
    lines.append("")

    lines.append("Extrinsic registration:")
    if report.extrinsic is not None:
        extr = report.extrinsic
        lines += [
            f"  Registrations: {extr.num_registrations} ({extr.num_validated} validated)",
            f"  Worst RMS:     {extr.worst_rms_m:.3f} m",
            f"  Worst p90:     {extr.worst_p90_m:.3f} m",
            f"  Worst max:     {extr.worst_max_m:.3f} m",
            f"  Mean RMS:      {extr.mean_rms_m:.3f} m",
            f"  Min inliers:   {extr.min_inlier_ratio:.2f}",
        ]
    else:
        lines.append("  (none persisted)")
    lines.append("")

    lines.append("Notes:")
    if report.notes:
        lines.extend(f"  - {note}" for note in report.notes)
    else:
        lines.append("  (none)")
    lines.append("=" * 60)
    return lines


def _format_human(reports: Sequence[CameraConfidenceReport]) -> str:
    """Render the reports as a human-readable block per camera."""
    blocks: list[str] = []
    for report in reports:
        blocks.append("\n".join(_format_one_human(report)))
    return "\n\n".join(blocks)


@calibrate_app.command("confidence-report")
def confidence_report_command(
    camera: str | None = typer.Option(
        None, "--camera", help="Camera name or id (mutually exclusive with --tenant)"
    ),
    tenant: str | None = typer.Option(
        None, "--tenant", help="Tenant id; reports every camera (e.g., 'icozee')"
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.HUMAN,
        "--format",
        "-f",
        help="Output format",
    ),
) -> None:
    """Report per-camera calibration confidence from persisted records only.

    Joins each camera's persisted intrinsic lens table with its persisted
    extrinsic registration records into one Good/Fair/Poor confidence record;
    cameras missing either half are reported as ``Uncalibrated``. Recomputes no
    calibration. Provide EXACTLY ONE of ``--camera`` / ``--tenant``.
    """
    if bool(camera) == bool(tenant):
        typer.echo("Error: provide exactly one of --camera or --tenant.", err=True)
        raise typer.Exit(1)

    if not os.environ.get("DATABASE_URL"):
        typer.echo("Error: DATABASE_URL environment variable is not set.", err=True)
        raise typer.Exit(1)

    if camera:
        cam_info = get_camera_by_name(camera)
        if not cam_info:
            available = [c["name"] for c in get_camera_configs()]
            typer.echo(
                f"Error: Camera '{camera}' not found. Available cameras: {', '.join(available)}",
                err=True,
            )
            raise typer.Exit(1)
        camera_specs = [(cam_info["id"], cam_info.get("name") or cam_info["id"])]
        tenant_id = None
    else:
        cams = get_cameras_for_tenant(tenant or "")
        if not cams:
            typer.echo(f"Error: No cameras found for tenant '{tenant}'.", err=True)
            raise typer.Exit(1)
        camera_specs = [(c["id"], c.get("name") or c["id"]) for c in cams]
        tenant_id = tenant

    reports = _run_confidence_report(camera_specs=camera_specs, session_factory=get_session)

    if output_format == OutputFormat.JSON:
        typer.echo(_format_json(reports, tenant=tenant_id))
    else:
        typer.echo(_format_human(reports))


__all__ = ["confidence_report_command"]
