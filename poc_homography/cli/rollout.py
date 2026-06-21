"""Multi-camera calibration rollout driver CLI command (#367).

A thin loop over the already-merged leaves -- it adds **no** new calibration,
solver, or capture logic. For one tenant it:

1. Loads a ``calib_cameras.json`` (default ``/tmp/calib_cameras.json``) describing
   the ``available`` and ``excluded`` cameras for a probed site.
2. Resolves each available JSON camera ``name`` (e.g. ``"cam02"``) to its real
   camera id (``icozee-camptz-02``) via :func:`get_cameras_for_tenant`, matching
   the zero-padded ``Cam{NN}`` name field.
3. Skips any camera in the ``excluded`` set (defensively, even though
   ``available`` already omits them).
4. Invokes leaf-2's per-camera runner (:func:`_run_camera_registration`) for each
   camera and collects leaf-3's per-camera confidence
   (:class:`CameraConfidenceReport`).
5. Prints a per-camera pass/fail + confidence rollout summary.
6. Continues on error: one bad camera (e.g. :class:`HoldoutValidationError`) is
   recorded and the loop continues; the process exits non-zero if ANY camera
   failed.

The pure mapping/exclusion helper (:func:`_resolve_targets`) is the primary
offline test target. The orchestration seam (:func:`_run_rollout`) references the
per-camera runner, the confidence collector, the ortho loader, the frame store,
and the session factory as module-level names so tests can swap them for doubles
and exercise the whole loop with NO live MinIO, Neon, camera, or network access.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path  # runtime import: Typer resolves the --cameras-file annotation
from typing import TYPE_CHECKING

import typer

from poc_homography.calibration.extrinsic import (
    HoldoutValidationError,
    InsufficientCorrespondenceError,
)
from poc_homography.camera_config import get_cameras_for_tenant
from poc_homography.cli.confidence_report import _collect_reports
from poc_homography.cli.main import calibrate_app
from poc_homography.cli.run_camera import _load_ortho_context, _run_camera_registration
from poc_homography.infrastructure.clients.minio_frame_store import MinioFrameStore
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
    from poc_homography.calibration.lens_distortion.frame_source import FrameStore

_DEFAULT_CAMERAS_FILE = Path("/tmp/calib_cameras.json")  # documented contract path
_NAME_DIGITS = re.compile(r"(\d+)\s*$")


@dataclass(frozen=True)
class RolloutTarget:
    """A resolved camera to run in the rollout.

    Attributes:
        json_name: The lowercase ``camNN`` name from ``calib_cameras.json``.
        camera_id: The resolved real camera id (e.g. ``icozee-camptz-02``).
        camera_name: The tenant config name (e.g. ``Cam02``).
        map_id: The orthophoto map id for the camera, or ``""`` when absent.
    """

    json_name: str
    camera_id: str
    camera_name: str
    map_id: str


@dataclass(frozen=True)
class RolloutAnnotation:
    """A camera that was not run (excluded by name, or unresolved).

    Attributes:
        json_name: The ``calib_cameras.json`` name the annotation is about.
        skipped: ``True`` for an excluded camera (benign); ``False`` for an
            unresolved name (counts as a failure).
        reason: Human-readable explanation for the summary.
    """

    json_name: str
    skipped: bool
    reason: str


@dataclass
class CameraOutcome:
    """The result of running (or not running) one camera in the rollout.

    Attributes:
        json_name: The ``calib_cameras.json`` name.
        camera_id: The resolved camera id, or ``None`` when unresolved/skipped.
        status: One of ``"PASS"``, ``"FAIL"``, or ``"SKIP"``.
        detail: Confidence label/score, skip reason, or failure message.
    """

    json_name: str
    camera_id: str | None
    status: str
    detail: str


def _normalize_name(json_name: str) -> str | None:
    """Map a ``calib_cameras.json`` name to the ``Cam{NN}`` config name.

    Args:
        json_name: The lowercase, zero-padded name (e.g. ``"cam02"``).

    Returns:
        The capitalized two-digit name (e.g. ``"Cam02"``), or ``None`` when the
        name carries no trailing digits.
    """
    match = _NAME_DIGITS.search(json_name)
    if match is None:
        return None
    return f"Cam{int(match.group(1)):02d}"


def _resolve_targets(
    spec: dict, tenant_cameras: Sequence[dict]
) -> tuple[list[RolloutTarget], list[RolloutAnnotation]]:
    """Resolve + filter the JSON camera list against the tenant camera list.

    Pure function (no I/O): maps each ``available`` JSON name to a tenant camera
    id by its zero-padded ``Cam{NN}`` name, skips any name appearing in the
    ``excluded`` set, and records an annotation for excluded and unresolved
    cameras. This is the primary offline test target.

    Args:
        spec: The parsed ``calib_cameras.json`` dict (``available`` + ``excluded``).
        tenant_cameras: Tenant camera dicts (each with ``id`` and ``name``).

    Returns:
        A ``(targets, annotations)`` pair: the cameras to run and the
        excluded/unresolved annotations (excluded ones have ``skipped=True``).
    """
    by_name = {str(cam.get("name", "")): cam for cam in tenant_cameras}
    excluded_names = {
        str(entry.get("name", "")).lower() for entry in spec.get("excluded", []) if entry
    }
    excluded_reasons = {
        str(entry.get("name", "")).lower(): str(entry.get("reason", "excluded"))
        for entry in spec.get("excluded", [])
        if entry
    }

    targets: list[RolloutTarget] = []
    annotations: list[RolloutAnnotation] = []
    for entry in spec.get("available", []):
        json_name = str(entry.get("name", ""))
        lower = json_name.lower()
        if lower in excluded_names:
            annotations.append(
                RolloutAnnotation(
                    json_name=json_name,
                    skipped=True,
                    reason=f"excluded: {excluded_reasons.get(lower, 'excluded')}",
                )
            )
            continue

        config_name = _normalize_name(json_name)
        cam = by_name.get(config_name) if config_name is not None else None
        if cam is None:
            annotations.append(
                RolloutAnnotation(
                    json_name=json_name,
                    skipped=False,
                    reason="unresolved: no matching tenant camera",
                )
            )
            continue

        targets.append(
            RolloutTarget(
                json_name=json_name,
                camera_id=str(cam["id"]),
                camera_name=str(cam.get("name") or cam["id"]),
                map_id=str(cam.get("map_id") or ""),
            )
        )

    return targets, annotations


def _collect_confidence(
    camera_id: str,
    camera_name: str,
    *,
    session_factory: Callable[[], AbstractContextManager[Session]],
) -> CameraConfidenceReport:
    """Collect leaf-3's confidence report for a single camera.

    Opens a session and delegates to leaf-3's read-only join
    (:func:`_collect_reports`) for one ``(camera_id, camera_name)`` spec. Extracted
    as a module-level seam so tests can swap it for an offline double.

    Args:
        camera_id: The camera id to report on.
        camera_name: The human-readable camera name.
        session_factory: Database session context-manager factory.

    Returns:
        The single :class:`CameraConfidenceReport` for the camera.
    """
    with session_factory() as session:
        lens_repo = RepoPostgresLensCalibrationTable(session)
        ptz_repo = RepoPostgresPtzRegistration(session)
        reports = _collect_reports(
            [(camera_id, camera_name)], lens_repo=lens_repo, ptz_repo=ptz_repo
        )
    return reports[0]


@dataclass
class _RolloutDeps:
    """Live collaborators for one rollout, resolved once after env preflight.

    Attributes:
        frame_store: Frame byte store shared across all cameras.
        session_factory: Database session context-manager factory.
        rms_threshold_m: Maximum acceptable hold-out reprojection RMS (meters).
        ortho_cache: Per-map ``(ortho_lines, geotiff)`` memo to avoid re-loading.
    """

    frame_store: FrameStore
    session_factory: Callable[[], AbstractContextManager[Session]]
    rms_threshold_m: float
    ortho_cache: dict = field(default_factory=dict)


def _run_one_camera(target: RolloutTarget, deps: _RolloutDeps) -> CameraOutcome:
    """Run leaf-2 + leaf-3 for one camera, capturing any failure as an outcome.

    Loads the camera's ortho context (memoized per map), runs the per-camera
    registration, then collects its confidence. Hold-out, correspondence, and
    generic failures are caught and recorded so the batch can continue.

    Args:
        target: The resolved camera to run.
        deps: Shared live collaborators for the rollout.

    Returns:
        The :class:`CameraOutcome` (``PASS`` or ``FAIL``) for the camera.
    """
    if not target.map_id:
        return CameraOutcome(
            json_name=target.json_name,
            camera_id=target.camera_id,
            status="FAIL",
            detail="no map_id configured",
        )

    try:
        if target.map_id not in deps.ortho_cache:
            deps.ortho_cache[target.map_id] = _load_ortho_context(target.map_id)
        ortho_lines, geotiff = deps.ortho_cache[target.map_id]

        _run_camera_registration(
            run_id=target.json_name,
            camera_id=target.camera_id,
            map_id=target.map_id,
            ortho_lines=ortho_lines,
            geotiff=geotiff,
            frame_store=deps.frame_store,
            session_factory=deps.session_factory,
            rms_threshold_m=deps.rms_threshold_m,
        )
        report = _collect_confidence(
            target.camera_id, target.camera_name, session_factory=deps.session_factory
        )
    except HoldoutValidationError as exc:
        return CameraOutcome(target.json_name, target.camera_id, "FAIL", f"hold-out: {exc}")
    except InsufficientCorrespondenceError as exc:
        return CameraOutcome(target.json_name, target.camera_id, "FAIL", f"matching: {exc}")
    except (RuntimeError, OSError, ValueError, typer.Exit) as exc:
        return CameraOutcome(target.json_name, target.camera_id, "FAIL", f"error: {exc}")

    return CameraOutcome(
        json_name=target.json_name,
        camera_id=target.camera_id,
        status="PASS",
        detail=f"{report.label} ({report.score:.2f})",
    )


def _run_rollout(
    targets: Sequence[RolloutTarget],
    annotations: Sequence[RolloutAnnotation],
    *,
    frame_store: FrameStore,
    session_factory: Callable[[], AbstractContextManager[Session]],
    rms_threshold_m: float,
) -> list[CameraOutcome]:
    """Run every resolved camera, continuing past any single-camera failure.

    Each target is run independently; a failure on one camera is recorded as a
    ``FAIL`` outcome and does not stop the others. Excluded annotations become
    ``SKIP`` outcomes and unresolved annotations become ``FAIL`` outcomes.

    Args:
        targets: The resolved cameras to run.
        annotations: The excluded/unresolved annotations.
        frame_store: Frame byte store shared across all cameras.
        session_factory: Database session context-manager factory.
        rms_threshold_m: Maximum acceptable hold-out reprojection RMS (meters).

    Returns:
        One :class:`CameraOutcome` per available camera, in input order.
    """
    deps = _RolloutDeps(
        frame_store=frame_store,
        session_factory=session_factory,
        rms_threshold_m=rms_threshold_m,
    )
    outcomes: list[CameraOutcome] = [_run_one_camera(target, deps) for target in targets]
    for annotation in annotations:
        outcomes.append(
            CameraOutcome(
                json_name=annotation.json_name,
                camera_id=None,
                status="SKIP" if annotation.skipped else "FAIL",
                detail=annotation.reason,
            )
        )
    return outcomes


def _format_summary(outcomes: Sequence[CameraOutcome]) -> str:
    """Render the per-camera rollout summary plus a totals line.

    Args:
        outcomes: The per-camera outcomes to summarize.

    Returns:
        A newline-joined summary, one line per camera, ending with a totals line.
    """
    lines: list[str] = []
    passed = failed = skipped = 0
    for outcome in outcomes:
        camera_id = outcome.camera_id or "-"
        lines.append(
            f"{outcome.json_name}  {camera_id}  {outcome.status}  {outcome.detail}".rstrip()
        )
        if outcome.status == "PASS":
            passed += 1
        elif outcome.status == "FAIL":
            failed += 1
        else:
            skipped += 1
    lines.append(f"Rollout: {passed} passed, {failed} failed, {skipped} skipped")
    return "\n".join(lines)


def _load_cameras_file(cameras_file: Path) -> dict:
    """Load and parse the ``calib_cameras.json`` file or exit with a clear error.

    Args:
        cameras_file: Path to the cameras JSON file.

    Returns:
        The parsed JSON dict.

    Raises:
        typer.Exit: If the file is missing or cannot be parsed.
    """
    if not cameras_file.exists():
        typer.echo(f"Error: cameras file not found: {cameras_file}", err=True)
        raise typer.Exit(1)
    try:
        spec = json.loads(cameras_file.read_text())
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: Failed to load cameras file: {exc}", err=True)
        raise typer.Exit(1)
    if not isinstance(spec, dict):
        typer.echo("Error: cameras file must contain a JSON object.", err=True)
        raise typer.Exit(1)
    return spec


@calibrate_app.command("rollout")
def rollout_command(
    tenant: str = typer.Option(..., "--tenant", help="Tenant id (e.g., 'icozee')"),
    cameras_file: Path = typer.Option(
        _DEFAULT_CAMERAS_FILE,
        "--cameras-file",
        help="Path to calib_cameras.json (available + excluded cameras)",
    ),
    rms_threshold_m: float = typer.Option(
        1.0,
        "--rms-threshold-m",
        help="Maximum acceptable hold-out reprojection RMS in meters",
    ),
) -> None:
    """Run the per-camera calibration pipeline across a tenant's available cameras.

    A thin loop over the merged leaves: resolves each ``calib_cameras.json`` name
    to its tenant camera id, skips the excluded set, runs the per-camera
    registration + confidence collection for each, prints a per-camera pass/fail +
    confidence summary, and exits non-zero if ANY camera failed. Adds no new
    calibration logic.
    """
    spec = _load_cameras_file(cameras_file)

    tenant_cameras = get_cameras_for_tenant(tenant)
    if not tenant_cameras:
        typer.echo(f"Error: No cameras found for tenant '{tenant}'.", err=True)
        raise typer.Exit(1)

    # Pre-flight the env contracts up front so a misconfiguration fails fast.
    try:
        frame_store = MinioFrameStore.from_env()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    if not os.environ.get("DATABASE_URL"):
        typer.echo("Error: DATABASE_URL environment variable is not set.", err=True)
        raise typer.Exit(1)

    targets, annotations = _resolve_targets(spec, tenant_cameras)
    outcomes = _run_rollout(
        targets,
        annotations,
        frame_store=frame_store,
        session_factory=get_session,
        rms_threshold_m=rms_threshold_m,
    )

    typer.echo(_format_summary(outcomes))

    if any(outcome.status == "FAIL" for outcome in outcomes):
        raise typer.Exit(1)


__all__ = ["rollout_command"]
