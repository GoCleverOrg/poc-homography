"""Unattended per-zoom scene self-calibration CLI command (#351).

Pure composition of existing read-side pieces: enumerate a survey run's stored
frames offline via :func:`iter_survey_frames` (frame metadata from Neon's
``clean_plate_frames`` via :class:`RepoPostgresCleanPlateFrame`, image bytes from
MinIO via :class:`MinioFrameStore`), feed the decoded ``(image, zoom)`` pairs to
:func:`calibrate_scene_self`, and persist the resulting per-zoom
:class:`LensCalibrationTable` (``K(zoom)`` + distortion) to Neon -- closing the
#338 (capture) -> #339 (calibrate) loop. The helper is fully offline-testable:
the loader, the calibrator, and both repositories are referenced as module-level
names so tests can swap them for doubles.
"""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003  # runtime import: Typer resolves the --yaml-dir annotation
from typing import TYPE_CHECKING, cast

import typer

from poc_homography.calibration.lens_distortion.frame_source import iter_survey_frames
from poc_homography.calibration.lens_distortion.scene_self_calibration import (
    calibrate_scene_self,
)
from poc_homography.cli.main import calibrate_app
from poc_homography.infrastructure.clients.minio_frame_store import MinioFrameStore
from poc_homography.infrastructure.database import get_session
from poc_homography.infrastructure.repositories.repo_postgres_clean_plate_frame import (
    RepoPostgresCleanPlateFrame,
)
from poc_homography.infrastructure.repositories.repo_postgres_lens_calibration_table import (
    RepoPostgresLensCalibrationTable,
)
from poc_homography.infrastructure.repositories.repo_yaml_lens_calibration_table import (
    RepoYamlLensCalibrationTable,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    import numpy as np
    from sqlalchemy.orm import Session

    from poc_homography.calibration.lens_distortion.frame_source import FrameRepo, FrameStore
    from poc_homography.calibration.lens_distortion.scene_self_calibration import (
        SceneCalibrationResult,
    )

# Grouping window (in zoom units) the calibrator buckets frames into; mirrors
# ``ScenePerZoomConfig.zoom_tolerance``. A ``--zoom`` filter value matches a
# frame iff both round to the *same* group key, so the filter admits exactly the
# frames the calibrator would route to the requested zoom group -- not the wider
# neighbouring groups a symmetric +/- tolerance window would also capture.
_ZOOM_GROUP_TOLERANCE = 0.1


@calibrate_app.command("self-calibrate")
def self_calibrate_command(
    run_id: str = typer.Option(..., "--run-id", help="Survey run id to calibrate from"),
    camera: str = typer.Option(..., "--camera", help="Camera id (table id)"),
    zoom: list[float] | None = typer.Option(
        None, "--zoom", help="Optional zoom factor(s) to restrict to"
    ),
    yaml_dir: Path | None = typer.Option(
        None, "--yaml-dir", help="Optional dir to also write a YAML table for inspection"
    ),
) -> None:
    """
    Calibrate a survey run's stored frames into a persisted per-zoom lens table.

    Enumerates the run's frames offline (Neon metadata + MinIO bytes), self-
    calibrates ``K(zoom)`` plus distortion per zoom group via
    :func:`calibrate_scene_self`, persists the resulting
    :class:`LensCalibrationTable` to Neon (and optionally a YAML copy for
    inspection), then prints a per-zoom residual report. Both the MinIO and the
    database env contracts are pre-flighted up front so a misconfiguration fails
    fast.
    """
    # Pre-flight the env contracts before any work so misconfig fails fast.
    try:
        frame_store = MinioFrameStore.from_env()
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    if not os.environ.get("DATABASE_URL"):
        typer.echo("Error: DATABASE_URL environment variable is not set.", err=True)
        raise typer.Exit(1)

    result = _run_self_calibration(
        run_id=run_id,
        camera_id=camera,
        frame_store=frame_store,
        session_factory=get_session,
        zoom_filter=zoom or None,
        yaml_dir=yaml_dir,
    )

    _print_report(camera_id=camera, result=result)


def _run_self_calibration(
    *,
    run_id: str,
    camera_id: str,
    frame_store: FrameStore,
    session_factory: Callable[[], AbstractContextManager[Session]],
    zoom_filter: list[float] | None = None,
    yaml_dir: Path | None = None,
) -> SceneCalibrationResult:
    """Calibrate a run's stored frames into a persisted lens table.

    Extracted from the command body so tests can inject a fake frame store, a
    fake session factory, and monkeypatch the loader/calibrator/repos -- with no
    live MinIO, Neon, or solver access. Opens a session, drains the run's frames
    via :func:`iter_survey_frames`, optionally filters them to the requested
    zoom group(s), self-calibrates per zoom, and saves the table to Postgres
    (then, after the DB commit, a YAML copy when ``yaml_dir`` is given).

    Args:
        run_id: Survey run to calibrate from.
        camera_id: Camera id; also the persisted table id.
        frame_store: Frame byte store (injected; offline-testable).
        session_factory: Database session context-manager factory.
        zoom_filter: Optional zoom factor(s) to restrict the frames to.
        yaml_dir: Optional directory to also write a YAML table copy.

    Returns:
        The :class:`SceneCalibrationResult` carrying the table and skipped zooms.

    Raises:
        typer.Exit: If no frames are found for ``run_id``/``camera_id``.
    """
    with session_factory() as session:
        # The concrete repo's query surface is wider than the loader's FrameRepo
        # protocol (extra optional filters, a richer row type); cast at the
        # boundary since it structurally satisfies the reads the loader makes.
        frame_repo = cast("FrameRepo", RepoPostgresCleanPlateFrame(session))
        image_zoom_pairs: list[tuple[np.ndarray, float]] = []
        for frame in iter_survey_frames(
            run_id=run_id, camera_id=camera_id, repo=frame_repo, store=frame_store
        ):
            if zoom_filter is not None and not _zoom_matches(frame.commanded_zoom, zoom_filter):
                continue
            image_zoom_pairs.append((frame.image, frame.commanded_zoom))

        if not image_zoom_pairs:
            typer.echo(
                f"Error: No frames found for run_id={run_id} camera_id={camera_id}"
                + (f" matching zoom(s) {zoom_filter}" if zoom_filter else "")
                + ".",
                err=True,
            )
            raise typer.Exit(1)

        result = calibrate_scene_self(image_zoom_pairs, camera_id=camera_id)

        RepoPostgresLensCalibrationTable(session).save(result.table)

    # The YAML copy is an optional diagnostic artifact written only after the DB
    # session has committed, so a YAML write failure (bad path, permissions,
    # disk full) cannot roll back the persisted lens table.
    if yaml_dir is not None:
        RepoYamlLensCalibrationTable(yaml_dir).save(result.table)

    return result


def _zoom_group_key(zoom: float) -> float:
    """Round ``zoom`` to its calibrator group key (mirrors group_images_by_zoom)."""
    return round(round(zoom / _ZOOM_GROUP_TOLERANCE) * _ZOOM_GROUP_TOLERANCE, 10)


def _zoom_matches(commanded_zoom: float, zoom_filter: list[float]) -> bool:
    """Return True if ``commanded_zoom`` shares a zoom group with any filter value.

    Compares rounded group keys (not a symmetric tolerance window) so a
    ``--zoom`` value admits exactly the frames the calibrator routes to that
    zoom group, never the adjacent groups a wider window would also capture.
    """
    key = _zoom_group_key(commanded_zoom)
    return any(_zoom_group_key(z) == key for z in zoom_filter)


def _print_report(*, camera_id: str, result: SceneCalibrationResult) -> None:
    """Print a per-zoom residual report plus a final summary line."""
    for entry in result.table.entries:
        typer.echo(
            f"zoom={float(entry.zoom_factor):<6} "
            f"lines={entry.num_lines_used:<3} "
            f"rmse={entry.validation_rmse:<8} "
            f"fx={float(entry.fx):<10} "
            f"fy={float(entry.fy):<10} "
            f"reproj_px={entry.reprojection_error_px}"
        )
    for skipped in result.skipped_zooms:
        typer.echo(
            f"skipped zoom={skipped.zoom_factor:<6} "
            f"lines={skipped.num_lines:<3} reason={skipped.reason}"
        )
    typer.echo(
        f"Done: camera_id={camera_id} "
        f"zooms_calibrated={len(result.table.entries)} "
        f"skipped={len(result.skipped_zooms)}"
    )
