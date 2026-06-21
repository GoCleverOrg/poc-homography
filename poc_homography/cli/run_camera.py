"""End-to-end single-camera calibration runner CLI command (#365).

Pure composition of the already-merged stages — this leaf adds **no** new
solver, capture, or calibration logic. It glues, for ONE camera:

1. :func:`calibrate_capture._run_calibration_capture` — live calibration-sweep
   capture (skipped when ``--offline-run-id`` reuses an already-captured run).
2. :func:`self_calibrate._run_self_calibration` — per-zoom scene self-calibration
   into a persisted :class:`LensCalibrationTable`.
3. :func:`extrinsic.auto_match.match_and_register` — per captured PTZ frame,
   discover the frame-line ↔ ortho-line correspondence (the AUTO matcher; an
   optional ``pose_prior`` may be supplied to disambiguate repetitive lines).
4. :func:`extrinsic.validation.validate_with_holdout` — hold-out reprojection
   validation in ground meters; a hold-out RMS over the threshold raises the
   typed :class:`HoldoutValidationError`, failing the whole camera.
5. :class:`RepoPostgresPtzRegistration` (#364) — persist one
   :class:`PtzRegistration` per validated frame.

The extracted helper :func:`_run_camera_registration` references every
collaborator (the calibrate helper, the frame loader, the extrinsic solvers, and
both repositories) as module-level names so tests can swap them for doubles and
exercise calibrate→register→validate→persist with NO live network, MinIO, or
Neon access. ``--offline-run-id`` selects that offline-testable path at the CLI.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path  # runtime import: Typer resolves the --plan annotation
from typing import TYPE_CHECKING, cast

import typer

from poc_homography.calibration.extrinsic import (
    HoldoutValidationError,
    InsufficientCorrespondenceError,
    match_and_register,
    validate_with_holdout,
)
from poc_homography.calibration.lens_distortion.ddd_sync import lens_table_to_camera_table
from poc_homography.calibration.lens_distortion.frame_source import iter_survey_frames
from poc_homography.calibration.lens_distortion.line_detection import LineDetector
from poc_homography.cli.calibrate_capture import (
    _run_calibration_capture,
    make_rtsp_url_resolver,
)
from poc_homography.cli.main import calibrate_app
from poc_homography.cli.self_calibrate import _run_self_calibration
from poc_homography.domain.entities.ptz_registration import PtzRegistration
from poc_homography.infrastructure.clients.minio_frame_store import MinioFrameStore
from poc_homography.infrastructure.database import get_session
from poc_homography.infrastructure.repositories.repo_postgres_camera_config import (
    RepoPostgresCameraConfig,
)
from poc_homography.infrastructure.repositories.repo_postgres_clean_plate_frame import (
    RepoPostgresCleanPlateFrame,
)
from poc_homography.infrastructure.repositories.repo_postgres_map import (
    RepoPostgresMap,
)
from poc_homography.infrastructure.repositories.repo_postgres_ptz_registration import (
    RepoPostgresPtzRegistration,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from contextlib import AbstractContextManager
    from typing import Protocol

    import numpy as np
    import numpy.typing as npt
    from sqlalchemy.orm import Session

    from poc_homography.calibration.lens_distortion.frame_source import FrameRepo, FrameStore
    from poc_homography.calibration.lens_distortion.line_detection import CandidateLine
    from poc_homography.domain.vo.geotiff import GeoTiff
    from poc_homography.domain.vo.ortho_line import OrthoLine

    class _FrameLineDetector(Protocol):
        """Structural type for a frame line detector (``LineDetector``-shaped)."""

        def detect(self, image: np.ndarray) -> list[CandidateLine]:
            """Detect candidate frame lines in a BGR image."""
            ...

    PosePriorProvider = Callable[[float, float], npt.NDArray[np.float64] | None]


def _run_camera_registration(
    *,
    run_id: str,
    camera_id: str,
    map_id: str,
    ortho_lines: Sequence[OrthoLine],
    geotiff: GeoTiff,
    frame_store: FrameStore,
    session_factory: Callable[[], AbstractContextManager[Session]],
    rms_threshold_m: float,
    line_detector: _FrameLineDetector | None = None,
    pose_prior_provider: PosePriorProvider | None = None,
) -> list[PtzRegistration]:
    """Glue calibrate→register→validate→persist for one camera's stored frames.

    Self-calibrates the run's frames into a lens table, then for every stored
    frame discovers the frame↔ortho correspondence with the AUTO matcher,
    hold-out-validates the resulting registration in ground meters, and persists
    one :class:`PtzRegistration` per validated frame. Every persisted save shares
    a single session/transaction, so a :class:`HoldoutValidationError` on ANY
    frame aborts the whole camera and rolls back the run's registrations.

    Extracted from the command body so tests can inject a fake frame store, a
    fake session factory, and a fake line detector (and monkeypatch the
    module-level collaborators) with no live MinIO, Neon, or camera access.

    Args:
        run_id: Survey run whose stored frames are registered.
        camera_id: Camera id; also the lens-table and registration join key.
        map_id: Orthophoto map identifier handed to the homography provider.
        ortho_lines: Orthophoto lines, keyed positionally as ``ortho_{i}``.
        geotiff: Georeference for converting map-pixel error to ground meters.
        frame_store: Frame byte store (injected; offline-testable).
        session_factory: Database session context-manager factory.
        rms_threshold_m: Maximum acceptable hold-out reprojection RMS (meters).
        line_detector: Frame line detector; defaults to a real :class:`LineDetector`.
        pose_prior_provider: Optional ``(zoom, tilt) -> 3x3 frame→ortho prior``
            used only to gate AUTO-matcher candidates; ``None`` keeps line-only
            matching.

    Returns:
        The list of persisted :class:`PtzRegistration` records, one per frame.

    Raises:
        HoldoutValidationError: If any frame's hold-out RMS exceeds the threshold.
        InsufficientCorrespondenceError: If the AUTO matcher cannot match a frame.
        typer.Exit: If self-calibration finds no frames for the run.
    """
    detector = line_detector if line_detector is not None else LineDetector()
    now = datetime.now().isoformat()

    calibration = _run_self_calibration(
        run_id=run_id,
        camera_id=camera_id,
        frame_store=frame_store,
        session_factory=session_factory,
    )
    calibration_table = lens_table_to_camera_table(calibration.table)

    registrations: list[PtzRegistration] = []
    with session_factory() as session:
        # The concrete repo's query surface is wider than the loader's FrameRepo
        # protocol; cast at the boundary since it structurally satisfies the reads.
        frame_repo = cast("FrameRepo", RepoPostgresCleanPlateFrame(session))
        ptz_repo = RepoPostgresPtzRegistration(session)

        for frame in iter_survey_frames(
            run_id=run_id, camera_id=camera_id, repo=frame_repo, store=frame_store
        ):
            frame_lines = detector.detect(frame.image)
            height, width = frame.image.shape[:2]
            pose_prior = (
                pose_prior_provider(frame.commanded_zoom, frame.commanded_tilt)
                if pose_prior_provider is not None
                else None
            )

            auto = match_and_register(
                frame_lines=frame_lines,
                ortho_lines=ortho_lines,
                calibration_table=calibration_table,
                commanded_zoom=frame.commanded_zoom,
                commanded_tilt=frame.commanded_tilt,
                map_id=map_id,
                image_width=int(width),
                image_height=int(height),
                pose_prior=pose_prior,
                run_id=run_id,
                camera_id=camera_id,
            )

            # Hold-out validation re-solves on a fit subset and checks the held-out
            # RMS; it raises HoldoutValidationError when RMS exceeds the threshold.
            validation = validate_with_holdout(
                frame_lines=frame_lines,
                ortho_lines=ortho_lines,
                seed=auto.seed,
                calibration_table=calibration_table,
                commanded_zoom=frame.commanded_zoom,
                commanded_tilt=frame.commanded_tilt,
                map_id=map_id,
                geotiff=geotiff,
                rms_threshold_m=rms_threshold_m,
                run_id=run_id,
                camera_id=camera_id,
            )

            record = PtzRegistration.from_result(
                validation.registration,
                reprojection_stats=validation.holdout_stats,
                created_date=now,
                last_modified=now,
            )
            ptz_repo.save(record)
            registrations.append(record)

    return registrations


def _require_reference_rows(
    camera_id: str,
    map_id: str,
    *,
    tenant_id: str,
    session_factory: Callable[[], AbstractContextManager[Session]],
) -> None:
    """Fail fast with a clear fix when the pipeline's FK reference rows are absent.

    The pipeline persists registrations keyed by ``camera_id`` under an FK chain
    rooted at the tenant -> map -> camera_config rows. When either the camera's
    ``camera_config`` or its ``map`` row is missing, persistence would raise a
    ForeignKeyViolation deep in the run; this preflight surfaces that up front
    and names the bootstrap fix instead of auto-mutating the DB.

    Args:
        camera_id: The camera whose ``camera_config`` row must exist.
        map_id: The map row the camera references.
        tenant_id: The tenant id, named in the remediation message.
        session_factory: Database session context-manager factory.

    Raises:
        typer.Exit: If either reference row is missing.
    """
    with session_factory() as session:
        camera_exists = RepoPostgresCameraConfig(session).get(camera_id) is not None
        map_exists = RepoPostgresMap(session).get(map_id) is not None
    if not (camera_exists and map_exists):
        typer.echo(
            f"Error: reference rows missing for camera '{camera_id}'. "
            f"Run: hom calibrate bootstrap --tenant {tenant_id} first.",
            err=True,
        )
        raise typer.Exit(1)


def _resolve_camera(camera: str) -> dict:
    """Resolve a camera name/id to its config dict or exit with a clear error."""
    from poc_homography.camera_config import get_camera_by_name, get_camera_configs

    cam_info = get_camera_by_name(camera)
    if not cam_info:
        available = [c["name"] for c in get_camera_configs()]
        typer.echo(
            f"Error: Camera '{camera}' not found. Available cameras: {', '.join(available)}",
            err=True,
        )
        raise typer.Exit(1)
    return cam_info


def _live_capture(
    *,
    cam_info: dict,
    plan: Path,
    frame_store: MinioFrameStore,
    run_id: str,
) -> None:
    """Drive ONE live camera through the calibration sweep (capture → frames).

    Composes the same public collaborators the standalone ``calibrate-capture``
    command uses (plan-config sidecar, tenant credentials, live ISAPI client,
    Postgres+MinIO sink) and delegates the sweep to the shared
    :func:`_run_calibration_capture` helper — it does not re-implement capture.
    """
    import yaml

    from poc_homography.camera_config import get_tenant_credentials
    from poc_homography.domain.entities.camera_config import CameraConfig
    from poc_homography.domain.enums import CameraSpec
    from poc_homography.domain.vo.credential import Credential
    from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig
    from poc_homography.infrastructure.clients.hikvision.isapi_client import HikvisionISAPIClient
    from poc_homography.infrastructure.survey.postgres_phase_sink import PostgresPhaseSink
    from poc_homography.survey.phases.runner import SurveyPlan

    try:
        with plan.open() as handle:
            config = SurveyPlanConfig.from_dict(yaml.safe_load(handle))
    except (OSError, yaml.YAMLError, ValueError, TypeError) as exc:
        typer.echo(f"Error: Failed to load plan config: {exc}", err=True)
        raise typer.Exit(1)
    runner_plan = SurveyPlan.from_plan_config(config)

    tenant_id = cam_info.get("tenant_id") or ""
    username, password = get_tenant_credentials(tenant_id)
    if not username or not password:
        typer.echo(
            f"Error: Camera credentials not set for tenant '{tenant_id}'.",
            err=True,
        )
        raise typer.Exit(1)

    camera_config = CameraConfig(
        id=cam_info["id"],
        tenant_id=tenant_id,
        map_id=cam_info.get("map_id", ""),
        name=cam_info.get("name", cam_info["id"]),
        spec=CameraSpec.HIKVISION_DS_2DF8425IX,
        credential=Credential(username, password),
        ip_address=cam_info["ip"],
    )
    try:
        client = HikvisionISAPIClient.from_config(camera_config)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    sink = PostgresPhaseSink(frame_store)
    base_output_dir = Path(tempfile.mkdtemp())
    try:
        _run_calibration_capture(
            camera=client,
            plan_config=config,
            runner_plan=runner_plan,
            camera_id=cam_info["id"],
            run_id=run_id,
            base_output_dir=base_output_dir,
            sink=sink,
            session_factory=get_session,
            rtsp_url_resolver=make_rtsp_url_resolver(cam_info.get("name", cam_info["id"])),
        )
    finally:
        shutil.rmtree(base_output_dir, ignore_errors=True)


def _load_ortho_context(map_id: str) -> tuple[Sequence[OrthoLine], GeoTiff]:
    """Load a map's ortho lines + georeference from Neon + MinIO (live path).

    Resolves the :class:`Map` entity (its authoritative ``geotiff`` + ``asset_key``)
    from Neon, downloads the GeoTIFF raster from MinIO, and detects the orthophoto
    lines. Live-only; the offline runner is fed ortho lines + geotiff directly.
    """
    from poc_homography.infrastructure.clients.minio_map_store import MinioMapStore
    from poc_homography.infrastructure.repositories.repo_postgres_map import RepoPostgresMap
    from poc_homography.maps.geotiff_io import load_ortho_raster_from_bytes
    from poc_homography.maps.ortho_line_detection import OrthoLineDetector

    with get_session() as session:
        map_entity = RepoPostgresMap(session).get(map_id)
    if map_entity is None or not map_entity.asset_key:
        typer.echo(
            f"Error: Map '{map_id}' not found or has no uploaded GeoTIFF asset.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        raster = MinioMapStore.from_env().get_map(map_entity.asset_key)
    except (RuntimeError, OSError) as exc:
        typer.echo(f"Error: Failed to load map asset: {exc}", err=True)
        raise typer.Exit(1)

    image, _raster_geotiff = load_ortho_raster_from_bytes(raster)
    ortho_lines = OrthoLineDetector().detect(image, map_entity.geotiff)
    return ortho_lines, map_entity.geotiff


@calibrate_app.command("run-camera")
def run_camera_command(
    camera: str = typer.Option(..., "--camera", help="Camera name or id (e.g., 'valte_cam01')"),
    plan: Path | None = typer.Option(
        None, "--plan", help="YAML survey plan config (required for live capture)"
    ),
    offline_run_id: str | None = typer.Option(
        None,
        "--offline-run-id",
        help="Skip live capture and reuse an already-captured survey run id",
    ),
    rms_threshold_m: float = typer.Option(
        1.0,
        "--rms-threshold-m",
        help="Maximum acceptable hold-out reprojection RMS in meters",
    ),
) -> None:
    """Run the end-to-end single-camera calibration→registration pipeline.

    With ``--offline-run-id`` the live capture is skipped and the pipeline runs
    calibrate→register→validate→persist over the already-captured frames. Live
    runs require ``--plan`` and drive the camera through the calibration sweep
    first. A hold-out RMS over ``--rms-threshold-m`` fails the camera with a
    non-zero exit and a clear message.
    """
    cam_info = _resolve_camera(camera)
    camera_id = cam_info["id"]
    map_id = cam_info.get("map_id") or ""
    if not map_id:
        typer.echo(f"Error: Camera '{camera}' has no map_id configured.", err=True)
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

    _require_reference_rows(
        camera_id,
        map_id,
        tenant_id=cam_info.get("tenant_id") or "",
        session_factory=get_session,
    )

    if offline_run_id:
        run_id = offline_run_id
    else:
        if plan is None:
            typer.echo(
                "Error: --plan is required for a live run (omit only with --offline-run-id).",
                err=True,
            )
            raise typer.Exit(1)
        run_id = uuid.uuid4().hex
        _live_capture(cam_info=cam_info, plan=plan, frame_store=frame_store, run_id=run_id)

    ortho_lines, geotiff = _load_ortho_context(map_id)

    try:
        registrations = _run_camera_registration(
            run_id=run_id,
            camera_id=camera_id,
            map_id=map_id,
            ortho_lines=ortho_lines,
            geotiff=geotiff,
            frame_store=frame_store,
            session_factory=get_session,
            rms_threshold_m=rms_threshold_m,
        )
    except HoldoutValidationError as exc:
        typer.echo(
            f"Error: Hold-out validation failed for camera {camera_id}: {exc}",
            err=True,
        )
        raise typer.Exit(1)
    except InsufficientCorrespondenceError as exc:
        typer.echo(f"Error: Automatic matching failed for camera {camera_id}: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Done: camera_id={camera_id} run_id={run_id} registrations={len(registrations)}")


__all__ = ["run_camera_command"]
