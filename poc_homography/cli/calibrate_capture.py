"""Live single-camera calibration-sweep capture CLI command (#344).

Pure composition of existing pieces: load a :class:`SurveyPlanConfig` sidecar,
resolve and build a live :class:`HikvisionISAPIClient`, drive ONE camera through
the nine-phase calibration sweep via :func:`execute_survey` (which already
brackets the sweep in :func:`preserve_ptz_position`), routing every frame to a
:class:`PostgresPhaseSink` (MinIO image + Neon ``clean_plate_frames`` row), then
persisting the ``survey_runs`` header and its plan-config sidecar.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import typer
import yaml

from poc_homography.camera_config import (
    get_camera_by_name,
    get_camera_configs,
    get_tenant_credentials,
)
from poc_homography.cli.cleanplate import cleanplate_app
from poc_homography.domain.entities.camera_config import CameraConfig
from poc_homography.domain.entities.survey.survey_run import SurveyRun
from poc_homography.domain.enums import CameraSpec
from poc_homography.domain.vo.credential import Credential
from poc_homography.domain.vo.survey_plan_config import SurveyPlanConfig
from poc_homography.infrastructure.clients.hikvision.isapi_client import HikvisionISAPIClient
from poc_homography.infrastructure.clients.minio_frame_store import MinioFrameStore
from poc_homography.infrastructure.database import get_session
from poc_homography.infrastructure.repositories.repo_postgres_survey_run import (
    RepoPostgresSurveyRun,
)
from poc_homography.infrastructure.survey.postgres_phase_sink import PostgresPhaseSink
from poc_homography.survey.phases.runner import SurveyPlan, execute_survey

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from datetime import datetime

    from sqlalchemy.orm import Session

    from poc_homography.survey.phases.common import PhaseSink
    from poc_homography.survey.phases.runner import SurveyExecution


@cleanplate_app.command("calibrate-capture")
def calibrate_capture_command(
    camera: str = typer.Option(..., help="Camera name or id (e.g., 'valte_cam01')"),
    plan: Path = typer.Option(..., help="Path to a YAML survey plan config file"),
    output_dir: Path | None = typer.Option(
        None, help="Phase scratch root; defaults to a temp directory"
    ),
) -> None:
    """
    Drive ONE live camera through the calibration sweep and persist its frames.

    Loads the plan-config sidecar, resolves the camera and its tenant
    credentials, builds a live ISAPI client, and runs the nine-phase sweep with
    a Postgres+MinIO sink so every frame lands in ``clean_plate_frames`` (image
    in MinIO, metadata in Neon). The ``survey_runs`` header and the plan-config
    sidecar are upserted on completion. PTZ position is restored by the runner.
    """
    # 1. Load the plan-config sidecar and build the runner plan.
    try:
        with Path(plan).open() as f:
            data = yaml.safe_load(f)
        config = SurveyPlanConfig.from_dict(data)
    except (OSError, yaml.YAMLError, ValueError, TypeError) as e:
        typer.echo(f"Error: Failed to load plan config: {e}", err=True)
        raise typer.Exit(1)
    runner_plan = SurveyPlan.from_plan_config(config)

    # 2. Resolve the camera and its tenant credentials.
    cam_info = get_camera_by_name(camera)
    if not cam_info:
        available = [c["name"] for c in get_camera_configs()]
        typer.echo(
            f"Error: Camera '{camera}' not found. Available cameras: {', '.join(available)}",
            err=True,
        )
        raise typer.Exit(1)

    tenant_id = cam_info.get("tenant_id") or ""
    username, password = get_tenant_credentials(tenant_id)
    if not username or not password:
        typer.echo(
            f"Error: Camera credentials not set for tenant '{tenant_id}'. "
            f"Set {tenant_id.upper()}_CAMERA_USERNAME / {tenant_id.upper()}_CAMERA_PASSWORD "
            "(or global CAMERA_USERNAME / CAMERA_PASSWORD) environment variables.",
            err=True,
        )
        raise typer.Exit(1)

    # 3. Build the live client via the required from_config seam.
    camera_config = CameraConfig(
        id=cam_info["id"],
        tenant_id=cam_info.get("tenant_id", ""),
        map_id=cam_info.get("map_id", ""),
        name=cam_info.get("name", cam_info["id"]),
        spec=CameraSpec.HIKVISION_DS_2DF8425IX,
        credential=Credential(username, password),
        ip_address=cam_info["ip"],
    )
    try:
        client = HikvisionISAPIClient.from_config(camera_config)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # 4. Build the real Postgres+MinIO sink.
    try:
        frame_store = MinioFrameStore.from_env()
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    sink = PostgresPhaseSink(frame_store)

    # 5/6/7. Run the sweep and persist the run header + plan-config sidecar.
    run_id = uuid.uuid4().hex
    base_output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp())
    execution = _run_calibration_capture(
        camera=client,
        plan_config=config,
        runner_plan=runner_plan,
        camera_id=cam_info["id"],
        run_id=run_id,
        base_output_dir=base_output_dir,
        sink=sink,
        session_factory=get_session,
    )

    frame_count = sum(len(result.frames) for result in execution.results)
    typer.echo(f"Done: run_id={run_id} camera_id={cam_info['id']} frames={frame_count}")


def _run_calibration_capture(
    *,
    camera: object,
    plan_config: SurveyPlanConfig,
    runner_plan: SurveyPlan,
    camera_id: str,
    run_id: str,
    base_output_dir: Path,
    sink: PhaseSink,
    session_factory: Callable[[], AbstractContextManager[Session]],
    clock: Callable[[], datetime] | None = None,
    uuid_factory: Callable[[], str] | None = None,
) -> SurveyExecution:
    """Run the sweep and persist the run header + plan-config sidecar.

    Extracted from the command body so tests can inject a fake camera, a stub
    sink, and a fake session factory without any live network, MinIO, or Neon
    access. ``clock``/``uuid_factory`` are forwarded to :func:`execute_survey`
    (``None`` keeps the runner's wall-clock defaults) purely as test seams.
    ``execute_survey`` already restores PTZ via :func:`preserve_ptz_position`,
    so no PTZ bracketing is added here.
    """
    execution = execute_survey(
        camera,  # type: ignore[arg-type]
        run_id=run_id,
        camera_id=camera_id,
        sink=sink,
        base_output_dir=base_output_dir,
        plan=runner_plan,
        clock=clock,
        uuid_factory=uuid_factory,
    )

    with session_factory() as session:
        repo = RepoPostgresSurveyRun(session, SurveyRun.from_dict)
        repo.save(execution.run)
        repo.save_plan_config(run_id, plan_config)

    return execution
