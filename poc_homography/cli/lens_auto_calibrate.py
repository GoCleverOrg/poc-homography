"""Automatic lens-distortion calibration CLI command (``hom calibrate lens-auto``).

Composes the existing pieces: a :class:`CalibrationDevice` (a live Hikvision
adapter, or an offline frame-folder replay), the painted-line detector, the
visibility precheck, the per-zoom solver, and the per-model aggregation, via
:func:`run_auto_calibration`. The orchestrator saves PTZ first and restores it in
a ``finally``, runs an adaptive zoom sweep, and folds every run into the per-model
table so precision grows with ``--runs``.

The ``--offline-dir`` mode reads pre-captured frames from a folder (filenames
encode the zoom, e.g. ``frame_zoom2.5_a.png``) so the command is runnable and
testable without hardware.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import typer

from poc_homography.calibration.lens_distortion.auto_calibration import (
    AutoCalibrationConfig,
    OfflineFrameDevice,
    run_auto_calibration,
)
from poc_homography.calibration.lens_distortion.visibility import (
    NoCalibratableViewError,
    VisibilityCriteria,
)
from poc_homography.cli.main import calibrate_app
from poc_homography.domain.enums.camera_spec import CameraSpec

if TYPE_CHECKING:
    import numpy as np

    from poc_homography.calibration.lens_distortion.auto_calibration import (
        AutoCalibrationResult,
        CalibrationDevice,
    )

# Pull a zoom factor out of a frame filename, e.g. "cam_zoom2.5_001.png" -> 2.5.
_ZOOM_RE = re.compile(r"zoom[_-]?(\d+(?:\.\d+)?)", re.IGNORECASE)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _camera_spec_from_name(name: str) -> CameraSpec:
    """Resolve a :class:`CameraSpec` from a model name or enum member name."""
    for spec in CameraSpec:
        if name in (spec.name, spec.model_name):
            return spec
    valid = ", ".join(s.name for s in CameraSpec)
    raise typer.BadParameter(f"Unknown model '{name}'. Valid: {valid}")


def _load_offline_pairs(offline_dir: Path) -> list[tuple[np.ndarray, float]]:
    """Load ``(image, zoom)`` pairs from a folder (zoom parsed from filenames)."""
    pairs: list[tuple[np.ndarray, float]] = []
    for path in sorted(offline_dir.iterdir()):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        match = _ZOOM_RE.search(path.stem)
        if not match:
            typer.echo(f"Skipping {path.name}: no 'zoom<N>' token in filename", err=True)
            continue
        image = cv2.imread(str(path))
        if image is None:
            typer.echo(f"Skipping {path.name}: failed to decode", err=True)
            continue
        pairs.append((image, float(match.group(1))))
    return pairs


def _build_live_device(ip: str, tenant: str) -> CalibrationDevice:
    """Build a live Hikvision-backed CalibrationDevice (credentials from env)."""
    from poc_homography.calibration.lens_distortion.auto_calibration import DeviceFrameCapturer
    from poc_homography.camera_config import get_tenant_credentials
    from poc_homography.infrastructure.clients.hikvision.isapi_client import HikvisionISAPIClient

    username, password = get_tenant_credentials(tenant)
    if not username or not password:
        typer.echo(
            f"Error: missing credentials; set {tenant.upper()}_CAMERA_USERNAME / "
            f"{tenant.upper()}_CAMERA_PASSWORD.",
            err=True,
        )
        raise typer.Exit(2)
    return DeviceFrameCapturer(HikvisionISAPIClient(ip, username, password))


@calibrate_app.command("lens-auto")
def lens_auto_command(
    camera_id: str = typer.Option(..., "--camera-id", help="Camera id (table id)"),
    model: str = typer.Option(
        CameraSpec.HIKVISION_DS_2DF8425IX.name, "--model", help="Camera model (CameraSpec)"
    ),
    runs: int = typer.Option(1, "--runs", help="Repeat capture/solve N times (precision grows)"),
    zoom_tol: float = typer.Option(0.01, "--zoom-tol", help="Adaptive zoom k1 tolerance"),
    zoom_min: float = typer.Option(1.0, "--zoom-min", help="Coarse sweep min zoom"),
    zoom_max: float = typer.Option(25.0, "--zoom-max", help="Coarse sweep max zoom"),
    coarse_steps: int = typer.Option(5, "--coarse-steps", help="Coarse zoom levels"),
    min_lines: int = typer.Option(8, "--min-lines", help="Visibility: min lines"),
    min_quadrants: int = typer.Option(2, "--min-quadrants", help="Visibility: min quadrants"),
    min_orientations: int = typer.Option(
        2, "--min-orientations", help="Visibility: min orientation buckets"
    ),
    output_dir: Path = typer.Option(
        Path("."), "--output-dir", help="Root dir for calibration_results/ YAML"
    ),
    offline_dir: Path | None = typer.Option(
        None, "--offline-dir", help="Replay pre-captured (image,zoom) frames instead of a camera"
    ),
    live: bool = typer.Option(False, "--live", help="Drive a live camera (requires --ip)"),
    ip: str | None = typer.Option(None, "--ip", help="Live camera host/IP (with --live)"),
    tenant: str = typer.Option(
        "ico", "--tenant", help="Tenant id for camera credentials (env {TENANT}_CAMERA_*)"
    ),
) -> None:
    """
    Automatically calibrate lens distortion across zoom, restoring PTZ.

    Use ``--offline-dir`` to calibrate from pre-captured frames, or
    ``--live --ip <host>`` to drive a real Hikvision camera (credentials from the
    ``{TENANT}_CAMERA_USERNAME`` / ``{TENANT}_CAMERA_PASSWORD`` env vars). The
    camera PTZ is saved first and restored in a ``finally``. Persists a per-camera
    table under ``calibration_results/{camera_id}.yaml`` and a per-model table
    under ``calibration_results/models/{model}.yaml``. Exits non-zero with a clear
    message when no zoom yields a calibratable view.
    """
    camera_spec = _camera_spec_from_name(model)
    config = AutoCalibrationConfig(
        zoom_min=zoom_min,
        zoom_max=zoom_max,
        coarse_steps=coarse_steps,
        zoom_tol=zoom_tol,
        runs=runs,
    )
    criteria = VisibilityCriteria(
        min_lines=min_lines,
        min_quadrants=min_quadrants,
        min_orientations=min_orientations,
    )

    device: CalibrationDevice
    if live:
        if not ip:
            typer.echo("Error: --live requires --ip <host>.", err=True)
            raise typer.Exit(2)
        device = _build_live_device(ip, tenant)
    elif offline_dir is not None:
        if not offline_dir.is_dir():
            typer.echo(f"Error: offline-dir '{offline_dir}' is not a directory.", err=True)
            raise typer.Exit(2)
        pairs = _load_offline_pairs(offline_dir)
        if not pairs:
            typer.echo(f"Error: no usable frames found in '{offline_dir}'.", err=True)
            raise typer.Exit(2)
        device = OfflineFrameDevice(pairs)
    else:
        typer.echo("Error: pass --offline-dir <dir> or --live --ip <host>.", err=True)
        raise typer.Exit(2)

    try:
        result = run_auto_calibration(
            device,
            camera_id=camera_id,
            camera_spec=camera_spec,
            config=config,
            criteria=criteria,
            output_dir=output_dir,
        )
    except NoCalibratableViewError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    _print_report(result)


def _print_report(result: AutoCalibrationResult) -> None:
    """Print per-zoom entries, skips, and a summary line."""
    table = result.camera_table
    for entry in table.entries:
        typer.echo(
            f"zoom={float(entry.zoom_factor):<6g} "
            f"k1={float(entry.distortion.k1):<+10.5f} "
            f"k2={float(entry.distortion.k2):<+10.5f} "
            f"lines={entry.num_lines_used:<3} "
            f"rmse={entry.validation_rmse:.4f}"
        )
    for skipped in result.skipped_zooms:
        typer.echo(
            f"skipped zoom={skipped.zoom_factor:<6g} "
            f"lines={skipped.num_lines:<3} reason={skipped.reason}"
        )
    typer.echo(
        f"Done: camera_id={table.id} model={result.model_table.model_name} "
        f"zooms_calibrated={len(table.entries)} skipped={len(result.skipped_zooms)} "
        f"model_runs={len(result.model_table.run_history)} restored={result.restored}"
    )
