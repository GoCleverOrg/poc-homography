"""Clean-plate offline reconstruction CLI commands (#277)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 (runtime type used by typer.Option)
from typing import TYPE_CHECKING

import numpy as np
import typer

from poc_homography.cleanplate import (
    CleanPlateDataset,
    GroundRaster,
    make_synthetic_visits,
    reconstruct_clean_plate,
    write_clean_plate,
)
from poc_homography.types import Meters, Unitless

if TYPE_CHECKING:
    from poc_homography.cleanplate import CleanPlateResult

cleanplate_app = typer.Typer(help="Clean-plate offline reconstruction commands")


def _build_raster(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    pixels_per_meter: float,
) -> GroundRaster:
    """Build a :class:`GroundRaster` from CLI extent options, validating it."""
    try:
        return GroundRaster(
            x_min=Meters(x_min),
            x_max=Meters(x_max),
            y_min=Meters(y_min),
            y_max=Meters(y_max),
            pixels_per_meter=Unitless(pixels_per_meter),
        )
    except ValueError as exc:
        typer.echo(f"Error: Invalid raster extent: {exc}", err=True)
        raise typer.Exit(1) from exc


def _coverage_percent(result: CleanPlateResult) -> float:
    """Percentage of raster cells observed empty at least once."""
    total = result.coverage.size
    if total == 0:
        return 0.0
    covered = int((result.coverage > 0).sum())
    return 100.0 * covered / total


def _suffix_path(base: Path, camera_id: str, pose_id: str) -> Path:
    """Insert a ``__{camera_id}__{pose_id}`` suffix before ``base``'s extension."""
    return base.with_name(f"{base.stem}__{camera_id}__{pose_id}{base.suffix}")


def _suffix_coverage(base: Path | None, camera_id: str, pose_id: str) -> Path | None:
    """Suffix an optional coverage path the same way as the orthophoto path."""
    if base is None:
        return None
    return _suffix_path(base, camera_id, pose_id)


@cleanplate_app.command("reconstruct")
def reconstruct_command(
    run_dir: Path = typer.Option(
        ..., help="Survey frames root containing {run_id}/{camera_id}/frames/*.yaml"
    ),
    run_id: str = typer.Option(..., help="Survey run id to reconstruct"),
    output: Path = typer.Option(..., help="Output orthophoto path (.png/.jpg/.tif)"),
    x_min: float = typer.Option(..., help="Raster min world X (East), meters"),
    x_max: float = typer.Option(..., help="Raster max world X (East), meters"),
    y_min: float = typer.Option(..., help="Raster min world Y (North), meters"),
    y_max: float = typer.Option(..., help="Raster max world Y (North), meters"),
    pixels_per_meter: float = typer.Option(..., help="Raster resolution (cells per meter)"),
    camera_id: str | None = typer.Option(None, help="Restrict to a single camera id"),
    pose_id: str | None = typer.Option(
        None, help="Restrict to a single pose id; if omitted, process every group"
    ),
    method: str = typer.Option("median", help="Per-cell reduction: 'median' or 'mode'"),
    photometric: bool = typer.Option(
        True, "--photometric/--no-photometric", help="Median-gray normalize each frame"
    ),
    coverage_output: Path | None = typer.Option(
        None, help="Optional coverage raster output path (.tif only)"
    ),
) -> None:
    """
    Reconstruct an empty-floor orthophoto from a persisted survey run.

    Loads the survey run, groups frames by ``(camera_id, pose_id)``, and for each
    selected group fuses the frames into a hole-free orthophoto on the requested
    :class:`GroundRaster`, writing it via :func:`write_clean_plate`.

    When more than one group is processed (``--pose-id`` omitted), one orthophoto
    is written PER group using a suffixed filename derived from ``--output``:
    ``{stem}__{camera_id}__{pose_id}{ext}`` (the coverage path, if given, is
    suffixed identically). With a single selected group the bare ``--output`` and
    ``--coverage-output`` paths are used as-is.
    """
    dataset = CleanPlateDataset.from_survey_run(run_dir, run_id, camera_id=camera_id)
    raster = _build_raster(x_min, x_max, y_min, y_max, pixels_per_meter)

    keys = list(dataset.groups().keys())
    if pose_id is not None:
        keys = [k for k in keys if k[1] == pose_id]
    if camera_id is not None:
        keys = [k for k in keys if k[0] == camera_id]

    if not keys:
        typer.echo(
            f"Error: No clean-plate groups found for run {run_id!r} "
            "(check --run-dir/--run-id/--camera-id/--pose-id)",
            err=True,
        )
        raise typer.Exit(1)

    multi = len(keys) > 1
    for group_camera_id, group_pose_id in keys:
        frames = dataset.frames_for(group_camera_id, group_pose_id)
        if not frames:
            typer.echo(
                f"Warning: group ({group_camera_id}, {group_pose_id}) has no "
                "loadable frames; skipping",
                err=True,
            )
            continue

        result = reconstruct_clean_plate(frames, raster, method=method, photometric=photometric)

        ortho_path = _suffix_path(output, group_camera_id, group_pose_id) if multi else output
        cov_path = (
            _suffix_coverage(coverage_output, group_camera_id, group_pose_id)
            if multi
            else coverage_output
        )
        write_clean_plate(result, ortho_path, cov_path)

        typer.echo(
            f"group ({group_camera_id}, {group_pose_id}): frames={len(frames)} "
            f"coverage={_coverage_percent(result):.1f}% -> {ortho_path}"
        )
        if cov_path is not None:
            typer.echo(f"  coverage raster -> {cov_path}")

    typer.echo(f"Done: {len(keys)} group(s) processed.")


@cleanplate_app.command("synth")
def synth_command(
    output: Path = typer.Option(..., help="Output orthophoto path (.png/.jpg/.tif)"),
    coverage_output: Path | None = typer.Option(
        None, help="Optional coverage raster output path (.tif only)"
    ),
    truth_output: Path | None = typer.Option(
        None, help="Optional ground-truth clean-background output path (.png/.jpg/.tif)"
    ),
    n_visits: int = typer.Option(4, help="Number of synthetic visits/poses to render"),
    seed: int = typer.Option(0, help="Deterministic RNG seed"),
    x_min: float = typer.Option(0.0, help="Raster min world X (East), meters"),
    x_max: float = typer.Option(6.0, help="Raster max world X (East), meters"),
    y_min: float = typer.Option(0.0, help="Raster min world Y (North), meters"),
    y_max: float = typer.Option(6.0, help="Raster max world Y (North), meters"),
    pixels_per_meter: float = typer.Option(16.0, help="Raster resolution (cells per meter)"),
    image_width: int = typer.Option(96, help="Rendered frame width in pixels"),
    image_height: int = typer.Option(96, help="Rendered frame height in pixels"),
    method: str = typer.Option("median", help="Per-cell reduction: 'median' or 'mode'"),
) -> None:
    """
    Emit a synthetic multi-visit clean-plate prototype and self-validate it.

    Builds a deterministic synthetic dataset via :func:`make_synthetic_visits`,
    reconstructs the empty-floor orthophoto, writes it (and optionally the
    coverage raster and the ground-truth clean background), then echoes the MAE
    between reconstruction and ground truth over cells observed empty at least
    once. Output is fully deterministic given ``--seed``.
    """
    raster = _build_raster(x_min, x_max, y_min, y_max, pixels_per_meter)
    truth, frames = make_synthetic_visits(
        n_visits=n_visits,
        raster=raster,
        image_size=(image_height, image_width),
        seed=seed,
    )

    result = reconstruct_clean_plate(frames, raster, method=method, photometric=True)
    write_clean_plate(result, output, coverage_output)

    if truth_output is not None:
        _write_truth(truth, truth_output, result)

    mae = _masked_mae(result.orthophoto, truth, result.coverage > 0)
    typer.echo(
        f"synth: visits={n_visits} seed={seed} "
        f"coverage={_coverage_percent(result):.1f}% MAE={mae:.4f} -> {output}"
    )
    if coverage_output is not None:
        typer.echo(f"  coverage raster -> {coverage_output}")
    if truth_output is not None:
        typer.echo(f"  ground truth -> {truth_output}")


def _write_truth(truth: np.ndarray, truth_output: Path, result: CleanPlateResult) -> None:
    """Persist the ground-truth background by reusing :func:`write_clean_plate`."""
    truth_result = result.__class__(
        orthophoto=truth,
        coverage=result.coverage,
        inpainted_mask=result.inpainted_mask,
        raster=result.raster,
    )
    write_clean_plate(truth_result, truth_output, None)


def _masked_mae(reconstruction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error between two uint8 rasters over ``mask`` cells."""
    if not mask.any():
        return float("nan")
    diff = np.abs(reconstruction[mask].astype(np.float64) - truth[mask].astype(np.float64))
    return float(diff.mean())
