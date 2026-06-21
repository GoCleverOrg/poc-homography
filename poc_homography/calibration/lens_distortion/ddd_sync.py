"""Sync legacy CameraCalibrationTable to the DDD LensCalibrationTable repo."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from poc_homography.calibration.lens_distortion.calibration_table import (
        CameraCalibrationTable,
    )
    from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable

logger = logging.getLogger(__name__)


def lens_table_to_camera_table(lens_table: LensCalibrationTable) -> CameraCalibrationTable:
    """Convert a DDD :class:`LensCalibrationTable` to the legacy solver table.

    The extrinsic solvers consume the legacy
    :class:`~poc_homography.calibration.lens_distortion.calibration_table.CameraCalibrationTable`
    (flat distortion fields, zoom-keyed dict), whereas the calibration repos yield
    the DDD :class:`~poc_homography.domain.entities.lens_calibration_table.LensCalibrationTable`
    (nested :class:`LensDistortion` value object). The two ``to_dict`` shapes are
    NOT interchangeable, so this rebuilds the legacy table field-by-field — the
    in-memory inverse of :func:`sync_to_ddd_repo`, co-located with it so the
    bidirectional bridge stays in one place.

    Args:
        lens_table: The persisted per-zoom DDD lens table.

    Returns:
        An equivalent :class:`CameraCalibrationTable` the extrinsic solvers accept.
    """
    from poc_homography.calibration.lens_distortion.calibration_table import (
        CameraCalibrationTable,
        ZoomCalibrationEntry,
    )

    table = CameraCalibrationTable(
        camera_id=lens_table.id,
        created_date=lens_table.created_date,
        last_modified=lens_table.last_modified,
    )
    for entry in lens_table.entries:
        distortion = entry.distortion
        table.entries[float(entry.zoom_factor)] = ZoomCalibrationEntry(
            zoom_factor=float(entry.zoom_factor),
            k1=float(distortion.k1),
            k2=float(distortion.k2),
            k3=float(distortion.k3),
            p1=float(distortion.p1),
            p2=float(distortion.p2),
            calibration_date=entry.calibration_date,
            source_images=tuple(entry.source_images),
            validation_rmse=entry.validation_rmse,
            num_lines_used=entry.num_lines_used,
            fx=float(entry.fx),
            fy=float(entry.fy),
            cx=float(entry.cx),
            cy=float(entry.cy),
            reprojection_error_px=entry.reprojection_error_px,
        )
    return table


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[2]  # fallback: assume standard layout


DEFAULT_DATA_DIR = _find_project_root() / "data" / "lens_calibrations"


def sync_to_ddd_repo(
    table: CameraCalibrationTable,
    data_dir: Path | None = None,
) -> None:
    """Persist a legacy CameraCalibrationTable into the DDD YAML repo.

    Merges with any existing entries for the same camera_id so that
    multi-zoom calibration accumulates over time.
    """
    from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
    from poc_homography.domain.vo.lens_distortion import LensDistortion
    from poc_homography.domain.vo.zoom_calibration_entry import (
        ZoomCalibrationEntry as DddEntry,
    )
    from poc_homography.infrastructure.repositories.repo_yaml_lens_calibration_table import (
        RepoYamlLensCalibrationTable,
    )
    from poc_homography.types import PixelsFloat, Unitless

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    repo = RepoYamlLensCalibrationTable(data_dir)

    existing = repo.get(table.camera_id)
    existing_by_zoom: dict[float, DddEntry] = {}
    if existing:
        for e in existing.entries:
            existing_by_zoom[float(e.zoom_factor)] = e

    for legacy_entry in table.entries.values():
        ddd_entry = DddEntry(
            zoom_factor=Unitless(legacy_entry.zoom_factor),
            distortion=LensDistortion(
                k1=Unitless(legacy_entry.k1),
                k2=Unitless(legacy_entry.k2),
                p1=Unitless(legacy_entry.p1),
                p2=Unitless(legacy_entry.p2),
                k3=Unitless(legacy_entry.k3),
            ),
            calibration_date=legacy_entry.calibration_date,
            source_images=tuple(legacy_entry.source_images),
            validation_rmse=legacy_entry.validation_rmse,
            num_lines_used=legacy_entry.num_lines_used,
            fx=PixelsFloat(legacy_entry.fx),
            fy=PixelsFloat(legacy_entry.fy),
            cx=PixelsFloat(legacy_entry.cx),
            cy=PixelsFloat(legacy_entry.cy),
            reprojection_error_px=legacy_entry.reprojection_error_px,
        )
        existing_by_zoom[float(legacy_entry.zoom_factor)] = ddd_entry

    now = datetime.now().isoformat()
    entity = LensCalibrationTable(
        id=table.camera_id,
        entries=tuple(existing_by_zoom[z] for z in sorted(existing_by_zoom)),
        created_date=existing.created_date if existing else now,
        last_modified=now,
    )
    repo.save(entity)
    logger.info(f"Synced calibration to DDD repo: {data_dir / f'{table.camera_id}.yaml'}")


def sync_to_ddd_repo_pg(
    table: CameraCalibrationTable,
    session: Session,
) -> None:
    """Persist a legacy CameraCalibrationTable into the DDD PostgreSQL repo.

    Merges with any existing entries for the same camera_id so that
    multi-zoom calibration accumulates over time.
    """
    from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
    from poc_homography.domain.vo.lens_distortion import LensDistortion
    from poc_homography.domain.vo.zoom_calibration_entry import (
        ZoomCalibrationEntry as DddEntry,
    )
    from poc_homography.infrastructure.repositories import RepoPostgresLensCalibrationTable
    from poc_homography.types import PixelsFloat, Unitless

    repo = RepoPostgresLensCalibrationTable(session)

    existing = repo.get(table.camera_id)
    existing_by_zoom: dict[float, DddEntry] = {}
    if existing:
        for e in existing.entries:
            existing_by_zoom[float(e.zoom_factor)] = e

    for legacy_entry in table.entries.values():
        ddd_entry = DddEntry(
            zoom_factor=Unitless(legacy_entry.zoom_factor),
            distortion=LensDistortion(
                k1=Unitless(legacy_entry.k1),
                k2=Unitless(legacy_entry.k2),
                p1=Unitless(legacy_entry.p1),
                p2=Unitless(legacy_entry.p2),
                k3=Unitless(legacy_entry.k3),
            ),
            calibration_date=legacy_entry.calibration_date,
            source_images=tuple(legacy_entry.source_images),
            validation_rmse=legacy_entry.validation_rmse,
            num_lines_used=legacy_entry.num_lines_used,
            fx=PixelsFloat(legacy_entry.fx),
            fy=PixelsFloat(legacy_entry.fy),
            cx=PixelsFloat(legacy_entry.cx),
            cy=PixelsFloat(legacy_entry.cy),
            reprojection_error_px=legacy_entry.reprojection_error_px,
        )
        existing_by_zoom[float(legacy_entry.zoom_factor)] = ddd_entry

    now = datetime.now().isoformat()
    entity = LensCalibrationTable(
        id=table.camera_id,
        entries=tuple(existing_by_zoom[z] for z in sorted(existing_by_zoom)),
        created_date=existing.created_date if existing else now,
        last_modified=now,
    )
    repo.save(entity)
    logger.info("Synced calibration to PostgreSQL repo for camera: %s", table.camera_id)
