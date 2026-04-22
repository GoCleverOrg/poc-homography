"""One-time YAML -> Postgres data migration script.

Reads all entities from YAML repositories and writes them to PostgreSQL,
respecting referential-integrity order.  Idempotent: skips entities that
already exist in the target database.

Usage::

    python scripts/migrate_yaml_to_postgres.py --data-dir data/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports resolve when invoked as
# ``python scripts/migrate_yaml_to_postgres.py``.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Also add the webapp directory for session model imports.
_WEBAPP_DIR = _PROJECT_ROOT / "webapp"
if str(_WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(_WEBAPP_DIR))

# ---------------------------------------------------------------------------
# Third-party / project imports (after path setup)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from poc_homography.infrastructure.database import get_session  # noqa: E402

# -- YAML repositories -----------------------------------------------------
from poc_homography.infrastructure.repositories import (  # noqa: E402
    RepoYamlAnnotation,
    RepoYamlCalibrationLineTraceSet,
    RepoYamlCameraCalibration,
    RepoYamlCameraConfig,
    RepoYamlCapturedFrame,
    RepoYamlDiagnosticSession,
    RepoYamlGroundControlPoint,
    RepoYamlLensCalibrationTable,
    RepoYamlLine,
    RepoYamlLineAnnotation,
    RepoYamlMap,
    RepoYamlStressTestSession,
    RepoYamlSurveySession,
    RepoYamlTenant,
)

# -- Postgres repositories --------------------------------------------------
from poc_homography.infrastructure.repositories.repo_postgres_annotation import (  # noqa: E402
    RepoPostgresAnnotation,
)
from poc_homography.infrastructure.repositories.repo_postgres_calibration_line_trace_set import (  # noqa: E402
    RepoPostgresCalibrationLineTraceSet,
)
from poc_homography.infrastructure.repositories.repo_postgres_camera_calibration import (  # noqa: E402
    RepoPostgresCameraCalibration,
)
from poc_homography.infrastructure.repositories.repo_postgres_camera_config import (  # noqa: E402
    RepoPostgresCameraConfig,
)
from poc_homography.infrastructure.repositories.repo_postgres_captured_frame import (  # noqa: E402
    RepoPostgresCapturedFrame,
)
from poc_homography.infrastructure.repositories.repo_postgres_diagnostic_session import (  # noqa: E402
    RepoPostgresDiagnosticSession,
)
from poc_homography.infrastructure.repositories.repo_postgres_ground_control_point import (  # noqa: E402
    RepoPostgresGroundControlPoint,
)
from poc_homography.infrastructure.repositories.repo_postgres_lens_calibration_table import (  # noqa: E402
    RepoPostgresLensCalibrationTable,
)
from poc_homography.infrastructure.repositories.repo_postgres_line import (  # noqa: E402
    RepoPostgresLine,
)
from poc_homography.infrastructure.repositories.repo_postgres_line_annotation import (  # noqa: E402
    RepoPostgresLineAnnotation,
)
from poc_homography.infrastructure.repositories.repo_postgres_map import (  # noqa: E402
    RepoPostgresMap,
)
from poc_homography.infrastructure.repositories.repo_postgres_stress_test_session import (  # noqa: E402
    RepoPostgresStressTestSession,
)
from poc_homography.infrastructure.repositories.repo_postgres_survey_session import (  # noqa: E402
    RepoPostgresSurveySession,
)
from poc_homography.infrastructure.repositories.repo_postgres_tenant import (  # noqa: E402
    RepoPostgresTenant,
)

# -- Session entity factories (webapp-layer models) -------------------------
from camera_diagnostic.models import DiagnosticSession, StressTestSession  # noqa: E402
from camera_survey.models import SurveySession  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data-dir subdirectory names
# ---------------------------------------------------------------------------
# These match the conventional layout under ``data/``.
SUBDIR_TENANTS = "tenants"
SUBDIR_MAPS = "maps"
SUBDIR_CAMERAS = "cameras"
SUBDIR_LENS_CALIBRATIONS = "lens_calibrations"
SUBDIR_CAPTURED_FRAMES = "captured_frames"
SUBDIR_ANNOTATIONS = "annotations"
SUBDIR_GCPS = "gcps"
SUBDIR_LINES = "lines"
SUBDIR_LINE_ANNOTATIONS = "line_annotations"
SUBDIR_CALIBRATION_LINE_TRACES = "calibration_line_traces"
SUBDIR_DIAGNOSTIC = "diagnostic"
SUBDIR_STRESS_TEST = "stress_test"
SUBDIR_SURVEY = "survey"


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------

class _MigrationStats:
    """Accumulates per-entity-type migration counters."""

    def __init__(self) -> None:
        self.migrated: int = 0
        self.skipped: int = 0
        self.errors: int = 0


def _migrate_domain_entities(
    *,
    label: str,
    yaml_repo: Any,
    pg_repo: Any,
) -> _MigrationStats:
    """Migrate entities that follow the standard RepoYaml / RepoPostgres pattern.

    Both repos must expose ``get_all() -> list``, ``exists(id) -> bool``,
    and ``save(entity) -> None``.
    """
    stats = _MigrationStats()
    entities: list[Any] = yaml_repo.get_all()
    logger.info("[%s] Found %d entities in YAML", label, len(entities))

    for entity in entities:
        try:
            if pg_repo.exists(entity.id):
                stats.skipped += 1
                continue
            pg_repo.save(entity)
            stats.migrated += 1
        except Exception:
            stats.errors += 1
            logger.exception("[%s] Failed to migrate entity %s", label, entity.id)

    return stats


def _migrate_session_entities(
    *,
    label: str,
    yaml_repo: Any,
    pg_repo: Any,
) -> _MigrationStats:
    """Migrate date-partitioned session entities.

    Session YAML repos use ``get_all(limit=...) -> (list, total)`` with a
    high limit to read everything.
    """
    stats = _MigrationStats()

    # Fetch all sessions (use a very large limit to get everything).
    sessions, total = yaml_repo.get_all(limit=100_000)
    logger.info("[%s] Found %d sessions in YAML", label, total)

    for entity in sessions:
        try:
            if pg_repo.exists(entity.id):
                stats.skipped += 1
                continue
            pg_repo.save(entity)
            stats.migrated += 1
        except Exception:
            stats.errors += 1
            logger.exception("[%s] Failed to migrate session %s", label, entity.id)

    return stats


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------

def run_migration(data_dir: Path) -> dict[str, _MigrationStats]:
    """Execute the full YAML -> Postgres migration.

    Returns a dict mapping entity label to its migration stats.
    """
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        logger.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)

    results: dict[str, _MigrationStats] = {}

    with get_session() as session:
        # ==================================================================
        # 1. Tenants (no FK deps)
        # ==================================================================
        tenants_dir = data_dir / SUBDIR_TENANTS
        if tenants_dir.is_dir():
            results["Tenants"] = _migrate_domain_entities(
                label="Tenants",
                yaml_repo=RepoYamlTenant(tenants_dir),
                pg_repo=RepoPostgresTenant(session),
            )
        else:
            logger.warning("Skipping Tenants: directory %s not found", tenants_dir)

        # ==================================================================
        # 2. Maps (FK: tenant_id -> tenants)
        # ==================================================================
        maps_dir = data_dir / SUBDIR_MAPS
        if maps_dir.is_dir():
            results["Maps"] = _migrate_domain_entities(
                label="Maps",
                yaml_repo=RepoYamlMap(maps_dir),
                pg_repo=RepoPostgresMap(session),
            )
        else:
            logger.warning("Skipping Maps: directory %s not found", maps_dir)

        # ==================================================================
        # 3. CameraConfigs (FK: tenant_id -> tenants, map_id -> maps)
        # ==================================================================
        cameras_dir = data_dir / SUBDIR_CAMERAS
        if cameras_dir.is_dir():
            results["CameraConfigs"] = _migrate_domain_entities(
                label="CameraConfigs",
                yaml_repo=RepoYamlCameraConfig(cameras_dir),
                pg_repo=RepoPostgresCameraConfig(session),
            )
        else:
            logger.warning("Skipping CameraConfigs: directory %s not found", cameras_dir)

        # ==================================================================
        # 4. CameraCalibrations (FK: id -> camera_configs)
        #    Note: shares the same data dir as camera configs in practice;
        #    the YAML repo reads from a dedicated "camera_calibrations"
        #    subdirectory if one exists, otherwise it may be absent.
        # ==================================================================
        camera_calibrations_dir = data_dir / "camera_calibrations"
        if camera_calibrations_dir.is_dir():
            results["CameraCalibrations"] = _migrate_domain_entities(
                label="CameraCalibrations",
                yaml_repo=RepoYamlCameraCalibration(camera_calibrations_dir),
                pg_repo=RepoPostgresCameraCalibration(session),
            )
        else:
            logger.info(
                "Skipping CameraCalibrations: directory %s not found (may not have data yet)",
                camera_calibrations_dir,
            )

        # ==================================================================
        # 5. LensCalibrationTables (FK: id -> camera_configs)
        # ==================================================================
        lens_dir = data_dir / SUBDIR_LENS_CALIBRATIONS
        if lens_dir.is_dir():
            results["LensCalibrationTables"] = _migrate_domain_entities(
                label="LensCalibrationTables",
                yaml_repo=RepoYamlLensCalibrationTable(lens_dir),
                pg_repo=RepoPostgresLensCalibrationTable(session),
            )
        else:
            logger.warning(
                "Skipping LensCalibrationTables: directory %s not found", lens_dir,
            )

        # ==================================================================
        # 6. CapturedFrames (FK: map_id -> maps)
        # ==================================================================
        frames_dir = data_dir / SUBDIR_CAPTURED_FRAMES
        if frames_dir.is_dir():
            results["CapturedFrames"] = _migrate_domain_entities(
                label="CapturedFrames",
                yaml_repo=RepoYamlCapturedFrame(frames_dir, create_dir=False),
                pg_repo=RepoPostgresCapturedFrame(session),
            )
        else:
            logger.warning(
                "Skipping CapturedFrames: directory %s not found", frames_dir,
            )

        # ==================================================================
        # 7. Annotations (FK: frame_id -> captured_frames)
        # ==================================================================
        annotations_dir = data_dir / SUBDIR_ANNOTATIONS
        if annotations_dir.is_dir():
            results["Annotations"] = _migrate_domain_entities(
                label="Annotations",
                yaml_repo=RepoYamlAnnotation(annotations_dir),
                pg_repo=RepoPostgresAnnotation(session),
            )
        else:
            logger.info(
                "Skipping Annotations: directory %s not found (may not have data yet)",
                annotations_dir,
            )

        # ==================================================================
        # 8. GroundControlPoints (FK: map_id -> maps)
        # ==================================================================
        gcps_dir = data_dir / SUBDIR_GCPS
        if gcps_dir.is_dir():
            results["GroundControlPoints"] = _migrate_domain_entities(
                label="GroundControlPoints",
                yaml_repo=RepoYamlGroundControlPoint(gcps_dir),
                pg_repo=RepoPostgresGroundControlPoint(session),
            )
        else:
            logger.warning(
                "Skipping GroundControlPoints: directory %s not found", gcps_dir,
            )

        # ==================================================================
        # 9. Lines (FK: map_id -> maps)
        # ==================================================================
        lines_dir = data_dir / SUBDIR_LINES
        if lines_dir.is_dir():
            results["Lines"] = _migrate_domain_entities(
                label="Lines",
                yaml_repo=RepoYamlLine(lines_dir),
                pg_repo=RepoPostgresLine(session),
            )
        else:
            logger.warning("Skipping Lines: directory %s not found", lines_dir)

        # ==================================================================
        # 10. LineAnnotations (FK: frame_id -> captured_frames)
        # ==================================================================
        line_ann_dir = data_dir / SUBDIR_LINE_ANNOTATIONS
        if line_ann_dir.is_dir():
            results["LineAnnotations"] = _migrate_domain_entities(
                label="LineAnnotations",
                yaml_repo=RepoYamlLineAnnotation(line_ann_dir),
                pg_repo=RepoPostgresLineAnnotation(session),
            )
        else:
            logger.warning(
                "Skipping LineAnnotations: directory %s not found", line_ann_dir,
            )

        # ==================================================================
        # 11. CalibrationLineTraceSets (no FK deps)
        # ==================================================================
        cal_traces_dir = data_dir / SUBDIR_CALIBRATION_LINE_TRACES
        if cal_traces_dir.is_dir():
            results["CalibrationLineTraceSets"] = _migrate_domain_entities(
                label="CalibrationLineTraceSets",
                yaml_repo=RepoYamlCalibrationLineTraceSet(cal_traces_dir),
                pg_repo=RepoPostgresCalibrationLineTraceSet(session),
            )
        else:
            logger.info(
                "Skipping CalibrationLineTraceSets: directory %s not found",
                cal_traces_dir,
            )

        # ==================================================================
        # 12. DiagnosticSessions (no FK deps)
        # ==================================================================
        diag_dir = data_dir / SUBDIR_DIAGNOSTIC
        if diag_dir.is_dir():
            results["DiagnosticSessions"] = _migrate_session_entities(
                label="DiagnosticSessions",
                yaml_repo=RepoYamlDiagnosticSession(
                    diag_dir, DiagnosticSession.from_dict,
                ),
                pg_repo=RepoPostgresDiagnosticSession(
                    session, DiagnosticSession.from_dict,
                ),
            )
        else:
            logger.warning(
                "Skipping DiagnosticSessions: directory %s not found", diag_dir,
            )

        # ==================================================================
        # 13. SurveySessions (no FK deps)
        # ==================================================================
        survey_dir = data_dir / SUBDIR_SURVEY
        if survey_dir.is_dir():
            results["SurveySessions"] = _migrate_session_entities(
                label="SurveySessions",
                yaml_repo=RepoYamlSurveySession(
                    survey_dir, SurveySession.from_dict,
                ),
                pg_repo=RepoPostgresSurveySession(
                    session, SurveySession.from_dict,
                ),
            )
        else:
            logger.info(
                "Skipping SurveySessions: directory %s not found (may not have data yet)",
                survey_dir,
            )

        # ==================================================================
        # 14. StressTestSessions (no FK deps)
        # ==================================================================
        stress_dir = data_dir / SUBDIR_STRESS_TEST
        if stress_dir.is_dir():
            results["StressTestSessions"] = _migrate_session_entities(
                label="StressTestSessions",
                yaml_repo=RepoYamlStressTestSession(
                    stress_dir, StressTestSession.from_dict,
                ),
                pg_repo=RepoPostgresStressTestSession(
                    session, StressTestSession.from_dict,
                ),
            )
        else:
            logger.warning(
                "Skipping StressTestSessions: directory %s not found", stress_dir,
            )

    return results


def _print_summary(results: dict[str, _MigrationStats], elapsed: float) -> None:
    """Print a human-readable migration summary."""
    total_migrated = 0
    total_skipped = 0
    total_errors = 0

    print("\n" + "=" * 60)
    print("  YAML -> Postgres Migration Summary")
    print("=" * 60)
    print(f"{'Entity':<30} {'Migrated':>9} {'Skipped':>9} {'Errors':>8}")
    print("-" * 60)

    for label, stats in results.items():
        print(
            f"{label:<30} {stats.migrated:>9} {stats.skipped:>9} {stats.errors:>8}"
        )
        total_migrated += stats.migrated
        total_skipped += stats.skipped
        total_errors += stats.errors

    print("-" * 60)
    print(
        f"{'TOTAL':<30} {total_migrated:>9} {total_skipped:>9} {total_errors:>8}"
    )
    print(f"\nCompleted in {elapsed:.1f}s")

    if total_errors:
        print(f"\n  WARNING: {total_errors} entities failed to migrate.")
        print("  Check the log output above for details.")
    else:
        print("\n  All entities migrated successfully.")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate YAML repository data to PostgreSQL.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/"),
        help="Root directory of the YAML data store (default: data/)",
    )
    args = parser.parse_args()

    logger.info("Starting YAML -> Postgres migration")
    logger.info("Data directory: %s", args.data_dir.resolve())

    t0 = time.monotonic()
    results = run_migration(args.data_dir)
    elapsed = time.monotonic() - t0

    _print_summary(results, elapsed)

    # Exit with non-zero code if any errors occurred.
    any_errors = any(s.errors > 0 for s in results.values())
    sys.exit(1 if any_errors else 0)


if __name__ == "__main__":
    main()
