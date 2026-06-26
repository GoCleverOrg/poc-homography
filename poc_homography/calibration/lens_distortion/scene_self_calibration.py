"""Per-zoom scene self-calibration service.

This service seeds a prior camera intrinsic matrix ``K`` from camera-spec
defaults (per zoom group), detects floor lines, jointly refines lens
distortion plus intrinsics off that prior using the *existing*
:class:`DistortionSolver`, and assembles one calibration entry per zoom into a
:class:`LensCalibrationTable`.

It reuses the existing solver, the domain-VO calibration entry/table, and the
YAML persistence layer. No new solver, table, or persistence is introduced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from poc_homography.calibration.lens_distortion.distortion_solver import (
    SolverConfig,
)
from poc_homography.calibration.lens_distortion.floor_line_curation import (
    curate_floor_lines,
)
from poc_homography.calibration.lens_distortion.line_detection import LineDetector
from poc_homography.calibration.lens_distortion.models import PTZPosition
from poc_homography.calibration.lens_distortion.robust_distortion import solve_robust
from poc_homography.camera.intrinsics import compute_intrinsics
from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry
from poc_homography.types import Degrees, PixelsFloat, Unitless

if TYPE_CHECKING:
    import numpy as np

    from poc_homography.calibration.lens_distortion.models import CameraLine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenePerZoomConfig:
    """Tunable knobs for per-zoom scene self-calibration.

    Attributes:
        min_lines: Minimum usable (curved) lines required to calibrate a zoom.
        max_lines_per_image: Cap on candidate lines kept per detected image.
        fx_refine_frac: Fractional window (+/-) around the prior fx/fy.
        cx_window_px: Half-window (pixels) around the prior cx.
        cy_window_px: Half-window (pixels) around the prior cy.
        num_samples_per_line: Points sampled along each line by the solver.
        max_iterations: Maximum solver optimisation iterations.
        zoom_tolerance: Rounding tolerance used to group images by zoom.
        optimize_intrinsics: When False (default) the prior ``K`` is held FIXED
            during the distortion solve, exactly like the proven plumb-line
            recipe (``distort_plumb.py``: fixed fx/fy, principal point at the
            image centre). This is what keeps k1 from pegging: letting fx/fy
            float on a near-degenerate floor scene lets the optimiser trade
            distortion against focal length and run to a coefficient rail.
        k1_bounds: (min, max) bounds for the radial k1 coefficient. Wide and
            symmetric so a correct barrel solve (~-0.18) sits well inside the
            range and never pegs; the *fixed K* (above) is what prevents
            run-away, not a tight box.
        k2_bounds: (min, max) bounds for the radial k2 coefficient.
        k3_bounds: (min, max) bounds for the radial k3 coefficient.
        reject_min_px: Floor for the robust per-line rejection threshold.
        reject_mad_scale: MAD multiplier added to the median per-line RMSE to
            form the robust rejection threshold (max(reject_min_px, med+scale*MAD)).
        robust_min_lines: Minimum lines the robust loop will keep before it
            stops rejecting (must stay >= min_lines).
        curation_angle_tol_deg: Chord-angle tolerance for floor-line orientation
            families during curation.
        curation_min_family_fraction: Minimum length-weighted fraction a chord
            angle bucket must hold to count as a dominant family.
    """

    min_lines: int = 3
    max_lines_per_image: int = 20
    fx_refine_frac: float = 0.05
    cx_window_px: float = 10.0
    cy_window_px: float = 10.0
    num_samples_per_line: int = 30
    max_iterations: int = 3000
    zoom_tolerance: float = 0.1
    # Hold the prior K fixed during the solve (the plumb-line recipe). This,
    # not a tight coefficient box, is what stops k1 pegging.
    optimize_intrinsics: bool = False
    # Wide, symmetric radial bounds: a correct barrel solve (~-0.18) is well
    # interior, so the optimiser never sits on a rail. With K fixed the solve
    # is well-posed and stays inside these on real floor lines.
    k1_bounds: tuple[float, float] = (-0.8, 0.8)
    k2_bounds: tuple[float, float] = (-0.5, 0.5)
    k3_bounds: tuple[float, float] = (-0.15, 0.15)
    # Robust IRLS rejection knobs.
    reject_min_px: float = 2.0
    reject_mad_scale: float = 3.0
    robust_min_lines: int = 6
    # Floor-line curation knobs.
    curation_angle_tol_deg: float = 12.0
    curation_min_family_fraction: float = 0.15


@dataclass(frozen=True)
class SkippedZoom:
    """A zoom group that could not be calibrated.

    Attributes:
        zoom_factor: The (rounded) zoom factor of the skipped group.
        num_lines: Number of usable lines available for the group.
        reason: Human-readable explanation for the skip.
    """

    zoom_factor: float
    num_lines: int
    reason: str
    num_detected: int = 0
    num_curated: int = 0
    num_kept: int = 0
    num_rejected: int = 0


@dataclass(frozen=True)
class SceneCalibrationResult:
    """Result of a per-zoom scene self-calibration run.

    Attributes:
        table: The assembled lens calibration table (one entry per zoom).
        skipped_zooms: Zoom groups that were skipped (too few lines / failure).
    """

    table: LensCalibrationTable
    skipped_zooms: tuple[SkippedZoom, ...]


def calibrate_zoom_from_lines(
    lines: list[CameraLine],
    zoom: float,
    *,
    camera_spec: CameraSpec,
    config: ScenePerZoomConfig,
) -> ZoomCalibrationEntry | SkippedZoom:
    """Calibrate a single zoom group from already-detected camera lines.

    Seeds a prior ``K`` via :func:`compute_intrinsics` for *zoom*, filters the
    lines by edge curvature, and refines distortion plus intrinsics (modest
    fx/fy window, tight cx/cy window) using the existing solver.

    Args:
        lines: Camera lines detected for this zoom group.
        zoom: Zoom factor for the group.
        camera_spec: Camera specification supplying sensor defaults.
        config: Per-zoom configuration knobs.

    Returns:
        A :class:`ZoomCalibrationEntry` on success, otherwise a
        :class:`SkippedZoom` describing why calibration was not performed.
    """
    intrinsics = compute_intrinsics(
        zoom=Unitless(zoom),
        image_width=camera_spec.image_width,
        image_height=camera_spec.image_height,
        sensor_width_mm=camera_spec.sensor_width,
        base_focal_length_mm=camera_spec.base_focal_length,
    )
    prior_k = intrinsics.K
    num_detected = len(lines)

    # 1) Curate to dominant floor-line orientation families: strips scattered
    #    clutter and off-family car edges before any solving.
    curated = curate_floor_lines(
        lines,
        float(camera_spec.image_width),
        float(camera_spec.image_height),
        angle_tol_deg=config.curation_angle_tol_deg,
        min_family_fraction=config.curation_min_family_fraction,
        min_lines=config.min_lines,
    )

    # 2) Drop lines that carry no distortion signal (straight by construction).
    usable_lines = [line for line in curated if line.has_edge_curvature()]
    num_usable = len(usable_lines)
    num_curated = len(curated)
    if num_usable < config.min_lines:
        return SkippedZoom(
            zoom_factor=zoom,
            num_lines=num_usable,
            reason=(
                f"only {num_usable} usable line(s) with edge curvature; need >= {config.min_lines}"
            ),
            num_detected=num_detected,
            num_curated=num_curated,
        )

    fx0 = float(prior_k[0, 0])
    fy0 = float(prior_k[1, 1])
    cx0 = float(prior_k[0, 2])
    cy0 = float(prior_k[1, 2])
    frac = config.fx_refine_frac

    solver_config = SolverConfig(
        use_radial_only=True,
        # Fixed K (default) mirrors the proven plumb-line recipe and is what
        # keeps the radial solve from pegging; see ScenePerZoomConfig.
        optimize_intrinsics=config.optimize_intrinsics,
        k1_bounds=config.k1_bounds,
        k2_bounds=config.k2_bounds,
        k3_bounds=config.k3_bounds,
        fx_bounds=(fx0 * (1.0 - frac), fx0 * (1.0 + frac)),
        fy_bounds=(fy0 * (1.0 - frac), fy0 * (1.0 + frac)),
        cx_bounds=(cx0 - config.cx_window_px, cx0 + config.cx_window_px),
        cy_bounds=(cy0 - config.cy_window_px, cy0 + config.cy_window_px),
        num_samples_per_line=config.num_samples_per_line,
        max_iterations=config.max_iterations,
    )

    # 3) Robust IRLS solve: reject distortion-inconsistent lines (car edges).
    #    Keep the robust floor at <= num_usable so a thin scene still solves.
    robust_min = min(config.robust_min_lines, num_usable)
    robust_min = max(robust_min, config.min_lines)
    try:
        robust = solve_robust(
            usable_lines,
            prior_k,
            solver_config,
            initial_guess=LensDistortion(),
            min_lines=robust_min,
            reject_min_px=config.reject_min_px,
            reject_mad_scale=config.reject_mad_scale,
        )
    except ValueError as exc:
        return SkippedZoom(
            zoom_factor=zoom,
            num_lines=num_usable,
            reason=f"solver error: {exc}",
            num_detected=num_detected,
            num_curated=num_curated,
        )

    result = robust.result
    num_kept = len(robust.kept_lines)
    num_rejected = len(robust.rejected_lines)

    if not result.success:
        return SkippedZoom(
            zoom_factor=zoom,
            num_lines=num_usable,
            reason=f"solver did not converge: {result.message}",
            num_detected=num_detected,
            num_curated=num_curated,
            num_kept=num_kept,
            num_rejected=num_rejected,
        )

    refined = result.intrinsics or {"fx": fx0, "fy": fy0, "cx": cx0, "cy": cy0}
    source_images = tuple(dict.fromkeys(line.image_path for line in robust.kept_lines))

    logger.info(
        "zoom %.3f: detected=%d curated=%d kept=%d rejected=%d k1=%.4f",
        zoom,
        num_detected,
        num_curated,
        num_kept,
        num_rejected,
        float(result.distortion.k1),
    )

    return ZoomCalibrationEntry(
        zoom_factor=Unitless(zoom),
        distortion=result.distortion,
        calibration_date=datetime.now().isoformat(),
        source_images=source_images,
        validation_rmse=result.overall_rmse,
        num_lines_used=num_kept,
        fx=PixelsFloat(refined["fx"]),
        fy=PixelsFloat(refined["fy"]),
        cx=PixelsFloat(refined["cx"]),
        cy=PixelsFloat(refined["cy"]),
    )


def calibrate_scene_self(
    image_zoom_pairs: list[tuple[np.ndarray, float]],
    *,
    camera_id: str,
    camera_spec: CameraSpec = CameraSpec.HIKVISION_DS_2DF8425IX,
    config: ScenePerZoomConfig | None = None,
    detector: LineDetector | None = None,
) -> SceneCalibrationResult:
    """Self-calibrate a scene per zoom group from (image, zoom) pairs.

    Groups the pairs by rounded zoom (``round(z / tol) * tol``), detects lines
    in every image of each group, converts them to camera lines, and refines
    distortion plus intrinsics per zoom. Produces a single
    :class:`LensCalibrationTable` with one entry per successfully calibrated
    zoom and reports skipped zooms separately.

    Args:
        image_zoom_pairs: List of ``(image_array, zoom_factor)`` tuples.
        camera_id: Camera identifier used as the table id.
        camera_spec: Camera specification supplying sensor defaults.
        config: Per-zoom configuration (defaults applied when None).
        detector: Line detector (a fresh :class:`LineDetector` when None).

    Returns:
        A :class:`SceneCalibrationResult` carrying the table and skipped zooms.
    """
    cfg = config or ScenePerZoomConfig()
    line_detector = detector or LineDetector()
    tol = cfg.zoom_tolerance

    # Group (image, zoom) pairs by rounded zoom, mirroring group_images_by_zoom.
    # Canonicalise the rounded key (round(z/tol)*tol can yield e.g.
    # 1.2000000000000002) so the persisted zoom_factor and image-path strings
    # stay clean.
    groups: dict[float, list[np.ndarray]] = {}
    for image, zoom in image_zoom_pairs:
        rounded = round(round(zoom / tol) * tol, 10)
        groups.setdefault(rounded, []).append(image)

    entries: list[ZoomCalibrationEntry] = []
    skipped: list[SkippedZoom] = []

    for zoom in sorted(groups):
        # A non-positive rounded zoom (e.g. an input zoom that rounds to 0.0)
        # is degenerate: PTZPosition / compute_intrinsics reject it. Report it
        # as skipped rather than letting it abort the whole run.
        if zoom <= 0.0:
            logger.warning("Skipping zoom %.3f: non-positive zoom factor after rounding", zoom)
            skipped.append(
                SkippedZoom(
                    zoom_factor=zoom,
                    num_lines=0,
                    reason="non-positive zoom factor after rounding",
                )
            )
            continue

        ptz = PTZPosition(pan_deg=Degrees(0.0), tilt_deg=Degrees(0.0), zoom_factor=zoom)
        lines: list[CameraLine] = []
        for idx, image in enumerate(groups[zoom]):
            candidates = line_detector.detect(image)
            image_path = f"{camera_id}_zoom{zoom}_img{idx}"
            # Keep up to max_lines_per_image *usable* (curved) lines: the cap
            # must not discard the curved lines that carry the distortion
            # signal in favour of straight ones.
            kept = 0
            for line_idx, candidate in enumerate(candidates):
                if kept >= cfg.max_lines_per_image:
                    break
                camera_line = candidate.to_camera_line(
                    line_id=f"{image_path}_line{line_idx}",
                    image_path=image_path,
                    ptz_position=ptz,
                )
                if not camera_line.has_edge_curvature():
                    continue
                lines.append(camera_line)
                kept += 1

        outcome = calibrate_zoom_from_lines(lines, zoom, camera_spec=camera_spec, config=cfg)
        if isinstance(outcome, SkippedZoom):
            logger.warning(
                "Skipping zoom %.3f: %s (%d usable line(s))",
                outcome.zoom_factor,
                outcome.reason,
                outcome.num_lines,
            )
            skipped.append(outcome)
        else:
            entries.append(outcome)

    now = datetime.now().isoformat()
    table = LensCalibrationTable(
        id=camera_id,
        entries=tuple(entries),
        created_date=now,
        last_modified=now,
    )
    return SceneCalibrationResult(table=table, skipped_zooms=tuple(skipped))
