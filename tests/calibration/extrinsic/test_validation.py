"""Offline tests for hold-out reprojection validation + multi-frame consolidation.

A known homography warps ortho lines into synthetic frame lines (optionally with
reproducible Gaussian endpoint noise); intrinsics come from a zero-distortion
:class:`CameraCalibrationTable`; a synthetic :class:`GeoTiff` converts map-pixel
reprojection error into ground meters. No live camera / MinIO / Postgres.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from poc_homography.calibration.extrinsic import (
    FrameCorrespondence,
    HoldoutValidationError,
    HoldoutValidationResult,
    MultiFrameConsolidationResult,
    ReprojectionStats,
    consolidate_ptz_frames,
    register_frame_lines,
    validate_with_holdout,
)
from poc_homography.calibration.lens_distortion.calibration_table import (
    CameraCalibrationTable,
    ZoomCalibrationEntry,
)
from poc_homography.calibration.lens_distortion.line_detection import CandidateLine
from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.domain.vo.ortho_line import OrthoLine

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Well-conditioned ORTHO map-pixel -> FRAME pixel warp (slight projective terms).
_H_KNOWN = np.array(
    [
        [1.10, 0.05, 40.0],
        [0.03, 1.20, 25.0],
        [1.0e-4, 5.0e-5, 1.0],
    ],
    dtype=np.float64,
)

# Ten varied ortho segments so a fit/hold-out split leaves both sets well posed.
_ORTHO_SEGMENTS: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((100.0, 100.0), (500.0, 110.0)),
    ((120.0, 150.0), (130.0, 600.0)),
    ((150.0, 200.0), (550.0, 580.0)),
    ((520.0, 160.0), (180.0, 560.0)),
    ((300.0, 120.0), (320.0, 640.0)),
    ((200.0, 300.0), (600.0, 340.0)),
    ((420.0, 220.0), (440.0, 700.0)),
    ((160.0, 420.0), (560.0, 470.0)),
    ((250.0, 500.0), (650.0, 250.0)),
    ((360.0, 180.0), (200.0, 620.0)),
]

# Ground sampling distance of 0.10 m/px => meters error == 0.10 * pixel error.
_GSD_M = 0.10


def _zero_distortion_table(camera_id: str = "cam1") -> CameraCalibrationTable:
    table = CameraCalibrationTable(camera_id=camera_id)
    table.add_entry(
        ZoomCalibrationEntry(
            zoom_factor=1.0,
            k1=0.0,
            k2=0.0,
            k3=0.0,
            p1=0.0,
            p2=0.0,
            calibration_date="2024-01-01T00:00:00",
            fx=1000.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0,
        )
    )
    return table


def _geotiff() -> GeoTiff:
    """North-up GeoTIFF with a 0.10 m/px ground sampling distance."""
    return GeoTiff(
        geotransform=GeoTransform(
            origin_easting=500000.0,
            pixel_width=_GSD_M,
            row_rotation=0.0,
            origin_northing=4000000.0,
            col_rotation=0.0,
            pixel_height=-_GSD_M,
        ),
        crs="EPSG:25830",
    )


def _ortho_lines() -> list[OrthoLine]:
    lines: list[OrthoLine] = []
    for i, (start, end) in enumerate(_ORTHO_SEGMENTS):
        lines.append(
            OrthoLine.from_endpoints(
                start_pixel=start,
                end_pixel=end,
                start_utm=(500000.0 + i, 4000000.0 + i),
                end_utm=(500010.0 + i, 4000010.0 + i),
            )
        )
    return lines


def _identity_seed(ortho_lines: list[OrthoLine]) -> dict[int, str]:
    return {i: f"ortho_{i}" for i in range(len(ortho_lines))}


def _warp_point(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    pt = np.array([[[point[0], point[1]]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, homography)[0, 0]
    return float(out[0]), float(out[1])


def _frame_lines_from_ortho(
    ortho_lines: list[OrthoLine],
    homography: np.ndarray,
    noise_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[CandidateLine]:
    """Project ortho map-pixel endpoints into frame pixels via H, with noise."""
    frame_lines: list[CandidateLine] = []
    for line in ortho_lines:
        start = _warp_point((float(line.start_pixel[0]), float(line.start_pixel[1])), homography)
        end = _warp_point((float(line.end_pixel[0]), float(line.end_pixel[1])), homography)
        if noise_px > 0.0 and rng is not None:
            start = (start[0] + rng.normal(0, noise_px), start[1] + rng.normal(0, noise_px))
            end = (end[0] + rng.normal(0, noise_px), end[1] + rng.normal(0, noise_px))
        frame_lines.append(CandidateLine(start=start, end=end))
    return frame_lines


def _rms_m_against_truth(
    homography: np.ndarray,
    ortho_lines: list[OrthoLine],
    true_frame_lines: list[CandidateLine],
    geotiff: GeoTiff,
) -> float:
    """RMS reprojection error (meters) of H over NOISE-FREE correspondences."""
    sq: list[float] = []
    for line, frame_line in zip(ortho_lines, true_frame_lines):
        for frame_pt, ortho_pt in (
            (frame_line.start, line.start_pixel),
            (frame_line.end, line.end_pixel),
        ):
            proj = _warp_point(frame_pt, homography)
            proj_e, proj_n = geotiff.pixel_to_geo(proj[0], proj[1])
            true_e, true_n = geotiff.pixel_to_geo(float(ortho_pt[0]), float(ortho_pt[1]))
            sq.append((float(proj_e) - float(true_e)) ** 2 + (float(proj_n) - float(true_n)) ** 2)
    return float(np.sqrt(np.mean(sq)))


# ---------------------------------------------------------------------------
# DoD 1 + 3: hold-out reprojection validation in meters
# ---------------------------------------------------------------------------


def test_holdout_reports_meters_stats_and_is_consistent() -> None:
    ortho_lines = _ortho_lines()
    rng = np.random.default_rng(7)
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN, noise_px=0.5, rng=rng)
    seed = _identity_seed(ortho_lines)

    result = validate_with_holdout(
        frame_lines=frame_lines,
        ortho_lines=ortho_lines,
        seed=seed,
        calibration_table=_zero_distortion_table(),
        commanded_zoom=1.0,
        commanded_tilt=-15.0,
        map_id="map_test",
        geotiff=_geotiff(),
        rms_threshold_m=1.0,
        holdout_fraction=0.3,
        rng_seed=0,
    )

    assert isinstance(result, HoldoutValidationResult)
    # Three lines held out (round(10 * 0.3)); two endpoints each.
    assert result.holdout_stats.n_points == 6
    assert result.fit_stats.n_points == 14
    # Stats ordered RMS <= p90 <= max and all finite & non-negative.
    for stats in (result.fit_stats, result.holdout_stats):
        assert isinstance(stats, ReprojectionStats)
        assert 0.0 <= stats.rms_m <= stats.p90_m <= stats.max_m
        assert np.isfinite(stats.max_m)
    # Fit and hold-out errors are the same order of magnitude (no overfitting):
    # both reflect the injected 0.5 px ~ 0.05 m endpoint noise.
    assert result.holdout_stats.rms_m < 0.5
    assert result.holdout_stats.rms_m <= 4.0 * result.fit_stats.rms_m + 0.05


def test_holdout_exact_homography_near_zero_error() -> None:
    ortho_lines = _ortho_lines()
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN)  # no noise
    seed = _identity_seed(ortho_lines)

    result = validate_with_holdout(
        frame_lines=frame_lines,
        ortho_lines=ortho_lines,
        seed=seed,
        calibration_table=_zero_distortion_table(),
        commanded_zoom=1.0,
        commanded_tilt=0.0,
        map_id="map_test",
        geotiff=_geotiff(),
        rms_threshold_m=0.01,
        holdout_fraction=0.3,
    )

    assert result.fit_stats.rms_m < 1e-3
    assert result.holdout_stats.rms_m < 1e-3


def test_holdout_over_threshold_raises_typed_error() -> None:
    ortho_lines = _ortho_lines()
    rng = np.random.default_rng(3)
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN, noise_px=3.0, rng=rng)
    seed = _identity_seed(ortho_lines)

    with pytest.raises(HoldoutValidationError) as excinfo:
        validate_with_holdout(
            frame_lines=frame_lines,
            ortho_lines=ortho_lines,
            seed=seed,
            calibration_table=_zero_distortion_table(),
            commanded_zoom=1.0,
            commanded_tilt=0.0,
            map_id="map_test",
            geotiff=_geotiff(),
            rms_threshold_m=1e-4,  # unsatisfiably tight -> must fail
            holdout_fraction=0.3,
        )

    err = excinfo.value
    assert err.holdout_rms_m > err.threshold_m
    assert err.threshold_m == 1e-4
    assert err.holdout_stats.rms_m == err.holdout_rms_m


def test_holdout_rejects_seed_too_small_to_split() -> None:
    ortho_lines = _ortho_lines()[:2]
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN)

    with pytest.raises(ValueError, match="at least 3"):
        validate_with_holdout(
            frame_lines=frame_lines,
            ortho_lines=ortho_lines,
            seed=_identity_seed(ortho_lines),
            calibration_table=_zero_distortion_table(),
            commanded_zoom=1.0,
            commanded_tilt=0.0,
            map_id="map_test",
            geotiff=_geotiff(),
            rms_threshold_m=1.0,
        )


def test_holdout_rejects_bad_fraction() -> None:
    ortho_lines = _ortho_lines()
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN)

    with pytest.raises(ValueError, match="holdout_fraction"):
        validate_with_holdout(
            frame_lines=frame_lines,
            ortho_lines=ortho_lines,
            seed=_identity_seed(ortho_lines),
            calibration_table=_zero_distortion_table(),
            commanded_zoom=1.0,
            commanded_tilt=0.0,
            map_id="map_test",
            geotiff=_geotiff(),
            rms_threshold_m=1.0,
            holdout_fraction=1.5,
        )


# ---------------------------------------------------------------------------
# DoD 2: multi-frame consolidation non-regression
# ---------------------------------------------------------------------------


def test_consolidation_reduces_or_matches_single_frame_error() -> None:
    ortho_lines = _ortho_lines()
    geotiff = _geotiff()
    table = _zero_distortion_table()
    seed = _identity_seed(ortho_lines)
    # Noise-free correspondences are the ground truth the estimates are judged on.
    true_frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN)

    rng = np.random.default_rng(123)
    noisy_per_frame = [
        _frame_lines_from_ortho(ortho_lines, _H_KNOWN, noise_px=1.5, rng=rng) for _ in range(5)
    ]

    # Per-frame fits, each judged against the noise-free ground truth.
    single_rms: list[float] = []
    for frame_lines in noisy_per_frame:
        reg = register_frame_lines(
            frame_lines=frame_lines,
            ortho_lines=ortho_lines,
            seed=seed,
            calibration_table=table,
            commanded_zoom=1.0,
            commanded_tilt=0.0,
            map_id="map_test",
        )
        single_rms.append(
            _rms_m_against_truth(reg.homography_matrix, ortho_lines, true_frame_lines, geotiff)
        )

    consolidated = consolidate_ptz_frames(
        frames=[FrameCorrespondence(frame_lines=fl, seed=seed) for fl in noisy_per_frame],
        ortho_lines=ortho_lines,
        calibration_table=table,
        commanded_zoom=1.0,
        commanded_tilt=0.0,
        map_id="map_test",
        geotiff=geotiff,
        run_id="run1",
        camera_id="cam1",
    )

    assert isinstance(consolidated, MultiFrameConsolidationResult)
    assert consolidated.num_frames == 5
    assert consolidated.num_lines == 5 * len(ortho_lines)
    assert consolidated.run_id == "run1"
    assert consolidated.commanded_zoom == 1.0

    consolidated_rms = _rms_m_against_truth(
        consolidated.homography_matrix, ortho_lines, true_frame_lines, geotiff
    )
    mean_single = float(np.mean(single_rms))
    # Pooling endpoints averages out per-frame noise: consolidated estimate is at
    # least as close to ground truth as the typical single-frame estimate.
    assert consolidated_rms <= mean_single + 1e-9


def test_consolidate_rejects_empty_frames() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        consolidate_ptz_frames(
            frames=[],
            ortho_lines=_ortho_lines(),
            calibration_table=_zero_distortion_table(),
            commanded_zoom=1.0,
            commanded_tilt=0.0,
            map_id="map_test",
            geotiff=_geotiff(),
        )
