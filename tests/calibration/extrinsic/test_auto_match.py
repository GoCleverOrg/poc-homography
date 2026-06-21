"""Offline, deterministic tests for the automatic frame↔ortho RANSAC matcher.

These mirror ``test_ptz_registration.py``'s conventions: a known homography
warps ortho map-pixel lines into a synthetic frame (ortho→frame), and the
matcher must recover the correspondence and the inverse frame→ortho homography
WITHOUT any operator seed. Intrinsics come from a zero-distortion
:class:`CameraCalibrationTable` so the math is exact.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from poc_homography.calibration.extrinsic import (
    AutoMatchResult,
    InsufficientCorrespondenceError,
    match_and_register,
)
from poc_homography.calibration.lens_distortion.calibration_table import (
    CameraCalibrationTable,
    ZoomCalibrationEntry,
)
from poc_homography.calibration.lens_distortion.line_detection import CandidateLine
from poc_homography.domain.vo.ortho_line import OrthoLine

# Frame is sized so cx=960, cy=540 sit at the image center.
_IMAGE_WIDTH = 1920
_IMAGE_HEIGHT = 1080

# Known perspective warp ORTHO map-pixel -> FRAME pixel; the matcher recovers
# frame -> ortho (≈ inverse of this).
_H_ORTHO_TO_FRAME = np.array(
    [
        [1.10, 0.05, 40.0],
        [0.03, 1.20, 25.0],
        [1.0e-4, 5.0e-5, 1.0],
    ],
    dtype=np.float64,
)

# Well-conditioned ortho segments with varied orientations, spread across the
# frame so the warped inlier endpoints achieve a "Good"/"Fair" distribution.
_ORTHO_SEGMENTS: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((150.0, 150.0), (1400.0, 180.0)),  # near-horizontal (top)
    ((200.0, 250.0), (240.0, 850.0)),  # near-vertical (left)
    ((300.0, 300.0), (1300.0, 820.0)),  # diagonal /
    ((1350.0, 260.0), (350.0, 800.0)),  # diagonal \
    ((900.0, 200.0), (940.0, 880.0)),  # another vertical-ish (center)
]

# Repetitive near-parallel lines (parking-stall style): same orientation,
# offset sideways, spread across the frame. Descriptor-only matching could swap.
_REPETITIVE_SEGMENTS: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((250.0, 150.0), (280.0, 850.0)),
    ((650.0, 150.0), (680.0, 850.0)),
    ((1050.0, 150.0), (1080.0, 850.0)),
    ((200.0, 180.0), (1300.0, 210.0)),  # one cross line to anchor the fit
]


def _zero_distortion_table(camera_id: str = "cam1") -> CameraCalibrationTable:
    """Calibration table with real intrinsics but zero distortion (exact math)."""
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


def _ortho_lines(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[OrthoLine]:
    """Build OrthoLine objects from canned segments (arbitrary UTM)."""
    lines: list[OrthoLine] = []
    for i, (start, end) in enumerate(segments):
        lines.append(
            OrthoLine.from_endpoints(
                start_pixel=start,
                end_pixel=end,
                start_utm=(500000.0 + i, 4000000.0 + i),
                end_utm=(500010.0 + i, 4000010.0 + i),
            )
        )
    return lines


def _warp_point(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    """Apply a homography to a single (x, y) point."""
    pt = np.array([[[point[0], point[1]]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, homography)[0, 0]
    return float(out[0]), float(out[1])


def _frame_lines_from_ortho(
    ortho_lines: list[OrthoLine], homography: np.ndarray
) -> list[CandidateLine]:
    """Project each ortho line's map-pixel endpoints into frame pixels via H."""
    frame_lines: list[CandidateLine] = []
    for line in ortho_lines:
        start = _warp_point((float(line.start_pixel[0]), float(line.start_pixel[1])), homography)
        end = _warp_point((float(line.end_pixel[0]), float(line.end_pixel[1])), homography)
        frame_lines.append(CandidateLine(start=start, end=end))
    return frame_lines


def _max_reprojection_error(
    result: AutoMatchResult,
    ortho_lines: list[OrthoLine],
    frame_lines: list[CandidateLine],
    seed: dict[int, str],
) -> float:
    """Max distance projecting each matched frame endpoint back to its ortho."""
    max_err = 0.0
    for f_idx, ortho_id in seed.items():
        o_idx = int(ortho_id.removeprefix("ortho_"))
        line = ortho_lines[o_idx]
        frame_line = frame_lines[f_idx]
        for frame_pt, ortho_pt in (
            (frame_line.start, line.start_pixel),
            (frame_line.end, line.end_pixel),
        ):
            recovered = _warp_point(frame_pt, result.homography_matrix)
            err = float(
                np.hypot(recovered[0] - float(ortho_pt[0]), recovered[1] - float(ortho_pt[1]))
            )
            max_err = max(max_err, err)
    return max_err


# ---------------------------------------------------------------------------
# DoD: recovery without seeding (incl. repetitive + spurious outlier lines)
# ---------------------------------------------------------------------------


def test_recovers_correspondence_without_seeding() -> None:
    ortho_lines = _ortho_lines(_ORTHO_SEGMENTS)
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_ORTHO_TO_FRAME)

    # Add spurious outlier frame lines that match NO ortho line.
    frame_lines.append(CandidateLine(start=(1500.0, 50.0), end=(1700.0, 60.0)))
    frame_lines.append(CandidateLine(start=(50.0, 900.0), end=(60.0, 1000.0)))

    result = match_and_register(
        frame_lines=frame_lines,
        ortho_lines=ortho_lines,
        calibration_table=_zero_distortion_table(),
        commanded_zoom=1.0,
        commanded_tilt=-15.0,
        map_id="map_test",
        image_width=_IMAGE_WIDTH,
        image_height=_IMAGE_HEIGHT,
        rng_seed=0,
        ransac_threshold=5.0,
    )

    assert isinstance(result, AutoMatchResult)
    # Each of the 5 real frame lines (indices 0..4) maps to ortho_i.
    expected = {i: f"ortho_{i}" for i in range(len(ortho_lines))}
    assert dict(result.seed) == expected
    assert result.num_inliers == len(ortho_lines)

    max_err = _max_reprojection_error(result, ortho_lines, frame_lines, dict(result.seed))
    assert max_err < 1.0, f"max reprojection error {max_err:.4f}px exceeds tolerance"


def test_recovers_repetitive_lines_descriptor_only() -> None:
    """Repetitive parallel lines, recovered by line-only matching (no prior)."""
    ortho_lines = _ortho_lines(_REPETITIVE_SEGMENTS)
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_ORTHO_TO_FRAME)

    result = match_and_register(
        frame_lines=frame_lines,
        ortho_lines=ortho_lines,
        calibration_table=_zero_distortion_table(),
        commanded_zoom=1.0,
        commanded_tilt=-15.0,
        map_id="map_test",
        image_width=_IMAGE_WIDTH,
        image_height=_IMAGE_HEIGHT,
        rng_seed=0,
        ransac_threshold=5.0,
    )

    expected = {i: f"ortho_{i}" for i in range(len(ortho_lines))}
    assert dict(result.seed) == expected
    max_err = _max_reprojection_error(result, ortho_lines, frame_lines, dict(result.seed))
    assert max_err < 1.0


# ---------------------------------------------------------------------------
# DoD: disambiguation via pose prior
# ---------------------------------------------------------------------------


def test_pose_prior_disambiguates_repetitive_lines() -> None:
    ortho_lines = _ortho_lines(_REPETITIVE_SEGMENTS)
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_ORTHO_TO_FRAME)

    # Prior is frame->ortho (same orientation as output): inverse of the
    # ortho->frame warp, then PERTURBED so it is only approximately correct.
    true_frame_to_ortho = np.linalg.inv(_H_ORTHO_TO_FRAME)
    perturbation = np.array(
        [
            [1.02, 0.01, 8.0],
            [-0.01, 0.98, -6.0],
            [1.0e-5, -1.0e-5, 1.0],
        ],
        dtype=np.float64,
    )
    pose_prior = perturbation @ true_frame_to_ortho

    result = match_and_register(
        frame_lines=frame_lines,
        ortho_lines=ortho_lines,
        calibration_table=_zero_distortion_table(),
        commanded_zoom=1.0,
        commanded_tilt=-15.0,
        map_id="map_test",
        image_width=_IMAGE_WIDTH,
        image_height=_IMAGE_HEIGHT,
        pose_prior=pose_prior,
        rng_seed=0,
        ransac_threshold=5.0,
        # Tight gate that only the prior-correct match survives.
        prior_distance_tolerance=60.0,
    )

    expected = {i: f"ortho_{i}" for i in range(len(ortho_lines))}
    assert dict(result.seed) == expected
    max_err = _max_reprojection_error(result, ortho_lines, frame_lines, dict(result.seed))
    assert max_err < 1.0


# ---------------------------------------------------------------------------
# DoD: fail-fast — too few inliers
# ---------------------------------------------------------------------------


def test_fail_fast_too_few_inliers() -> None:
    # Only two ortho lines -> at most 2 inlier lines, below min_inliers=4.
    ortho_lines = _ortho_lines(_ORTHO_SEGMENTS[:2])
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_ORTHO_TO_FRAME)

    with pytest.raises(InsufficientCorrespondenceError) as excinfo:
        match_and_register(
            frame_lines=frame_lines,
            ortho_lines=ortho_lines,
            calibration_table=_zero_distortion_table(),
            commanded_zoom=1.0,
            commanded_tilt=0.0,
            map_id="map_test",
            image_width=_IMAGE_WIDTH,
            image_height=_IMAGE_HEIGHT,
            rng_seed=0,
            ransac_threshold=5.0,
            min_inliers=4,
        )

    assert excinfo.value.num_inliers < 4


# ---------------------------------------------------------------------------
# DoD: fail-fast — Poor/Insufficient distribution
# ---------------------------------------------------------------------------


def test_fail_fast_poor_distribution() -> None:
    """All matched lines clustered in one small image region -> reject."""
    # Tightly clustered, distinctly-oriented ortho lines in a tiny corner so the
    # convex hull is minuscule and quadrant coverage is 1 -> "Poor"/"Insufficient".
    clustered_segments: list[tuple[tuple[float, float], tuple[float, float]]] = [
        ((100.0, 100.0), (140.0, 102.0)),
        ((105.0, 100.0), (107.0, 140.0)),
        ((100.0, 105.0), (138.0, 138.0)),
        ((138.0, 102.0), (102.0, 138.0)),
    ]
    ortho_lines = _ortho_lines(clustered_segments)
    # Mild warp keeps everything in the top-left frame corner.
    mild = np.array(
        [
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    frame_lines = _frame_lines_from_ortho(ortho_lines, mild)

    with pytest.raises(InsufficientCorrespondenceError) as excinfo:
        match_and_register(
            frame_lines=frame_lines,
            ortho_lines=ortho_lines,
            calibration_table=_zero_distortion_table(),
            commanded_zoom=1.0,
            commanded_tilt=0.0,
            map_id="map_test",
            image_width=_IMAGE_WIDTH,
            image_height=_IMAGE_HEIGHT,
            rng_seed=0,
            ransac_threshold=5.0,
            min_inliers=4,
        )

    assert excinfo.value.distribution_quality in {"Poor", "Insufficient"}


# ---------------------------------------------------------------------------
# DoD: determinism
# ---------------------------------------------------------------------------


def test_determinism_same_seed_identical_output() -> None:
    ortho_lines = _ortho_lines(_ORTHO_SEGMENTS)
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_ORTHO_TO_FRAME)
    frame_lines.append(CandidateLine(start=(1500.0, 50.0), end=(1700.0, 60.0)))

    kwargs = {
        "frame_lines": frame_lines,
        "ortho_lines": ortho_lines,
        "calibration_table": _zero_distortion_table(),
        "commanded_zoom": 1.0,
        "commanded_tilt": -15.0,
        "map_id": "map_test",
        "image_width": _IMAGE_WIDTH,
        "image_height": _IMAGE_HEIGHT,
        "rng_seed": 7,
        "ransac_threshold": 5.0,
    }

    result_a = match_and_register(**kwargs)
    result_b = match_and_register(**kwargs)

    assert dict(result_a.seed) == dict(result_b.seed)
    np.testing.assert_array_equal(result_a.homography_matrix, result_b.homography_matrix)
