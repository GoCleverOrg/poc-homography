"""Offline tests for per-PTZ-state image↔orthophoto homography registration.

These tests exercise the seeded registration pipeline end to end without any
live camera or MinIO/Postgres: a known homography warps ortho lines into a
synthetic frame, intrinsics come from a :class:`CameraCalibrationTable` fixture,
and the survey-frame integration reuses the FakeRepo/FakeStore pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from poc_homography.calibration.extrinsic import (
    PtzRegistrationResult,
    register_frame_lines,
    register_ptz_state,
)
from poc_homography.calibration.lens_distortion.calibration_table import (
    CameraCalibrationTable,
    ZoomCalibrationEntry,
)
from poc_homography.calibration.lens_distortion.frame_source import iter_survey_frames
from poc_homography.calibration.lens_distortion.line_detection import (
    CandidateLine,
    LineDetectionConfig,
    LineDetector,
)
from poc_homography.domain.vo.ortho_line import OrthoLine

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# A realistic, well-conditioned perspective warp mapping ORTHO map-pixel -> FRAME
# pixel. Slight projective terms keep it non-affine yet stable.
_H_KNOWN = np.array(
    [
        [1.10, 0.05, 40.0],
        [0.03, 1.20, 25.0],
        [1.0e-4, 5.0e-5, 1.0],
    ],
    dtype=np.float64,
)

# Ortho line segments (map-pixel endpoints) with varied orientations so the fit
# is well-determined. UTM endpoints are arbitrary-but-valid.
_ORTHO_SEGMENTS: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((100.0, 100.0), (500.0, 110.0)),  # near-horizontal
    ((120.0, 150.0), (130.0, 600.0)),  # near-vertical
    ((150.0, 200.0), (550.0, 580.0)),  # diagonal /
    ((520.0, 160.0), (180.0, 560.0)),  # diagonal \
    ((300.0, 120.0), (320.0, 640.0)),  # another vertical-ish
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


def _ortho_lines() -> list[OrthoLine]:
    """Build OrthoLine objects from the canned segments (arbitrary UTM)."""
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


# ---------------------------------------------------------------------------
# DoD 1 + 2: precise known-H recovery with intrinsics applied
# ---------------------------------------------------------------------------


def test_register_frame_lines_recovers_known_homography() -> None:
    ortho_lines = _ortho_lines()
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN)
    seed = {i: f"ortho_{i}" for i in range(len(ortho_lines))}

    result = register_frame_lines(
        frame_lines=frame_lines,
        ortho_lines=ortho_lines,
        seed=seed,
        calibration_table=_zero_distortion_table(),
        commanded_zoom=1.0,
        commanded_tilt=-15.0,
        map_id="map_test",
        run_id="run42",
        camera_id="cam1",
        ransac_threshold=5.0,
    )

    assert isinstance(result, PtzRegistrationResult)
    assert result.num_lines == len(ortho_lines)
    assert result.num_inliers == 2 * len(ortho_lines)  # two endpoints per line
    assert result.inlier_ratio == 1.0

    # Recovered homography maps frame -> ortho; projecting each frame endpoint
    # through it must land within tolerance of the ORIGINAL ortho map-pixel.
    max_err = 0.0
    for line, frame_line in zip(ortho_lines, frame_lines):
        for frame_pt, ortho_pt in (
            (frame_line.start, line.start_pixel),
            (frame_line.end, line.end_pixel),
        ):
            recovered = _warp_point(frame_pt, result.homography_matrix)
            err = float(
                np.hypot(recovered[0] - float(ortho_pt[0]), recovered[1] - float(ortho_pt[1]))
            )
            max_err = max(max_err, err)

    assert max_err < 1.0, f"max reprojection error {max_err:.4f}px exceeds tolerance"


def test_register_frame_lines_provenance_passthrough() -> None:
    ortho_lines = _ortho_lines()
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN)
    seed = {i: f"ortho_{i}" for i in range(len(ortho_lines))}

    result = register_frame_lines(
        frame_lines=frame_lines,
        ortho_lines=ortho_lines,
        seed=seed,
        calibration_table=_zero_distortion_table(),
        commanded_zoom=1.0,
        commanded_tilt=-22.5,
        map_id="map_test",
        run_id="run99",
        camera_id="camX",
        ransac_threshold=5.0,
    )

    assert result.run_id == "run99"
    assert result.camera_id == "camX"
    assert result.commanded_zoom == 1.0
    assert result.commanded_tilt == -22.5


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_register_frame_lines_rejects_too_few_seeds() -> None:
    ortho_lines = _ortho_lines()
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN)

    with pytest.raises(ValueError, match="at least 2 seeded correspondences"):
        register_frame_lines(
            frame_lines=frame_lines,
            ortho_lines=ortho_lines,
            seed={0: "ortho_0"},
            calibration_table=_zero_distortion_table(),
            commanded_zoom=1.0,
            commanded_tilt=0.0,
            map_id="map_test",
        )


def test_register_frame_lines_rejects_out_of_range_index() -> None:
    ortho_lines = _ortho_lines()
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN)

    with pytest.raises(ValueError, match="out of range"):
        register_frame_lines(
            frame_lines=frame_lines,
            ortho_lines=ortho_lines,
            seed={0: "ortho_0", 99: "ortho_1"},
            calibration_table=_zero_distortion_table(),
            commanded_zoom=1.0,
            commanded_tilt=0.0,
            map_id="map_test",
        )


def test_register_frame_lines_rejects_unknown_ortho_id() -> None:
    ortho_lines = _ortho_lines()
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN)

    with pytest.raises(ValueError, match="unknown ortho id"):
        register_frame_lines(
            frame_lines=frame_lines,
            ortho_lines=ortho_lines,
            seed={0: "ortho_0", 1: "ortho_nope"},
            calibration_table=_zero_distortion_table(),
            commanded_zoom=1.0,
            commanded_tilt=0.0,
            map_id="map_test",
        )


# ---------------------------------------------------------------------------
# DoD: intrinsics/distortion actually wired through
# ---------------------------------------------------------------------------


def test_distortion_changes_result_vs_zero_distortion() -> None:
    """A non-zero-distortion table must change the fit vs. zero distortion."""
    ortho_lines = _ortho_lines()
    frame_lines = _frame_lines_from_ortho(ortho_lines, _H_KNOWN)
    seed = {i: f"ortho_{i}" for i in range(len(ortho_lines))}

    zero_table = _zero_distortion_table()

    distorted_table = CameraCalibrationTable(camera_id="cam1")
    distorted_table.add_entry(
        ZoomCalibrationEntry(
            zoom_factor=1.0,
            k1=-0.25,
            k2=0.08,
            k3=0.0,
            p1=0.001,
            p2=-0.001,
            calibration_date="2024-01-01T00:00:00",
            fx=1000.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0,
        )
    )

    common = {
        "frame_lines": frame_lines,
        "ortho_lines": ortho_lines,
        "seed": seed,
        "commanded_zoom": 1.0,
        "commanded_tilt": 0.0,
        "map_id": "map_test",
        "ransac_threshold": 50.0,
    }

    zero_result = register_frame_lines(calibration_table=zero_table, **common)
    distorted_result = register_frame_lines(calibration_table=distorted_table, **common)

    # Undistortion materially changes the correspondences -> different H.
    assert not np.allclose(
        zero_result.homography_matrix, distorted_result.homography_matrix, atol=1e-6
    )


# ---------------------------------------------------------------------------
# DoD: detection path via register_ptz_state
# ---------------------------------------------------------------------------


def _render_ortho_image(
    ortho_lines: list[OrthoLine], size: tuple[int, int] = (720, 720)
) -> np.ndarray:
    """Black canvas with thick white segments at each ortho line's map pixels."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for line in ortho_lines:
        pt1 = (int(line.start_pixel[0]), int(line.start_pixel[1]))
        pt2 = (int(line.end_pixel[0]), int(line.end_pixel[1]))
        cv2.line(img, pt1, pt2, (255, 255, 255), 6)
    return img


def test_register_ptz_state_detection_path_produces_valid_homography() -> None:
    ortho_lines = _ortho_lines()
    ortho_img = _render_ortho_image(ortho_lines)

    # Mild warp keeps the warped lines inside the frame and detectable.
    h_warp = np.array(
        [
            [1.05, 0.02, 10.0],
            [0.01, 1.05, 8.0],
            [2.0e-5, 1.0e-5, 1.0],
        ],
        dtype=np.float64,
    )
    frame = cv2.warpPerspective(ortho_img, h_warp, (720, 720))

    detector = LineDetector(LineDetectionConfig(min_line_length=60, min_confidence=0.0))
    detected = detector.detect(frame)
    assert len(detected) >= 2, "detection should find at least two lines"

    # Test-side seeding (operator-provided by contract): match each detected line
    # to the nearest ortho line by midpoint after warping ortho midpoints.
    def _midpoint(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
        return ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)

    warped_ortho_mids: list[tuple[float, float]] = []
    for line in ortho_lines:
        mid = _midpoint(
            (float(line.start_pixel[0]), float(line.start_pixel[1])),
            (float(line.end_pixel[0]), float(line.end_pixel[1])),
        )
        warped_ortho_mids.append(_warp_point(mid, h_warp))

    seed: dict[int, str] = {}
    used: set[int] = set()
    for det_idx, det in enumerate(detected):
        det_mid = _midpoint(det.start, det.end)
        best_j, best_d = -1, float("inf")
        for j, omid in enumerate(warped_ortho_mids):
            if j in used:
                continue
            d = float(np.hypot(det_mid[0] - omid[0], det_mid[1] - omid[1]))
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d < 60.0:
            seed[det_idx] = f"ortho_{best_j}"
            used.add(best_j)

    if len(seed) < 2:
        pytest.skip("Hough detection did not yield two reliably-seedable lines")

    result = register_ptz_state(
        frame=frame,
        ortho_lines=ortho_lines,
        seed=seed,
        calibration_table=_zero_distortion_table(),
        commanded_zoom=1.0,
        commanded_tilt=-10.0,
        map_id="map_test",
        line_detector=detector,
        ransac_threshold=50.0,
        min_inlier_ratio=0.0,
    )

    H = result.homography_matrix
    assert H.shape == (3, 3)
    assert np.all(np.isfinite(H))
    assert abs(float(np.linalg.det(H))) > 1e-9  # non-singular
    assert result.num_inliers > 0


# ---------------------------------------------------------------------------
# DoD: offline iter_survey_frames integration (FakeRepo/FakeStore)
# ---------------------------------------------------------------------------


@dataclass
class FakeRow:
    """Minimal stand-in for a clean-plate frame row."""

    minio_bucket: str
    minio_object_key: str
    commanded_zoom: float
    commanded_tilt: float
    run_id: str = "run1"
    camera_id: str = "cam1"


class FakeRepo:
    """Filters canned rows by run/camera and paginates."""

    def __init__(self, rows: list[FakeRow]) -> None:
        self._rows = rows

    def query_frames(
        self,
        *,
        run_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[FakeRow], int]:
        matched = [
            r
            for r in self._rows
            if (run_id is None or r.run_id == run_id)
            and (camera_id is None or r.camera_id == camera_id)
        ]
        return matched[offset : offset + limit], len(matched)


class FakeStore:
    """Serves bytes from an in-memory ``(bucket, key) -> bytes`` map."""

    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self._objects = objects

    def get_frame(self, key: str, *, bucket: str | None = None) -> bytes:
        return self._objects[(bucket, key)]


def test_offline_survey_frame_registration() -> None:
    ortho_lines = _ortho_lines()
    ortho_img = _render_ortho_image(ortho_lines)
    h_warp = np.array(
        [
            [1.05, 0.02, 10.0],
            [0.01, 1.05, 8.0],
            [2.0e-5, 1.0e-5, 1.0],
        ],
        dtype=np.float64,
    )
    frame_img = cv2.warpPerspective(ortho_img, h_warp, (720, 720))
    ok, buf = cv2.imencode(".jpg", frame_img)
    assert ok

    rows = [FakeRow("frames", "run1/cam1/0.jpg", commanded_zoom=1.0, commanded_tilt=-12.0)]
    store = FakeStore({("frames", "run1/cam1/0.jpg"): buf.tobytes()})
    repo = FakeRepo(rows)

    frames = list(iter_survey_frames(run_id="run1", camera_id="cam1", repo=repo, store=store))
    assert len(frames) == 1
    survey_frame = frames[0]

    # Use known frame lines derived from the same warp so the fit is robust and
    # deterministic regardless of Hough behaviour; this test targets the offline
    # plumbing + provenance, not detection.
    frame_lines = _frame_lines_from_ortho(ortho_lines, h_warp)
    seed = {i: f"ortho_{i}" for i in range(len(ortho_lines))}

    result = register_frame_lines(
        frame_lines=frame_lines,
        ortho_lines=ortho_lines,
        seed=seed,
        calibration_table=_zero_distortion_table(),
        commanded_zoom=survey_frame.commanded_zoom,
        commanded_tilt=survey_frame.commanded_tilt,
        map_id="map_test",
        run_id="run1",
        camera_id="cam1",
        ransac_threshold=5.0,
    )

    assert result.run_id == "run1"
    assert result.camera_id == "cam1"
    assert result.commanded_zoom == 1.0
    assert result.commanded_tilt == -12.0
    assert np.all(np.isfinite(result.homography_matrix))
    assert abs(float(np.linalg.det(result.homography_matrix))) > 1e-9
