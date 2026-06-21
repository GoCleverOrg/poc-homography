"""Tests for the per-zoom scene self-calibration service (#350).

Covers every Definition-of-Done bullet:

1. ``calibrate_scene_self`` across >=2 zooms returns a ``SceneCalibrationResult``
   whose ``.table`` is a ``LensCalibrationTable`` with one entry per distinct
   zoom, each carrying refined fx/fy/cx/cy and a validation_rmse — driven
   through a fake duck-typed detector.
2. On synthetic barrel-distorted floor lines with a known ``true_k1`` built off
   the prior K, ``calibrate_zoom_from_lines`` recovers a k1 closer to the truth
   than the zero-distortion prior, with a post-calibration RMSE below the
   pre-calibration straightness RMSE.
3. A ``LensCalibrationTable`` round-trips through ``RepoYamlLensCalibrationTable``.
4. A zoom group with too few usable lines is reported as a ``SkippedZoom`` and
   does not raise; valid zooms still produce entries.
"""

from __future__ import annotations

import numpy as np

from poc_homography.calibration.lens_distortion.apply_calibration import distort_points
from poc_homography.calibration.lens_distortion.distortion_solver import (
    straightness_rmse,
)
from poc_homography.calibration.lens_distortion.models import CameraLine, PTZPosition
from poc_homography.calibration.lens_distortion.scene_self_calibration import (
    ScenePerZoomConfig,
    SkippedZoom,
    calibrate_scene_self,
    calibrate_zoom_from_lines,
)
from poc_homography.camera.intrinsics import compute_intrinsics
from poc_homography.domain.entities.lens_calibration_table import LensCalibrationTable
from poc_homography.domain.enums.camera_spec import CameraSpec
from poc_homography.domain.vo.lens_distortion import LensDistortion
from poc_homography.domain.vo.zoom_calibration_entry import ZoomCalibrationEntry
from poc_homography.infrastructure.repositories.repo_yaml_lens_calibration_table import (
    RepoYamlLensCalibrationTable,
)
from poc_homography.types import PixelsFloat, Unitless

SPEC = CameraSpec.HIKVISION_DS_2DF8425IX


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _prior_k_for_zoom(zoom: float) -> np.ndarray:
    """Prior intrinsic matrix exactly as the service seeds it."""
    return compute_intrinsics(
        zoom=Unitless(zoom),
        image_width=SPEC.image_width,
        image_height=SPEC.image_height,
        sensor_width_mm=SPEC.sensor_width,
        base_focal_length_mm=SPEC.base_focal_length,
    ).K


def _distorted_lines(
    zoom: float,
    true_k1: float,
    true_k2: float = 0.0,
    *,
    pts_per_line: int = 50,
) -> list[CameraLine]:
    """Build barrel-distorted straight floor lines off the prior K for *zoom*.

    Generates a mix of horizontal / vertical / diagonal lines spanning the
    frame so the recovery problem is well conditioned. Uses the SAME fx/fy/cx/cy
    that ``compute_intrinsics`` produces so the fx<->k1 degeneracy does not make
    recovery meaningless.
    """
    k = _prior_k_for_zoom(zoom)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    width = float(SPEC.image_width)
    height = float(SPEC.image_height)
    margin = 120.0

    ptz = PTZPosition(pan_deg=0.0, tilt_deg=0.0, zoom_factor=zoom)
    specs: list[tuple[str, np.ndarray]] = []

    for i, y_pos in enumerate(np.linspace(margin, height - margin, 3)):
        undist = np.column_stack(
            [np.linspace(margin, width - margin, pts_per_line), np.full(pts_per_line, y_pos)]
        )
        specs.append((f"h_{i}", undist))
    for i, x_pos in enumerate(np.linspace(margin, width - margin, 3)):
        undist = np.column_stack(
            [np.full(pts_per_line, x_pos), np.linspace(margin, height - margin, pts_per_line)]
        )
        specs.append((f"v_{i}", undist))
    for i, (xs, ys, xe, ye) in enumerate(
        [
            (margin, margin, width - margin, height - margin),
            (width - margin, margin, margin, height - margin),
        ]
    ):
        undist = np.column_stack(
            [np.linspace(xs, xe, pts_per_line), np.linspace(ys, ye, pts_per_line)]
        )
        specs.append((f"d_{i}", undist))

    lines: list[CameraLine] = []
    for line_id, undist in specs:
        # Reuse the shipped forward model (k3=p1=p2=0) so recovery is validated
        # against the production distortion, not a hand-rolled twin.
        distorted = distort_points(undist, true_k1, true_k2, 0.0, 0.0, 0.0, fx, fy, cx, cy)
        edge_pixels = tuple((float(p[0]), float(p[1])) for p in distorted)
        lines.append(
            CameraLine(
                line_id=line_id,
                image_path=f"synthetic_zoom{zoom}",
                start_pixel=edge_pixels[0],
                end_pixel=edge_pixels[-1],
                ptz_position=ptz,
                edge_pixels=edge_pixels,
            )
        )
    return lines


class _FakeCandidate:
    """Minimal CandidateLine stand-in exposing ``to_camera_line``."""

    def __init__(self, edge_pixels: tuple[tuple[float, float], ...]) -> None:
        self.edge_pixels = edge_pixels

    def to_camera_line(
        self,
        line_id: str,
        image_path: str,
        ptz_position: PTZPosition,
        line_type: str = "parking",
    ) -> CameraLine:
        return CameraLine(
            line_id=line_id,
            image_path=image_path,
            start_pixel=self.edge_pixels[0],
            end_pixel=self.edge_pixels[-1],
            ptz_position=ptz_position,
            line_type=line_type,
            edge_pixels=self.edge_pixels,
        )


def _fake_candidates_for_zoom(zoom: float, true_k1: float) -> list[_FakeCandidate]:
    return [_FakeCandidate(line.edge_pixels) for line in _distorted_lines(zoom, true_k1)]  # type: ignore[arg-type]


class _QueueDetector:
    """Duck-typed LineDetector returning a queued candidate set per detect call.

    The service processes zoom groups in sorted order, so the queue is seeded in
    that order (one entry per detected image).
    """

    def __init__(self, queue: list[list[_FakeCandidate]]) -> None:
        self._queue = queue

    def detect(self, image: np.ndarray) -> list[_FakeCandidate]:
        return self._queue.pop(0)


# ---------------------------------------------------------------------------
# DoD#1 — full image entrypoint across >=2 zooms via fake detector
# ---------------------------------------------------------------------------


def test_calibrate_scene_self_one_entry_per_zoom() -> None:
    zoom_a, zoom_b = 1.0, 2.0
    true_k1 = -0.30
    img = np.zeros((SPEC.image_height, SPEC.image_width, 3), dtype=np.uint8)

    # Each zoom group needs candidates built off ITS prior K. The service
    # processes zooms in sorted order, so queue the per-zoom candidate sets in
    # that order: zoom_a (1.0) first, then zoom_b (2.0).
    detector = _QueueDetector(
        [
            _fake_candidates_for_zoom(zoom_a, true_k1),
            _fake_candidates_for_zoom(zoom_b, true_k1),
        ]
    )
    result = calibrate_scene_self(
        [(img, zoom_a), (img, zoom_b)],
        camera_id="cam",
        detector=detector,
    )

    assert isinstance(result.table, LensCalibrationTable)
    assert result.skipped_zooms == ()
    assert len(result.table.entries) == 2

    zoom_list = [float(e.zoom_factor) for e in result.table.entries]
    assert set(zoom_list) == {zoom_a, zoom_b}
    # The table contract states entries are ordered by zoom_factor.
    assert zoom_list == sorted(zoom_list)

    cfg = ScenePerZoomConfig()
    for entry in result.table.entries:
        # Each entry's intrinsics must be the *refined* values: within the
        # configured solver window of the prior K seeded for that zoom (a bare
        # `> 0` check would pass regardless of whether refinement ran).
        prior_k = _prior_k_for_zoom(float(entry.zoom_factor))
        fx0, fy0 = float(prior_k[0, 0]), float(prior_k[1, 1])
        cx0, cy0 = float(prior_k[0, 2]), float(prior_k[1, 2])
        assert abs(entry.fx - fx0) <= fx0 * cfg.fx_refine_frac + 1e-6
        assert abs(entry.fy - fy0) <= fy0 * cfg.fx_refine_frac + 1e-6
        assert abs(entry.cx - cx0) <= cfg.cx_window_px + 1e-6
        assert abs(entry.cy - cy0) <= cfg.cy_window_px + 1e-6
        assert entry.validation_rmse >= 0.0
        assert entry.num_lines_used >= 3


# ---------------------------------------------------------------------------
# DoD#2 — key correctness: k1 recovery off the prior K
# ---------------------------------------------------------------------------


def test_calibrate_zoom_from_lines_recovers_k1() -> None:
    zoom = 1.0
    true_k1 = -0.30
    lines = _distorted_lines(zoom, true_k1)
    prior_k = _prior_k_for_zoom(zoom)
    config = ScenePerZoomConfig()

    pre_rmse = straightness_rmse(lines, prior_k, distortion=None)

    entry = calibrate_zoom_from_lines(lines, zoom, camera_spec=SPEC, config=config)
    assert isinstance(entry, ZoomCalibrationEntry)

    recovered_k1 = float(entry.distortion.k1)
    # Recovered k1 is closer to the truth than the zero-distortion prior.
    assert abs(recovered_k1 - true_k1) < abs(0.0 - true_k1)
    # Post-calibration straightness improves over the uncorrected prior.
    assert entry.validation_rmse < pre_rmse


# ---------------------------------------------------------------------------
# DoD#3 — persistence round-trip
# ---------------------------------------------------------------------------


def test_lens_calibration_table_round_trip(tmp_path) -> None:
    entry = ZoomCalibrationEntry(
        zoom_factor=Unitless(2.0),
        distortion=LensDistortion(k1=Unitless(-0.31), k2=Unitless(0.05)),
        calibration_date="2026-06-21T00:00:00",
        source_images=("cam_zoom2.0_img0",),
        validation_rmse=1.234,
        num_lines_used=8,
        fx=PixelsFloat(2500.0),
        fy=PixelsFloat(2500.0),
        cx=PixelsFloat(1280.0),
        cy=PixelsFloat(720.0),
    )
    table = LensCalibrationTable(
        id="cam-roundtrip",
        entries=(entry,),
        created_date="2026-06-21T00:00:00",
        last_modified="2026-06-21T00:00:00",
    )

    repo = RepoYamlLensCalibrationTable(tmp_path)
    repo.save(table)
    loaded = repo.get("cam-roundtrip")

    assert loaded is not None
    assert len(loaded.entries) == 1
    got = loaded.entries[0]
    assert float(got.zoom_factor) == float(entry.zoom_factor)
    assert got.distortion.to_dict() == entry.distortion.to_dict()
    assert float(got.fx) == float(entry.fx)
    assert float(got.fy) == float(entry.fy)
    assert float(got.cx) == float(entry.cx)
    assert float(got.cy) == float(entry.cy)
    assert got.validation_rmse == entry.validation_rmse
    assert got.num_lines_used == entry.num_lines_used
    assert got.source_images == entry.source_images


# ---------------------------------------------------------------------------
# DoD#4 — too-few-lines group is skipped, others still calibrate
# ---------------------------------------------------------------------------


def test_too_few_lines_reported_as_skipped() -> None:
    zoom = 1.0
    lines = _distorted_lines(zoom, true_k1=-0.30)[:1]  # only one usable line

    outcome = calibrate_zoom_from_lines(lines, zoom, camera_spec=SPEC, config=ScenePerZoomConfig())
    assert isinstance(outcome, SkippedZoom)
    assert outcome.zoom_factor == zoom
    assert outcome.num_lines == 1


def test_skipped_zoom_does_not_abort_other_zooms() -> None:
    valid_zoom, sparse_zoom = 1.0, 2.0
    true_k1 = -0.30
    img = np.zeros((SPEC.image_height, SPEC.image_width, 3), dtype=np.uint8)

    valid_candidates = _fake_candidates_for_zoom(valid_zoom, true_k1)
    sparse_candidates = valid_candidates[:1]  # too few for the sparse zoom

    # Pairs are processed in sorted-zoom order: valid_zoom (1.0) then sparse (2.0).
    detector = _QueueDetector([valid_candidates, sparse_candidates])
    result = calibrate_scene_self(
        [(img, valid_zoom), (img, sparse_zoom)],
        camera_id="cam",
        detector=detector,
    )

    assert len(result.table.entries) == 1
    assert float(result.table.entries[0].zoom_factor) == valid_zoom
    assert len(result.skipped_zooms) == 1
    assert result.skipped_zooms[0].zoom_factor == sparse_zoom


def test_near_zero_zoom_is_skipped_not_fatal() -> None:
    """A zoom that rounds to 0.0 is degenerate — skip it, never abort the run."""
    valid_zoom, tiny_zoom = 1.0, 0.02  # 0.02 rounds to 0.0 at the default tolerance
    true_k1 = -0.30
    img = np.zeros((SPEC.image_height, SPEC.image_width, 3), dtype=np.uint8)

    # Only the valid (>0) zoom group reaches detection; the tiny zoom is skipped
    # before any detect call, so a single-entry queue suffices.
    detector = _QueueDetector([_fake_candidates_for_zoom(valid_zoom, true_k1)])
    result = calibrate_scene_self(
        [(img, tiny_zoom), (img, valid_zoom)],
        camera_id="cam",
        detector=detector,
    )

    assert len(result.table.entries) == 1
    assert float(result.table.entries[0].zoom_factor) == valid_zoom
    assert len(result.skipped_zooms) == 1
    assert result.skipped_zooms[0].zoom_factor == 0.0
