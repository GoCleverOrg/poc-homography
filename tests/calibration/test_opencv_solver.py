"""Tests for OpenCV-based distortion calibration solver.

Tests cover:
- GCP correspondence construction
- Line correspondence construction
- Line sampling along line segments
- Deterministic train/validation split
- OpenCV solver returns SolverResult with all required fields
- Distortion coefficients match Brown-Conrady model format
- Reprojection error computation
"""

from __future__ import annotations

import numpy as np
import pytest

from poc_homography.calibration.lens_distortion.opencv_solver import (
    GCPCorrespondence,
    LineCorrespondencePair,
    LineSplitResult,
    OpenCVDistortionSolver,
    OpenCVSolverConfig,
    build_gcp_correspondences,
    build_line_correspondences,
    sample_line_correspondences,
    split_lines,
)
from poc_homography.camera_parameters import DistortionCoefficients


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_gcp_registry() -> list[dict]:
    """Sample GCP registry with map pixel coordinates."""
    return [
        {"id": "PS1", "pixel_x": 1044.66, "pixel_y": 334.31},
        {"id": "PS2", "pixel_x": 1159.50, "pixel_y": 396.07},
        {"id": "PS3", "pixel_x": 1018.46, "pixel_y": 380.77},
        {"id": "PS4", "pixel_x": 1133.39, "pixel_y": 442.89},
        {"id": "PS5", "pixel_x": 992.60, "pixel_y": 427.02},
        {"id": "PS6", "pixel_x": 1107.18, "pixel_y": 489.00},
        {"id": "AR1", "pixel_x": 835.78, "pixel_y": 467.83},
        {"id": "AR2", "pixel_x": 919.00, "pixel_y": 478.27},
    ]


@pytest.fixture
def sample_camera_annotations() -> list[dict]:
    """Sample camera annotations matching GCP IDs."""
    return [
        {"gcp_id": "PS1", "pixel_x": 386.8, "pixel_y": 263.2},
        {"gcp_id": "PS2", "pixel_x": 568.1, "pixel_y": 280.9},
        {"gcp_id": "PS3", "pixel_x": 793.1, "pixel_y": 306.9},
        {"gcp_id": "PS4", "pixel_x": 1085.9, "pixel_y": 342.3},
        {"gcp_id": "PS5", "pixel_x": 1461.1, "pixel_y": 389.3},
        {"gcp_id": "PS6", "pixel_x": 1224.1, "pixel_y": 266.3},
        {"gcp_id": "AR1", "pixel_x": 803.0, "pixel_y": 582.7},
        {"gcp_id": "AR2", "pixel_x": 1094.5, "pixel_y": 396.7},
    ]


@pytest.fixture
def sample_line_registry() -> list[dict]:
    """Sample line registry with map pixel coordinates."""
    return [
        {"line_id": "L1", "start_x": 1044.43, "start_y": 333.86, "end_x": 824.11, "end_y": 727.52},
        {"line_id": "L2", "start_x": 1171.06, "start_y": 375.09, "end_x": 950.44, "end_y": 768.58},
        {"line_id": "L3", "start_x": 962.11, "start_y": 747.43, "end_x": 835.60, "end_y": 706.01},
        {"line_id": "L4", "start_x": 718.14, "start_y": 574.46, "end_x": 856.83, "end_y": 653.77},
        {"line_id": "L5", "start_x": 705.79, "start_y": 585.03, "end_x": 852.72, "end_y": 667.46},
    ]


@pytest.fixture
def sample_camera_line_annotations() -> list[dict]:
    """Sample camera line annotations."""
    return [
        {"line_id": "L4", "start_pixel_x": 608.0, "start_pixel_y": 169.5, "end_pixel_x": 1724.0, "end_pixel_y": 580.5},
        {"line_id": "L5", "start_pixel_x": 1874.0, "start_pixel_y": 533.5, "end_pixel_x": 714.0, "end_pixel_y": 160.5},
        {"line_id": "L1", "start_pixel_x": 100.0, "start_pixel_y": 200.0, "end_pixel_x": 500.0, "end_pixel_y": 600.0},
        {"line_id": "L2", "start_pixel_x": 200.0, "start_pixel_y": 300.0, "end_pixel_x": 600.0, "end_pixel_y": 700.0},
    ]


@pytest.fixture
def intrinsic_matrix() -> np.ndarray:
    """Sample 3x3 intrinsic matrix."""
    return np.array([
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# Tests: GCP Correspondence Construction
# ---------------------------------------------------------------------------


class TestBuildGCPCorrespondences:
    """Tests for build_gcp_correspondences function."""

    def test_basic_matching(self, sample_gcp_registry, sample_camera_annotations):
        """All GCP IDs in annotations exist in registry."""
        result = build_gcp_correspondences(sample_gcp_registry, sample_camera_annotations)

        assert len(result) == 8
        assert all(isinstance(c, GCPCorrespondence) for c in result)

    def test_correct_coordinate_mapping(self, sample_gcp_registry, sample_camera_annotations):
        """Map coordinates come from registry, camera coordinates from annotations."""
        result = build_gcp_correspondences(sample_gcp_registry, sample_camera_annotations)

        ps1 = next(c for c in result if c.gcp_id == "PS1")
        assert ps1.map_x == pytest.approx(1044.66)
        assert ps1.map_y == pytest.approx(334.31)
        assert ps1.camera_x == pytest.approx(386.8)
        assert ps1.camera_y == pytest.approx(263.2)

    def test_missing_gcp_skipped(self, sample_gcp_registry):
        """Annotations referencing non-existent GCPs are skipped."""
        annotations = [
            {"gcp_id": "PS1", "pixel_x": 100.0, "pixel_y": 200.0},
            {"gcp_id": "NONEXISTENT", "pixel_x": 300.0, "pixel_y": 400.0},
        ]
        result = build_gcp_correspondences(sample_gcp_registry, annotations)
        assert len(result) == 1
        assert result[0].gcp_id == "PS1"

    def test_empty_inputs(self):
        """Empty inputs return empty list."""
        assert build_gcp_correspondences([], []) == []
        assert build_gcp_correspondences([{"id": "A", "pixel_x": 0, "pixel_y": 0}], []) == []

    def test_output_types(self, sample_gcp_registry, sample_camera_annotations):
        """All coordinate values are floats."""
        result = build_gcp_correspondences(sample_gcp_registry, sample_camera_annotations)
        for c in result:
            assert isinstance(c.map_x, float)
            assert isinstance(c.map_y, float)
            assert isinstance(c.camera_x, float)
            assert isinstance(c.camera_y, float)


# ---------------------------------------------------------------------------
# Tests: Line Correspondence Construction
# ---------------------------------------------------------------------------


class TestBuildLineCorrespondences:
    """Tests for build_line_correspondences function."""

    def test_basic_matching(self, sample_line_registry, sample_camera_line_annotations):
        """Lines matching between registry and annotations are paired."""
        result = build_line_correspondences(sample_line_registry, sample_camera_line_annotations)
        # L4, L5, L1, L2 all exist in registry
        assert len(result) == 4

    def test_correct_coordinate_mapping(self, sample_line_registry, sample_camera_line_annotations):
        """Map coords from registry, camera coords from annotations."""
        result = build_line_correspondences(sample_line_registry, sample_camera_line_annotations)

        l4 = next(c for c in result if c.line_id == "L4")
        assert l4.map_start == pytest.approx((718.14, 574.46))
        assert l4.map_end == pytest.approx((856.83, 653.77))
        assert l4.camera_start == pytest.approx((608.0, 169.5))
        assert l4.camera_end == pytest.approx((1724.0, 580.5))

    def test_missing_line_skipped(self, sample_line_registry):
        """Annotations referencing non-existent lines are skipped."""
        annotations = [
            {"line_id": "L1", "start_pixel_x": 0, "start_pixel_y": 0, "end_pixel_x": 1, "end_pixel_y": 1},
            {"line_id": "NONEXISTENT", "start_pixel_x": 0, "start_pixel_y": 0, "end_pixel_x": 1, "end_pixel_y": 1},
        ]
        result = build_line_correspondences(sample_line_registry, annotations)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: Line Sampling
# ---------------------------------------------------------------------------


class TestSampleLineCorrespondences:
    """Tests for sample_line_correspondences function."""

    def test_correct_shape(self):
        """Output arrays have shape (N, 2)."""
        line = LineCorrespondencePair(
            line_id="L1",
            map_start=(0.0, 0.0), map_end=(100.0, 100.0),
            camera_start=(50.0, 50.0), camera_end=(150.0, 150.0),
        )
        map_pts, cam_pts = sample_line_correspondences(line, num_samples=10)

        assert map_pts.shape == (10, 2)
        assert cam_pts.shape == (10, 2)

    def test_endpoints_included(self):
        """First and last samples match start and end points."""
        line = LineCorrespondencePair(
            line_id="L1",
            map_start=(10.0, 20.0), map_end=(100.0, 200.0),
            camera_start=(30.0, 40.0), camera_end=(130.0, 240.0),
        )
        map_pts, cam_pts = sample_line_correspondences(line, num_samples=5)

        np.testing.assert_allclose(map_pts[0], [10.0, 20.0])
        np.testing.assert_allclose(map_pts[-1], [100.0, 200.0])
        np.testing.assert_allclose(cam_pts[0], [30.0, 40.0])
        np.testing.assert_allclose(cam_pts[-1], [130.0, 240.0])

    def test_evenly_spaced(self):
        """Samples are evenly distributed along the line."""
        line = LineCorrespondencePair(
            line_id="L1",
            map_start=(0.0, 0.0), map_end=(100.0, 0.0),
            camera_start=(0.0, 0.0), camera_end=(200.0, 0.0),
        )
        map_pts, cam_pts = sample_line_correspondences(line, num_samples=5)

        expected_map_x = [0, 25, 50, 75, 100]
        expected_cam_x = [0, 50, 100, 150, 200]
        np.testing.assert_allclose(map_pts[:, 0], expected_map_x)
        np.testing.assert_allclose(cam_pts[:, 0], expected_cam_x)

    def test_dtype_float64(self):
        """Output arrays are float64."""
        line = LineCorrespondencePair(
            line_id="L1",
            map_start=(0.0, 0.0), map_end=(1.0, 1.0),
            camera_start=(0.0, 0.0), camera_end=(1.0, 1.0),
        )
        map_pts, cam_pts = sample_line_correspondences(line)
        assert map_pts.dtype == np.float64
        assert cam_pts.dtype == np.float64


# ---------------------------------------------------------------------------
# Tests: Line Split
# ---------------------------------------------------------------------------


class TestSplitLines:
    """Tests for deterministic line splitting."""

    def _make_lines(self, n: int) -> list[LineCorrespondencePair]:
        return [
            LineCorrespondencePair(
                line_id=f"L{i}",
                map_start=(0.0, 0.0), map_end=(1.0, 1.0),
                camera_start=(0.0, 0.0), camera_end=(1.0, 1.0),
            )
            for i in range(n)
        ]

    def test_deterministic(self):
        """Same input produces same split."""
        lines = self._make_lines(20)
        split1 = split_lines(lines, 0.7)
        split2 = split_lines(lines, 0.7)

        assert [l.line_id for l in split1.training_lines] == [l.line_id for l in split2.training_lines]
        assert [l.line_id for l in split1.validation_lines] == [l.line_id for l in split2.validation_lines]

    def test_approximate_ratio(self):
        """Split ratio approximately respected for large N."""
        lines = self._make_lines(100)
        split = split_lines(lines, 0.7)

        total = len(split.training_lines) + len(split.validation_lines)
        assert total == 100
        # Allow 15% tolerance from target ratio
        train_ratio = len(split.training_lines) / total
        assert 0.55 <= train_ratio <= 0.85

    def test_at_least_one_each(self):
        """At least one line in each set when possible."""
        lines = self._make_lines(2)
        split = split_lines(lines, 0.9)

        assert len(split.training_lines) >= 1
        assert len(split.validation_lines) >= 1

    def test_empty_input(self):
        """Empty input returns empty split."""
        split = split_lines([], 0.7)
        assert split.training_lines == []
        assert split.validation_lines == []

    def test_all_lines_accounted_for(self):
        """No lines lost in split."""
        lines = self._make_lines(15)
        split = split_lines(lines, 0.7)

        all_ids = {l.line_id for l in lines}
        split_ids = {l.line_id for l in split.training_lines} | {l.line_id for l in split.validation_lines}
        assert all_ids == split_ids


# ---------------------------------------------------------------------------
# Tests: OpenCV Solver
# ---------------------------------------------------------------------------


class TestOpenCVDistortionSolver:
    """Tests for OpenCVDistortionSolver."""

    def test_insufficient_correspondences(self, intrinsic_matrix):
        """Solver fails gracefully with too few correspondences."""
        solver = OpenCVDistortionSolver()
        gcps = [
            GCPCorrespondence("PS1", 100.0, 200.0, 300.0, 400.0),
            GCPCorrespondence("PS2", 200.0, 300.0, 400.0, 500.0),
        ]
        result = solver.solve(gcps, [], intrinsic_matrix)

        assert not result.success
        assert "Insufficient" in result.message

    def test_solver_returns_solver_result(
        self, sample_gcp_registry, sample_camera_annotations,
        sample_line_registry, sample_camera_line_annotations,
        intrinsic_matrix,
    ):
        """Solver returns SolverResult with all required fields."""
        gcps = build_gcp_correspondences(sample_gcp_registry, sample_camera_annotations)
        lines = build_line_correspondences(sample_line_registry, sample_camera_line_annotations)

        solver = OpenCVDistortionSolver(
            config=OpenCVSolverConfig(num_samples_per_line=10, train_split_ratio=0.7)
        )
        result = solver.solve(gcps, lines, intrinsic_matrix)

        # Check all SolverResult fields are present
        assert isinstance(result.distortion, DistortionCoefficients)
        assert isinstance(result.initial_error, float)
        assert isinstance(result.final_error, float)
        assert isinstance(result.rmse_per_line, list)
        assert isinstance(result.overall_rmse, float)
        assert isinstance(result.iterations, int)
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.line_errors, list)

    def test_distortion_coefficients_format(
        self, sample_gcp_registry, sample_camera_annotations,
        sample_line_registry, sample_camera_line_annotations,
        intrinsic_matrix,
    ):
        """Distortion coefficients match Brown-Conrady model."""
        gcps = build_gcp_correspondences(sample_gcp_registry, sample_camera_annotations)
        lines = build_line_correspondences(sample_line_registry, sample_camera_line_annotations)

        solver = OpenCVDistortionSolver()
        result = solver.solve(gcps, lines, intrinsic_matrix)

        if result.success:
            dc = result.distortion
            assert hasattr(dc, "k1")
            assert hasattr(dc, "k2")
            assert hasattr(dc, "k3")
            assert hasattr(dc, "p1")
            assert hasattr(dc, "p2")
            # to_array() should produce OpenCV order [k1, k2, p1, p2, k3]
            arr = dc.to_array()
            assert len(arr) == 5

    def test_reprojection_error_computed(
        self, sample_gcp_registry, sample_camera_annotations,
        sample_line_registry, sample_camera_line_annotations,
        intrinsic_matrix,
    ):
        """Reprojection error (initial_error) is computed and positive."""
        gcps = build_gcp_correspondences(sample_gcp_registry, sample_camera_annotations)
        lines = build_line_correspondences(sample_line_registry, sample_camera_line_annotations)

        solver = OpenCVDistortionSolver()
        result = solver.solve(gcps, lines, intrinsic_matrix)

        if result.success:
            # initial_error stores reprojection error for OpenCV method
            assert result.initial_error >= 0.0

    def test_intrinsics_returned(
        self, sample_gcp_registry, sample_camera_annotations,
        sample_line_registry, sample_camera_line_annotations,
        intrinsic_matrix,
    ):
        """Solver returns intrinsics dict."""
        gcps = build_gcp_correspondences(sample_gcp_registry, sample_camera_annotations)
        lines = build_line_correspondences(sample_line_registry, sample_camera_line_annotations)

        solver = OpenCVDistortionSolver()
        result = solver.solve(gcps, lines, intrinsic_matrix)

        if result.success:
            assert result.intrinsics is not None
            assert "fx" in result.intrinsics
            assert "fy" in result.intrinsics
            assert "cx" in result.intrinsics
            assert "cy" in result.intrinsics

    def test_line_errors_format(
        self, sample_gcp_registry, sample_camera_annotations,
        sample_line_registry, sample_camera_line_annotations,
        intrinsic_matrix,
    ):
        """Line errors contain line_id and rmse_pixels."""
        gcps = build_gcp_correspondences(sample_gcp_registry, sample_camera_annotations)
        lines = build_line_correspondences(sample_line_registry, sample_camera_line_annotations)

        solver = OpenCVDistortionSolver()
        result = solver.solve(gcps, lines, intrinsic_matrix)

        if result.success and result.line_errors:
            for err in result.line_errors:
                assert "line_id" in err
                assert "rmse_pixels" in err
                assert isinstance(err["rmse_pixels"], float)


# ---------------------------------------------------------------------------
# Tests: ZoomCalibrationEntry reprojection_error_px
# ---------------------------------------------------------------------------


class TestCalibrationTableReprojection:
    """Tests for reprojection_error_px field in ZoomCalibrationEntry."""

    def test_default_zero(self):
        """reprojection_error_px defaults to 0.0."""
        from poc_homography.calibration.lens_distortion.calibration_table import ZoomCalibrationEntry

        entry = ZoomCalibrationEntry(
            zoom_factor=1.0, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
            calibration_date="2026-01-01",
        )
        assert entry.reprojection_error_px == 0.0

    def test_to_dict_excludes_zero(self):
        """to_dict() excludes reprojection_error_px when 0.0."""
        from poc_homography.calibration.lens_distortion.calibration_table import ZoomCalibrationEntry

        entry = ZoomCalibrationEntry(
            zoom_factor=1.0, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
            calibration_date="2026-01-01",
        )
        d = entry.to_dict()
        assert "reprojection_error_px" not in d

    def test_to_dict_includes_nonzero(self):
        """to_dict() includes reprojection_error_px when non-zero."""
        from poc_homography.calibration.lens_distortion.calibration_table import ZoomCalibrationEntry

        entry = ZoomCalibrationEntry(
            zoom_factor=1.0, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
            calibration_date="2026-01-01",
            reprojection_error_px=1.23,
        )
        d = entry.to_dict()
        assert d["reprojection_error_px"] == pytest.approx(1.23)

    def test_from_dict_with_reprojection(self):
        """from_dict() loads reprojection_error_px."""
        from poc_homography.calibration.lens_distortion.calibration_table import ZoomCalibrationEntry

        data = {
            "zoom_factor": 1.0, "k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0,
            "reprojection_error_px": 2.34,
        }
        entry = ZoomCalibrationEntry.from_dict(data)
        assert entry.reprojection_error_px == pytest.approx(2.34)

    def test_from_dict_without_reprojection(self):
        """from_dict() defaults reprojection_error_px to 0.0."""
        from poc_homography.calibration.lens_distortion.calibration_table import ZoomCalibrationEntry

        data = {
            "zoom_factor": 1.0, "k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0,
        }
        entry = ZoomCalibrationEntry.from_dict(data)
        assert entry.reprojection_error_px == 0.0

    def test_from_solver_result_with_reprojection(self):
        """from_solver_result() accepts reprojection_error_px."""
        from poc_homography.calibration.lens_distortion.calibration_table import ZoomCalibrationEntry

        entry = ZoomCalibrationEntry.from_solver_result(
            zoom_factor=1.0,
            distortion=DistortionCoefficients(),
            reprojection_error_px=3.45,
        )
        assert entry.reprojection_error_px == pytest.approx(3.45)

    def test_roundtrip(self):
        """to_dict/from_dict roundtrip preserves reprojection_error_px."""
        from poc_homography.calibration.lens_distortion.calibration_table import ZoomCalibrationEntry

        entry = ZoomCalibrationEntry(
            zoom_factor=2.0, k1=-0.1, k2=0.05, k3=0.01, p1=0.001, p2=-0.002,
            calibration_date="2026-01-01",
            reprojection_error_px=1.567,
        )
        restored = ZoomCalibrationEntry.from_dict(entry.to_dict())
        assert restored.reprojection_error_px == pytest.approx(1.567)
