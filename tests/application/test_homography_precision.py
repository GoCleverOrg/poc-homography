#!/usr/bin/env python3
"""Unit tests for the homography-precision application service.

These tests exercise the framework-agnostic service extracted from the
``homography_precision`` Django views.  They use synthetic data and stubs so no
DVC-backed fixtures are required.

Run with: python -m pytest tests/application/test_homography_precision.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from poc_homography.application import homography_precision as svc
from poc_homography.domain.vo import PixelPoint
from poc_homography.map_points import GCPRegistry, MapPoint


class TestPerpendicularDistance:
    """Tests for ``perpendicular_distance``."""

    def test_point_off_horizontal_line(self):
        """Perpendicular distance to a horizontal segment equals the vertical offset."""
        p = np.array([1.0, 3.0])
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        assert svc.perpendicular_distance(p, a, b) == pytest.approx(3.0)

    def test_point_on_line_is_zero(self):
        """A point lying on the line has zero perpendicular distance."""
        p = np.array([5.0, 0.0])
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 0.0])
        assert svc.perpendicular_distance(p, a, b) == pytest.approx(0.0)

    def test_zero_length_line_returns_distance_to_point(self):
        """When a == b the result is the Euclidean distance from p to that point."""
        p = np.array([3.0, 4.0])
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        assert svc.perpendicular_distance(p, a, b) == pytest.approx(5.0)


class _FakeCoeffs:
    """Minimal stand-in for lens-distortion coefficients."""

    k1 = -0.1
    k2 = 0.2
    k3 = 0.0
    p1 = 0.01
    p2 = -0.02


class _FakeTable:
    """Minimal stand-in for a per-camera calibration table."""

    def get_coefficients(self, zoom_factor: float) -> _FakeCoeffs:
        return _FakeCoeffs()

    def get_intrinsics(self, zoom_factor: float) -> dict[str, float]:
        return {"fx": 1800.0, "fy": 1810.0, "cx": 1280.0, "cy": 720.0}


class TestBuildDistortionParams:
    """Tests for ``build_distortion_params``."""

    def test_returns_empty_when_table_missing(self, monkeypatch):
        """No calibration table -> empty dict."""
        monkeypatch.setattr(svc, "load_calibration_for_camera", lambda name, d: None)
        result = svc.build_distortion_params("cam1", 1.0, Path("/nope"))
        assert result == {}

    def test_returns_empty_when_intrinsics_missing(self, monkeypatch):
        """Table present but no intrinsics -> empty dict."""

        class _NoIntrinsics(_FakeTable):
            def get_intrinsics(self, zoom_factor: float):
                return None

        monkeypatch.setattr(svc, "load_calibration_for_camera", lambda name, d: _NoIntrinsics())
        assert svc.build_distortion_params("cam1", 1.0, Path("/x")) == {}

    def test_returns_empty_on_exception(self, monkeypatch):
        """A raising calibration loader is swallowed -> empty dict."""

        def _boom(name, d):
            raise RuntimeError("boom")

        monkeypatch.setattr(svc, "load_calibration_for_camera", _boom)
        assert svc.build_distortion_params("cam1", 1.0, Path("/x")) == {}

    def test_maps_coefficients_and_intrinsics(self, monkeypatch):
        """Coefficients + intrinsics are merged into the expected dict."""
        monkeypatch.setattr(svc, "load_calibration_for_camera", lambda name, d: _FakeTable())
        result = svc.build_distortion_params("cam1", 2.5, Path("/x"))
        assert result == {
            "k1": -0.1,
            "k2": 0.2,
            "k3": 0.0,
            "p1": 0.01,
            "p2": -0.02,
            "fx": 1800.0,
            "fy": 1810.0,
            "cx": 1280.0,
            "cy": 720.0,
        }


class _IdentityMapPoint:
    """Projection target exposing ``pixel_x``/``pixel_y`` like a ``MapPoint``."""

    def __init__(self, x: float, y: float) -> None:
        self.pixel_x = x
        self.pixel_y = y


class _IdentityHomography:
    """Identity projection stub for line-error aggregation tests."""

    def camera_to_map(self, camera_pixel: PixelPoint, point_id: str = "") -> _IdentityMapPoint:
        return _IdentityMapPoint(float(camera_pixel.x), float(camera_pixel.y))

    def map_to_camera(self, map_coord: PixelPoint) -> PixelPoint:
        return PixelPoint.create(float(map_coord.x), float(map_coord.y))


class TestComputeLineErrors:
    """Tests for ``compute_line_errors`` aggregation."""

    def _registry(self) -> dict[str, dict[str, float]]:
        return {
            "L1": {"start_x": 0.0, "start_y": 0.0, "end_x": 10.0, "end_y": 0.0},
            "L2": {"start_x": 0.0, "start_y": 5.0, "end_x": 10.0, "end_y": 5.0},
        }

    def _annotations(self) -> list[dict[str, float | str]]:
        return [
            {
                "line_id": "L1",
                "start_pixel_x": 0.0,
                "start_pixel_y": 0.0,
                "end_pixel_x": 10.0,
                "end_pixel_y": 0.0,
            },
            {
                "line_id": "L2",
                "start_pixel_x": 0.0,
                "start_pixel_y": 5.0,
                "end_pixel_x": 10.0,
                "end_pixel_y": 5.0,
            },
        ]

    def test_identity_projection_yields_zero_error(self):
        """With identity projection and matching geometry, all errors are zero."""
        result = svc.compute_line_errors(
            _IdentityHomography(), self._annotations(), self._registry()
        )
        payload = result.to_payload()
        assert payload["success"] is True
        assert payload["metrics"] == {
            "num_lines": 2,
            "mean_line_error": 0.0,
            "max_line_error": 0.0,
        }
        assert len(payload["per_line_errors"]) == 2
        first = payload["per_line_errors"][0]
        assert set(first) == {
            "line_id",
            "error_px",
            "start_error",
            "end_error",
            "map_start_error",
            "map_end_error",
        }
        overlays = payload["line_overlays"]
        assert set(overlays["camera"]) == {"annotations", "reprojected_lines"}
        assert set(overlays["map"]) == {"gcp_lines", "projected_lines"}

    def test_rounding_applied(self):
        """Per-line errors are rounded to two decimals."""
        registry = {"L1": {"start_x": 0.0, "start_y": 0.0, "end_x": 10.0, "end_y": 0.0}}
        annotations = [
            {
                "line_id": "L1",
                "start_pixel_x": 0.0,
                "start_pixel_y": 1.23456,
                "end_pixel_x": 10.0,
                "end_pixel_y": 1.23456,
            }
        ]
        result = svc.compute_line_errors(_IdentityHomography(), annotations, registry)
        per_line = result.to_payload()["per_line_errors"][0]
        # Camera-space perpendicular error of a parallel offset line == the offset.
        assert per_line["error_px"] == round(1.23456, 2)
        assert per_line["map_start_error"] == round(1.23456, 2)

    def test_missing_line_raises(self):
        """An annotation referencing an unknown line id raises with the line id."""
        annotations = [
            {
                "line_id": "MISSING",
                "start_pixel_x": 0.0,
                "start_pixel_y": 0.0,
                "end_pixel_x": 1.0,
                "end_pixel_y": 0.0,
            }
        ]
        with pytest.raises(svc.LineNotInRegistryError) as exc_info:
            svc.compute_line_errors(_IdentityHomography(), annotations, self._registry())
        assert exc_info.value.line_id == "MISSING"
        assert "MISSING" in str(exc_info.value)

    def test_to_payload_golden_structure(self):
        """Lock the full nested payload shape (keys + overlay structure) byte-for-byte.

        A single line that matches its registry definition under identity
        projection yields all-zero errors, so the entire payload is exactly
        predictable. This guards against silent drift in key names, nesting, or
        overlay layout that the key-set assertions above would not catch.
        """
        registry = {"L1": {"start_x": 0.0, "start_y": 0.0, "end_x": 10.0, "end_y": 0.0}}
        annotations = [
            {
                "line_id": "L1",
                "start_pixel_x": 0.0,
                "start_pixel_y": 0.0,
                "end_pixel_x": 10.0,
                "end_pixel_y": 0.0,
            }
        ]
        payload = svc.compute_line_errors(_IdentityHomography(), annotations, registry).to_payload()
        line = {"line_id": "L1", "start": [0.0, 0.0], "end": [10.0, 0.0]}
        assert payload == {
            "success": True,
            "metrics": {"num_lines": 1, "mean_line_error": 0.0, "max_line_error": 0.0},
            "per_line_errors": [
                {
                    "line_id": "L1",
                    "error_px": 0.0,
                    "start_error": 0.0,
                    "end_error": 0.0,
                    "map_start_error": 0.0,
                    "map_end_error": 0.0,
                }
            ],
            "line_overlays": {
                "camera": {"annotations": [line], "reprojected_lines": [line]},
                "map": {"gcp_lines": [line], "projected_lines": [line]},
            },
        }


class TestComputeGcpPrecision:
    """Tests for ``compute_gcp_precision`` aggregation over a real homography."""

    def test_identity_correspondences(self):
        """4 coincident camera/map points yield an identity-like homography.

        ``compute_from_gcps`` runs cv2.findHomography on synthetic data, so no
        DVC fixtures are required.  With camera == map coordinates the result is
        near-identity and all reprojection errors round to zero.
        """
        coords = {
            "G1": (0.0, 0.0),
            "G2": (100.0, 0.0),
            "G3": (100.0, 100.0),
            "G4": (0.0, 100.0),
            "G5": (50.0, 25.0),
        }
        registry = GCPRegistry(
            map_id="map-test",
            points={gid: MapPoint(pixel_x=x, pixel_y=y) for gid, (x, y) in coords.items()},
        )
        annotations = [
            {"gcp_id": gid, "pixel_x": x, "pixel_y": y} for gid, (x, y) in coords.items()
        ]

        result = svc.compute_gcp_precision(
            annotations=annotations,
            registry=registry,
            distortion={},
            ransac_threshold=50.0,
            min_inlier_ratio=0.5,
        )
        payload = result.to_payload()

        assert payload["success"] is True
        assert payload["metrics"]["num_gcps"] == 5
        assert set(payload["metrics"]) == {
            "num_gcps",
            "num_inliers",
            "inlier_ratio",
            "mean_reproj_error",
            "max_reproj_error",
            "rmse",
        }
        assert len(payload["per_point_errors"]) == 5
        first = payload["per_point_errors"][0]
        assert set(first) == {
            "gcp_id",
            "error_px",
            "camera_dx",
            "camera_dy",
            "map_dx",
            "map_dy",
            "camera_original",
            "camera_reprojected",
            "map_original",
            "map_projected",
        }
        # Identity correspondences -> negligible reprojection error.
        assert payload["metrics"]["mean_reproj_error"] == pytest.approx(0.0, abs=0.01)
        overlays = payload["overlays"]
        assert set(overlays["camera"]) == {"annotations", "reprojected_gcps"}
        assert set(overlays["map"]) == {"gcps", "projected_annotations"}
