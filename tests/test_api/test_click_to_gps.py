"""Integration tests for the click-to-GPS endpoints (issue #33).

These patch the router's frame/registry/homography helpers so the endpoint runs
without a real database, filesystem, or camera. A real :class:`GeoTiff` is used
so the pixel→UTM→lat/lon chain is exercised end to end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.map_points.map_point import MapPoint
from poc_homography.types import Easting, Meters, Northing, Unitless

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

PROJECT_URL = "/click-to-gps/api/project/?tenant_id=t1"
FRAMES_URL = "/click-to-gps/api/frames/?tenant_id=t1"

_MODULE = "api.routers.click_to_gps"


def _valte_geotiff() -> GeoTiff:
    """A real georeferenced GeoTiff (Valte map params, EPSG:25830)."""
    return GeoTiff(
        geotransform=GeoTransform(
            origin_easting=Easting(737575.05),
            pixel_width=Meters(0.15),
            row_rotation=Unitless(0.0),
            origin_northing=Northing(4391595.45),
            col_rotation=Unitless(0.0),
            pixel_height=Meters(-0.15),
        ),
        crs="EPSG:25830",
    )


def _frame(stem: str = "frame1") -> MagicMock:
    """A stand-in CapturedFrame exposing the attributes the router reads."""
    frame = MagicMock()
    frame.id = f"id-{stem}"
    frame.image_path.stem = stem
    frame.image_path.name = f"{stem}.jpg"
    return frame


def _four_annotations() -> list[dict]:
    return [
        {"gcp_id": "G1", "pixel_x": 10.0, "pixel_y": 20.0},
        {"gcp_id": "G2", "pixel_x": 30.0, "pixel_y": 40.0},
        {"gcp_id": "G3", "pixel_x": 50.0, "pixel_y": 60.0},
        {"gcp_id": "G4", "pixel_x": 70.0, "pixel_y": 80.0},
    ]


def _homography_returning(map_x: float, map_y: float, inlier_ratio: float = 0.9) -> MagicMock:
    """A MapPointHomography mock projecting every click to (map_x, map_y)."""
    instance = MagicMock()
    result = MagicMock()
    result.inlier_ratio = inlier_ratio
    instance.compute_from_gcps.return_value = result
    instance.camera_to_map.return_value = MapPoint(pixel_x=map_x, pixel_y=map_y)
    cls = MagicMock(return_value=instance)
    return cls


class TestProject:
    """Tests for ``POST /click-to-gps/api/project/``."""

    @patch(f"{_MODULE}.MapPointHomography")
    @patch(f"{_MODULE}.from_gcp_repo_pg")
    @patch(f"{_MODULE}._load_geotiff")
    @patch(f"{_MODULE}.load_annotations_for_frame")
    @patch(f"{_MODULE}.list_frames")
    @patch(f"{_MODULE}.get_map_for_tenant")
    def test_project_success_returns_gps(
        self,
        mock_map: MagicMock,
        mock_frames: MagicMock,
        mock_anns: MagicMock,
        mock_geotiff: MagicMock,
        mock_registry: MagicMock,
        mock_homography: MagicMock,
        client: TestClient,
    ) -> None:
        """A valid click projects to a WGS84 coordinate with confidence."""
        mock_map.return_value = MagicMock(id="map1")
        mock_frames.return_value = [_frame("frame1")]
        mock_anns.return_value = _four_annotations()
        mock_geotiff.return_value = (_valte_geotiff(), 640, 640)
        mock_registry.return_value = MagicMock(map_id="map1")
        # Map pixel (0,0) → the Valte origin GPS (~39.641, -0.231).
        mock_homography.side_effect = _homography_returning(0.0, 0.0, inlier_ratio=0.82)

        resp = client.post(
            PROJECT_URL,
            json={"test_case_name": "frame1", "pixel_x": 123.0, "pixel_y": 456.0},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["latitude"] == pytest.approx(39.6412, abs=1e-3)
        assert data["longitude"] == pytest.approx(-0.2314, abs=1e-3)
        assert data["confidence"] == pytest.approx(0.82, abs=1e-6)
        assert data["crs"] == "EPSG:25830"
        assert data["on_horizon"] is False

    @patch(f"{_MODULE}.MapPointHomography")
    @patch(f"{_MODULE}.from_gcp_repo_pg")
    @patch(f"{_MODULE}._load_geotiff")
    @patch(f"{_MODULE}.load_annotations_for_frame")
    @patch(f"{_MODULE}.list_frames")
    @patch(f"{_MODULE}.get_map_for_tenant")
    def test_point_beyond_map_is_flagged_on_horizon(
        self,
        mock_map: MagicMock,
        mock_frames: MagicMock,
        mock_anns: MagicMock,
        mock_geotiff: MagicMock,
        mock_registry: MagicMock,
        mock_homography: MagicMock,
        client: TestClient,
    ) -> None:
        """A pixel projecting far outside the map is reported as on-horizon."""
        mock_map.return_value = MagicMock(id="map1")
        mock_frames.return_value = [_frame("frame1")]
        mock_anns.return_value = _four_annotations()
        mock_geotiff.return_value = (_valte_geotiff(), 640, 640)
        mock_registry.return_value = MagicMock(map_id="map1")
        # Project way outside the 640×640 map → on horizon.
        mock_homography.side_effect = _homography_returning(99999.0, 99999.0)

        resp = client.post(
            PROJECT_URL,
            json={"test_case_name": "frame1", "pixel_x": 1.0, "pixel_y": 1.0},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["on_horizon"] is True
        assert data["latitude"] is None

    @patch(f"{_MODULE}.load_annotations_for_frame")
    @patch(f"{_MODULE}.list_frames")
    @patch(f"{_MODULE}.get_map_for_tenant")
    def test_insufficient_annotations_returns_400(
        self,
        mock_map: MagicMock,
        mock_frames: MagicMock,
        mock_anns: MagicMock,
        client: TestClient,
    ) -> None:
        """A frame with fewer than four annotations cannot project."""
        mock_map.return_value = MagicMock(id="map1")
        mock_frames.return_value = [_frame("frame1")]
        mock_anns.return_value = _four_annotations()[:2]

        resp = client.post(
            PROJECT_URL,
            json={"test_case_name": "frame1", "pixel_x": 1.0, "pixel_y": 1.0},
        )

        assert resp.status_code == 400

    @patch(f"{_MODULE}.get_map_for_tenant")
    def test_no_map_returns_404(self, mock_map: MagicMock, client: TestClient) -> None:
        """No configured map → 404, matching the sibling routers' convention."""
        mock_map.return_value = None

        resp = client.post(
            PROJECT_URL,
            json={"test_case_name": "frame1", "pixel_x": 1.0, "pixel_y": 1.0},
        )

        assert resp.status_code == 404


class TestListFrames:
    """Tests for ``GET /click-to-gps/api/frames/``."""

    @patch(f"{_MODULE}.load_annotations_for_frame")
    @patch(f"{_MODULE}.list_frames")
    @patch(f"{_MODULE}.get_map_for_tenant")
    def test_only_frames_with_enough_annotations_listed(
        self,
        mock_map: MagicMock,
        mock_frames: MagicMock,
        mock_anns: MagicMock,
        client: TestClient,
    ) -> None:
        """Frames with fewer than four annotations are excluded."""
        mock_map.return_value = MagicMock(id="map1")
        mock_frames.return_value = [_frame("good"), _frame("sparse")]
        mock_anns.side_effect = lambda frame_id, _session: (
            _four_annotations() if frame_id == "id-good" else _four_annotations()[:1]
        )

        resp = client.get(FRAMES_URL)

        assert resp.status_code == 200
        names = [f["name"] for f in resp.json()]
        assert names == ["good"]
