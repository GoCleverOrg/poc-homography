"""Integration tests for the point-picker router (map selector + map_id forwarding)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from poc_homography.domain.entities.map import Map
from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.domain.vo.photo import Photo
from poc_homography.types import Easting, Meters, Northing, Pixels, Unitless

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------


def _make_map(mid: str = "acme-map", tenant_id: str = "acme") -> Map:
    photo = Photo(path=Path("acme.tif"), width=Pixels(1024), height=Pixels(768))
    geotransform = GeoTransform(
        origin_easting=Easting(500_000.0),
        pixel_width=Meters(0.1),
        row_rotation=Unitless(0.0),
        origin_northing=Northing(4_500_000.0),
        col_rotation=Unitless(0.0),
        pixel_height=Meters(-0.1),
    )
    geotiff = GeoTiff(geotransform=geotransform, crs="EPSG:25830")
    return Map(id=mid, tenant_id=tenant_id, photo=photo, geotiff=geotiff)


def _fake_state(width: int = 1000, height: int = 800) -> MagicMock:
    """Return a minimal stand-in for ``PointPickerState``."""
    state = MagicMock()
    state.width = width
    state.height = height
    state.geotiff = None
    state.geotiff_path = Path("acme.tif")
    state.registry.map_id = "acme-map"
    state.registry.points = {}
    return state


# ---------------------------------------------------------------------------
# GET /point-picker/api/maps/
# ---------------------------------------------------------------------------


class TestListMaps:
    """Tests for ``GET /point-picker/api/maps/``."""

    @patch("api.routers.point_picker.map_has_image")
    @patch("api.routers.point_picker.RepoPostgresMap")
    def test_returns_maps_with_image_configured_flags(
        self,
        mock_repo_cls: object,
        mock_has_image: object,
        client: TestClient,
    ) -> None:
        configured = _make_map("map-configured", "acme")
        unconfigured = _make_map("map-unconfigured", "acme")
        mock_repo_cls.return_value.get_by_tenant.return_value = {  # type: ignore[union-attr]
            "map-unconfigured": unconfigured,
            "map-configured": configured,
        }
        # Configured map -> True, unconfigured -> False.
        mock_has_image.side_effect = lambda m: m.id == "map-configured"  # type: ignore[union-attr]

        resp = client.get("/point-picker/api/maps/", params={"tenant_id": "acme"})

        assert resp.status_code == 200
        data = resp.json()
        # Sorted by id for determinism.
        assert [m["id"] for m in data["maps"]] == ["map-configured", "map-unconfigured"]
        assert data["maps"][0] == {
            "id": "map-configured",
            "label": "map-configured",
            "image_configured": True,
        }
        assert data["maps"][1]["image_configured"] is False

    @patch("api.routers.point_picker.RepoPostgresMap")
    def test_returns_empty_when_no_maps(self, mock_repo_cls: object, client: TestClient) -> None:
        mock_repo_cls.return_value.get_by_tenant.return_value = {}  # type: ignore[union-attr]

        resp = client.get("/point-picker/api/maps/", params={"tenant_id": "acme"})

        assert resp.status_code == 200
        assert resp.json() == {"maps": []}

    def test_requires_tenant_id(self, client: TestClient) -> None:
        resp = client.get("/point-picker/api/maps/")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# map_id forwarding
# ---------------------------------------------------------------------------


class TestMapIdForwarding:
    """The image/points endpoints forward ``map_id`` to ``get_state``."""

    @patch("api.routers.point_picker.get_state")
    def test_image_info_forwards_map_id(self, mock_get_state: object, client: TestClient) -> None:
        mock_get_state.return_value = _fake_state()  # type: ignore[union-attr]

        resp = client.get(
            "/point-picker/api/image/info/",
            params={"tenant_id": "acme", "map_id": "map-7"},
        )

        assert resp.status_code == 200
        _, kwargs = mock_get_state.call_args  # type: ignore[union-attr]
        assert kwargs["map_id"] == "map-7"

    @patch("api.routers.point_picker.get_state")
    def test_list_points_forwards_map_id(self, mock_get_state: object, client: TestClient) -> None:
        mock_get_state.return_value = _fake_state()  # type: ignore[union-attr]

        resp = client.get(
            "/point-picker/api/points/",
            params={"tenant_id": "acme", "map_id": "map-9"},
        )

        assert resp.status_code == 200
        _, kwargs = mock_get_state.call_args  # type: ignore[union-attr]
        assert kwargs["map_id"] == "map-9"

    @patch("api.routers.point_picker.get_state")
    def test_map_id_defaults_to_none(self, mock_get_state: object, client: TestClient) -> None:
        mock_get_state.return_value = _fake_state()  # type: ignore[union-attr]

        resp = client.get("/point-picker/api/points/", params={"tenant_id": "acme"})

        assert resp.status_code == 200
        _, kwargs = mock_get_state.call_args  # type: ignore[union-attr]
        assert kwargs["map_id"] is None
