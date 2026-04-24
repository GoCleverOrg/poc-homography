"""Integration tests for the GCP router (tenants, maps, map IDs)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from poc_homography.domain.entities.map import Map
from poc_homography.domain.entities.tenant import Tenant
from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.domain.vo.photo import Photo
from poc_homography.types import Easting, Meters, Northing, Pixels, Unitless

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------


def _make_tenant(tid: str = "acme", name: str = "Acme Corp") -> Tenant:
    return Tenant(id=tid, name=name, description="Test tenant")


def _make_map(mid: str = "acme-map", tenant_id: str = "acme") -> Map:
    photo = Photo(path=Path("/fake/map.png"), width=Pixels(1024), height=Pixels(768))
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


# ---------------------------------------------------------------------------
# GET /gcp/api/tenants/
# ---------------------------------------------------------------------------


class TestGetTenants:
    """Tests for ``GET /gcp/api/tenants/``."""

    @patch("api.routers.gcp.RepoYamlTenant")
    def test_returns_tenant_list(
        self, mock_repo_cls: object, client: TestClient
    ) -> None:
        mock_instance = mock_repo_cls.return_value  # type: ignore[union-attr]
        mock_instance.get_all.return_value = [
            _make_tenant("t1", "Tenant One"),
            _make_tenant("t2", "Tenant Two"),
        ]

        resp = client.get("/gcp/api/tenants/")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tenants"]) == 2
        assert data["tenants"][0]["id"] == "t1"
        assert data["tenants"][1]["name"] == "Tenant Two"

    @patch("api.routers.gcp.RepoYamlTenant")
    def test_returns_empty_list(
        self, mock_repo_cls: object, client: TestClient
    ) -> None:
        mock_instance = mock_repo_cls.return_value  # type: ignore[union-attr]
        mock_instance.get_all.return_value = []

        resp = client.get("/gcp/api/tenants/")

        assert resp.status_code == 200
        assert resp.json() == {"tenants": []}


# ---------------------------------------------------------------------------
# GET /gcp/api/tenants/{tenant_id}/maps/
# ---------------------------------------------------------------------------


class TestGetTenantMaps:
    """Tests for ``GET /gcp/api/tenants/{tenant_id}/maps/``."""

    @patch("api.routers.gcp.RepoYamlMap")
    def test_returns_maps_for_tenant(
        self, mock_repo_cls: object, client: TestClient
    ) -> None:
        m = _make_map("map-1", "acme")
        mock_instance = mock_repo_cls.return_value  # type: ignore[union-attr]
        mock_instance.get_by_tenant.return_value = {"map-1": m}

        resp = client.get("/gcp/api/tenants/acme/maps/")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["maps"]) == 1
        assert data["maps"][0]["id"] == "map-1"
        assert data["maps"][0]["tenant_id"] == "acme"

    @patch("api.routers.gcp.RepoYamlMap")
    def test_returns_empty_when_no_maps(
        self, mock_repo_cls: object, client: TestClient
    ) -> None:
        mock_instance = mock_repo_cls.return_value  # type: ignore[union-attr]
        mock_instance.get_by_tenant.return_value = {}

        resp = client.get("/gcp/api/tenants/acme/maps/")

        assert resp.status_code == 200
        assert resp.json() == {"maps": []}


# ---------------------------------------------------------------------------
# GET /gcp/api/map-ids/
# ---------------------------------------------------------------------------


class TestGetMapIds:
    """Tests for ``GET /gcp/api/map-ids/?tenant_id=…``."""

    @patch("api.routers.gcp.list_map_ids")
    @patch("api.routers.gcp.RepoYamlMap")
    def test_returns_filtered_map_ids(
        self,
        mock_repo_cls: object,
        mock_list_ids: object,
        client: TestClient,
    ) -> None:
        mock_instance = mock_repo_cls.return_value  # type: ignore[union-attr]
        mock_instance.get_by_tenant.return_value = {
            "map-a": _make_map("map-a"),
            "map-b": _make_map("map-b"),
        }
        mock_list_ids.return_value = ["map-a", "map-c"]  # type: ignore[union-attr]

        resp = client.get("/gcp/api/map-ids/", params={"tenant_id": "acme"})

        assert resp.status_code == 200
        assert resp.json() == {"map_ids": ["map-a"]}

    def test_requires_tenant_id_query_param(self, client: TestClient) -> None:
        """Omitting ``tenant_id`` should return 422 (validation error)."""
        resp = client.get("/gcp/api/map-ids/")
        assert resp.status_code == 422
