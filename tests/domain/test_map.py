"""Unit tests for the Map domain entity serialization."""

from __future__ import annotations

from pathlib import Path

from poc_homography.domain.entities.map import Map
from poc_homography.domain.vo.geotiff import GeoTiff, GeoTransform
from poc_homography.domain.vo.photo import Photo
from poc_homography.types import (
    Easting,
    Meters,
    Northing,
    Pixels,
    Unitless,
)


def _make_map(**overrides: object) -> Map:
    base = {
        "id": "valte",
        "tenant_id": "valte",
        "photo": Photo(path=Path("valte.png"), width=Pixels(1000), height=Pixels(800)),
        "geotiff": GeoTiff(
            geotransform=GeoTransform(
                origin_easting=Easting(500000.0),
                pixel_width=Meters(0.5),
                row_rotation=Unitless(0.0),
                origin_northing=Northing(4400000.0),
                col_rotation=Unitless(0.0),
                pixel_height=Meters(-0.5),
            ),
            crs="EPSG:25830",
        ),
    }
    base.update(overrides)
    return Map(**base)  # type: ignore[arg-type]


def test_asset_ref_defaults_to_none() -> None:
    m = _make_map()
    assert m.asset_key is None
    assert m.asset_url is None
    data = m.to_dict()
    assert data["asset_key"] is None
    assert data["asset_url"] is None


def test_to_dict_from_dict_round_trips_asset_ref() -> None:
    m = _make_map(
        asset_key="maps/valte.tif",
        asset_url="https://s3.example.test/cleanplate/maps/valte.tif",
    )
    restored = Map.from_dict(m.to_dict())
    assert restored.asset_key == "maps/valte.tif"
    assert restored.asset_url == "https://s3.example.test/cleanplate/maps/valte.tif"


def test_from_dict_tolerates_missing_asset_ref() -> None:
    """Rows persisted before the migration carry no asset fields."""
    data = _make_map().to_dict()
    del data["asset_key"]
    del data["asset_url"]
    restored = Map.from_dict(data)
    assert restored.asset_key is None
    assert restored.asset_url is None
