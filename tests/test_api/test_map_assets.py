"""Unit + end-to-end tests for object-storage-backed map GeoTIFF resolution.

``resolve_map_geotiff`` materialises a map's GeoTIFF from object storage into a
``/tmp`` cache when the map carries an ``asset_key``, and falls back to the
legacy ``data/maps`` path otherwise. These tests inject a fake store (the
``client`` DI pattern mirrored at the resolver level) so they run fully offline.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from api.utils import map_assets
from api.utils.map_assets import resolve_map_geotiff
from api.utils.tiles import render_tile

if TYPE_CHECKING:
    import pytest


class _FakeStore:
    """Records ``get_map`` calls and serves bytes from an in-memory mapping."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.calls: list[str] = []

    def get_map(self, key: str) -> bytes:
        self.calls.append(key)
        return self.objects[key]


def _map(*, asset_key: str | None = None, path: str = "tenant/map.tif") -> SimpleNamespace:
    """Lightweight stand-in for a ``Map`` entity (only the fields used here)."""
    return SimpleNamespace(asset_key=asset_key, photo=SimpleNamespace(path=Path(path)))


def _tiny_geotiff_bytes(width: int = 64, height: int = 48) -> bytes:
    """Encode a small RGB TIFF in-memory (stands in for a map asset)."""
    arr = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()


def test_resolves_asset_key_via_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_assets, "_CACHE_DIR", tmp_path / "cache")
    data = b"II*\x00 geotiff bytes"
    store = _FakeStore({"tenant/map.tif": data})

    resolved = resolve_map_geotiff(_map(asset_key="tenant/map.tif"), store=store)

    assert resolved is not None
    assert resolved.read_bytes() == data
    assert store.calls == ["tenant/map.tif"]


def test_caches_download_across_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_assets, "_CACHE_DIR", tmp_path / "cache")
    store = _FakeStore({"tenant/map.tif": b"bytes"})
    entity = _map(asset_key="tenant/map.tif")

    first = resolve_map_geotiff(entity, store=store)
    second = resolve_map_geotiff(entity, store=store)

    assert first == second
    # Second resolution is served from the /tmp cache — no extra download.
    assert store.calls == ["tenant/map.tif"]


def test_falls_back_to_local_when_no_asset_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(map_assets, "DATA_MAPS_DIR", tmp_path)
    local = tmp_path / "tenant" / "map.tif"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local tiff")

    resolved = resolve_map_geotiff(_map(asset_key=None, path="tenant/map.tif"))

    assert resolved == local


def test_returns_none_when_no_asset_key_and_local_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(map_assets, "DATA_MAPS_DIR", tmp_path)

    resolved = resolve_map_geotiff(_map(asset_key=None, path="tenant/absent.tif"))

    assert resolved is None


def test_tile_serving_end_to_end_from_object_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download a GeoTIFF from the (fake) store and render a PNG tile from it."""
    monkeypatch.setattr(map_assets, "_CACHE_DIR", tmp_path / "cache")
    geotiff = _tiny_geotiff_bytes(width=64, height=48)
    store = _FakeStore({"tenant/map.tif": geotiff})

    resolved = resolve_map_geotiff(_map(asset_key="tenant/map.tif"), store=store)
    assert resolved is not None

    png = render_tile(image_path=resolved, width=64, height=48, x=0, y=0, z=0, size=256)

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
