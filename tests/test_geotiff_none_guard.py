"""Tests for _get_map_geotiff_file None guard (Bug #1).

Ensures no crash when map_id is None and that _get_map_info handles missing maps.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Django setup (same pattern as test_camera_diagnostic.py)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "homography_web.settings")
sys.path.insert(0, str(Path(__file__).parent.parent / "webapp"))

import django

django.setup()

from homography_precision.views import _get_map_geotiff_file, _get_map_info


class TestGetMapGeotiffFile:
    def test_returns_none_when_no_map(self, monkeypatch: object) -> None:
        """When get_map_from_tenant_id() returns None, _get_map_geotiff_file must return None."""
        import homography_precision.views as views_mod

        monkeypatch.setattr(views_mod, "get_map_from_tenant_id", lambda _tid: None)  # type: ignore[attr-defined]
        assert _get_map_geotiff_file("nonexistent") is None

    def test_returns_path_when_map_exists(self, monkeypatch: object, tmp_path: Path) -> None:
        import homography_precision.views as views_mod

        fake_tif = tmp_path / "testmap.tif"
        fake_tif.touch()
        fake_entity = MagicMock()
        fake_entity.photo.path = Path("testmap.tif")
        monkeypatch.setattr(views_mod, "get_map_from_tenant_id", lambda _tid: fake_entity)  # type: ignore[attr-defined]
        monkeypatch.setattr(views_mod, "DATA_MAPS_DIR", tmp_path)  # type: ignore[attr-defined]
        result = _get_map_geotiff_file("test_tenant")
        assert result is not None
        assert result.name == "testmap.tif"

    def test_no_crash_on_none_map_id(self, monkeypatch: object) -> None:
        """_get_map_info must return None (not crash) when no map is configured."""
        import homography_precision.views as views_mod

        monkeypatch.setattr(views_mod, "get_map_from_tenant_id", lambda _tid: None)  # type: ignore[attr-defined]
        # Clear the cache so _get_map_info re-evaluates
        views_mod._image_info_cache.clear()  # type: ignore[attr-defined]
        assert _get_map_info("nonexistent") is None
