"""Tests for _get_map_geotiff_file None guard (Bug #1).

Ensures no crash when map_id is None and that _get_map_info handles missing maps.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Django setup (same pattern as test_camera_diagnostic.py)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "homography_web.settings")
sys.path.insert(0, str(Path(__file__).parent.parent / "webapp"))

import django

django.setup()

from homography_precision.views import _get_map_geotiff_file, _get_map_info


class TestGetMapGeotiffFile:
    def test_returns_none_when_no_map(self, monkeypatch: object) -> None:
        """When get_default_map_id() returns None, _get_map_geotiff_file must return None."""
        import homography_precision.views as views_mod

        monkeypatch.setattr(views_mod, "get_default_map_id", lambda: None)  # type: ignore[attr-defined]
        assert _get_map_geotiff_file() is None

    def test_returns_path_when_map_exists(self, monkeypatch: object) -> None:
        import homography_precision.views as views_mod

        monkeypatch.setattr(views_mod, "get_default_map_id", lambda: "testmap")  # type: ignore[attr-defined]
        result = _get_map_geotiff_file()
        assert result is not None
        assert result.name == "testmap.tif"

    def test_no_crash_on_none_map_id(self, monkeypatch: object) -> None:
        """_get_map_info must return None (not crash) when no map is configured."""
        import homography_precision.views as views_mod

        monkeypatch.setattr(views_mod, "get_default_map_id", lambda: None)  # type: ignore[attr-defined]
        # Clear the cache so _get_map_info re-evaluates
        views_mod._image_info_cache.clear()  # type: ignore[attr-defined]
        assert _get_map_info() is None
