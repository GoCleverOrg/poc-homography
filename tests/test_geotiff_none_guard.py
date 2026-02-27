"""Tests for resolve_map_file None guard (Bug #1).

Ensures no crash when the tenant has no map and that get_map_info handles
missing maps gracefully (returns None rather than raising).
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

from homography_precision.services import get_map_info, resolve_map_file


class TestResolveMapFile:
    def test_returns_none_when_no_map(self, monkeypatch: object) -> None:
        """When resolve_map_for_tenant() raises, resolve_map_file must return None."""
        import homography_precision.services as svc_mod

        monkeypatch.setattr(  # type: ignore[attr-defined]
            svc_mod,
            "resolve_map_for_tenant",
            _raise_runtime("No map configured for tenant: nonexistent"),
        )
        assert resolve_map_file("nonexistent") is None

    def test_returns_path_when_map_exists(self, monkeypatch: object, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        import homography_precision.services as svc_mod

        fake_tif = tmp_path / "testmap.tif"
        fake_tif.touch()
        fake_entity = MagicMock()
        monkeypatch.setattr(  # type: ignore[attr-defined]
            svc_mod,
            "resolve_map_for_tenant",
            lambda _tid: (fake_entity, fake_tif),
        )
        result = resolve_map_file("test_tenant")
        assert result is not None
        assert result.name == "testmap.tif"

    def test_no_crash_on_none_map_id(self, monkeypatch: object) -> None:
        """get_map_info must return None (not crash) when no map is configured."""
        import homography_precision.services as svc_mod

        monkeypatch.setattr(  # type: ignore[attr-defined]
            svc_mod,
            "resolve_map_for_tenant",
            _raise_runtime("No map configured for tenant: nonexistent"),
        )
        # Clear the cache so get_map_info re-evaluates
        svc_mod._image_info_cache.clear()
        assert get_map_info("nonexistent") is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raise_runtime(msg: str):
    """Return a callable that raises RuntimeError with *msg*."""
    def _raiser(_tenant_id: str):
        raise RuntimeError(msg)
    return _raiser
