"""Tests for the camera annotator save-annotations view (issue #235).

The annotator auto-saves the full annotation list after every mutation. Deleting
the last annotation fires a save with an empty list, which must persist (overwrite
the YAML with an empty annotations section) rather than be rejected. These tests
monkeypatch the view's repository-writing collaborators so the endpoint is
exercised without any real frames or filesystem writes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Add webapp to path for Django imports
PROJECT_ROOT = Path(__file__).parent.parent
WEBAPP_DIR = PROJECT_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "homography_web.settings")

import django

django.setup()

from django.test import Client

# URL prefix under which camera_annotator.urls is mounted (homography_web/urls.py).
PREFIX = "/camera-annotator"
SAVE_URL = f"{PREFIX}/api/save-annotations/"


@pytest.fixture
def patched_views(monkeypatch):
    """Stub the view's repo collaborators and record what gets saved.

    Django resolves the URLconf via the ``camera_annotator.views`` module object
    (webapp/ is on sys.path), so we patch that exact module.
    """
    import camera_annotator.views as views

    saved: dict = {}

    def fake_save(image_filename, annotations):
        saved["image_filename"] = image_filename
        saved["annotations"] = annotations

    monkeypatch.setattr(views, "_get_tenant_map_id", lambda request: "map-1")
    monkeypatch.setattr(views, "get_current_image", lambda request, map_id=None: "cam_a/frame.jpg")
    monkeypatch.setattr(views, "save_annotations_to_repo", fake_save)
    monkeypatch.setattr(views, "invalidate_cache", lambda: None)
    return saved


@pytest.fixture
def client():
    return Client()


def test_save_empty_annotations_succeeds(client, patched_views):
    """Saving an empty list persists and reports zero saved (DoD, issue #235)."""
    resp = client.post(
        SAVE_URL,
        data=json.dumps({"annotations": []}),
        content_type="application/json",
    )

    assert resp.status_code == 200
    assert resp.json() == {"success": True, "saved": 0}
    # The empty list is forwarded to the repo so the YAML is overwritten.
    assert patched_views["annotations"] == []


def test_save_nonempty_annotations_succeeds(client, patched_views):
    """A populated list still saves and rounds pixel values to one decimal."""
    resp = client.post(
        SAVE_URL,
        data=json.dumps({"annotations": [{"gcp_id": "G1", "pixel_x": 10.04, "pixel_y": 20.06}]}),
        content_type="application/json",
    )

    assert resp.status_code == 200
    assert resp.json() == {"success": True, "saved": 1}
    assert patched_views["annotations"] == [{"gcp_id": "G1", "pixel_x": 10.0, "pixel_y": 20.1}]
