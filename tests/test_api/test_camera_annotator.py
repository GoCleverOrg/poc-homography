"""Integration tests for the camera-annotator save endpoint (issue #235).

The SPA auto-saves the full annotation list after every mutation. Deleting the
last annotation fires a save with an empty list, which must persist (returning
``saved: 0``) rather than be rejected. These tests patch the router's frame
helpers so the endpoint runs without a real database or filesystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

SAVE_URL = "/camera-annotator/api/save-annotations/?tenant_id=t1"


def _mock_frame() -> MagicMock:
    """A stand-in CapturedFrame with the attributes the view reads."""
    frame = MagicMock()
    frame.id = "frame-1"
    frame.ptz_state = MagicMock()
    return frame


class TestSaveAnnotations:
    """Tests for ``POST /camera-annotator/api/save-annotations/``."""

    @patch("api.routers.camera_annotator.save_annotations_for_frame")
    @patch("api.routers.camera_annotator.image_filename_to_frame")
    def test_empty_list_saves_zero(
        self,
        mock_frame: MagicMock,
        mock_save: MagicMock,
        client: TestClient,
    ) -> None:
        """An empty list is a valid save and reports zero saved (DoD #235)."""
        mock_frame.return_value = _mock_frame()

        resp = client.post(
            SAVE_URL,
            json={"image_filename": "cam/frame.jpg", "annotations": []},
        )

        assert resp.status_code == 200
        assert resp.json() == {"success": True, "saved": 0}
        # The empty list is still forwarded to the repo so it overwrites the
        # stored annotations rather than leaving a stale entry behind.
        mock_save.assert_called_once()
        assert mock_save.call_args.args[1] == []

    @patch("api.routers.camera_annotator.save_annotations_for_frame")
    @patch("api.routers.camera_annotator.image_filename_to_frame")
    def test_nonempty_list_rounds_and_saves(
        self,
        mock_frame: MagicMock,
        mock_save: MagicMock,
        client: TestClient,
    ) -> None:
        """A populated list saves, rounding pixel coordinates to one decimal."""
        mock_frame.return_value = _mock_frame()

        resp = client.post(
            SAVE_URL,
            json={
                "image_filename": "cam/frame.jpg",
                "annotations": [
                    {"gcp_id": "G1", "pixel_x": 10.06, "pixel_y": 20.04},
                ],
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"success": True, "saved": 1}

        entities = mock_save.call_args.args[1]
        assert len(entities) == 1
        assert entities[0].gcp_id == "G1"
        assert float(entities[0].pixel.x) == 10.1
        assert float(entities[0].pixel.y) == 20.0

    @patch("api.routers.camera_annotator.image_filename_to_frame")
    def test_unknown_image_returns_400(
        self,
        mock_frame: MagicMock,
        client: TestClient,
    ) -> None:
        """An unresolvable image still fails — the empty-list fix is narrow."""
        mock_frame.return_value = None

        resp = client.post(
            SAVE_URL,
            json={"image_filename": "missing.jpg", "annotations": []},
        )

        assert resp.status_code == 400
