"""Integration tests for the lens-calibration save endpoint (issue #214).

The SPA's multi-zoom batch calibration flow posts an array of per-zoom entries
that must persist as a single multi-entry ``CameraCalibrationTable``, while the
legacy single-entry body stays fully backward-compatible. These tests patch the
router's ``sync_to_ddd_repo_pg`` collaborator to capture the table without a
real database.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

SAVE_URL = "/lens-calibration/api/save/"


class TestSaveCalibration:
    """Tests for ``POST /lens-calibration/api/save/``."""

    @patch("api.routers.lens_calibration.sync_to_ddd_repo_pg")
    def test_multi_entry_save(
        self,
        mock_sync: MagicMock,
        client: TestClient,
    ) -> None:
        """A multi-zoom batch persists one table with one entry per zoom."""
        resp = client.post(
            SAVE_URL,
            json={
                "camera_id": "cam-multi",
                "zoom_entries": [
                    {
                        "zoom": 1.0,
                        "coefficients": {"k1": -0.1, "k2": 0.01, "k3": 0.0, "p1": 0.0, "p2": 0.0},
                        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0},
                        "validation_rmse": 1.5,
                        "num_lines": 12,
                    },
                    {
                        "zoom": 2.0,
                        "coefficients": {"k1": -0.2, "k2": 0.02, "k3": 0.0, "p1": 0.0, "p2": 0.0},
                        "intrinsics": {"fx": 2000.0, "fy": 2000.0, "cx": 961.0, "cy": 541.0},
                        "validation_rmse": 2.5,
                        "num_lines": 8,
                    },
                    {
                        "zoom": 3.0,
                        "coefficients": {"k1": -0.3, "k2": 0.03, "k3": 0.0, "p1": 0.0, "p2": 0.0},
                        "validation_rmse": 0.0,
                        "num_lines": 0,
                    },
                ],
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"success": True, "camera_id": "cam-multi"}

        mock_sync.assert_called_once()
        table = mock_sync.call_args.args[0]
        assert table.camera_id == "cam-multi"
        assert len(table.entries) == 3
        assert sorted(table.entries.keys()) == [1.0, 2.0, 3.0]

        e1 = table.entries[1.0]
        assert e1.k1 == -0.1
        assert e1.k2 == 0.01
        assert e1.fx == 1000.0
        assert e1.cy == 540.0
        assert e1.validation_rmse == 1.5
        assert e1.num_lines_used == 12

        e2 = table.entries[2.0]
        assert e2.k1 == -0.2
        assert e2.fx == 2000.0

        # No intrinsics provided -> zeroed focal/principal point.
        e3 = table.entries[3.0]
        assert e3.k1 == -0.3
        assert e3.fx == 0.0
        assert e3.cx == 0.0

    @patch("api.routers.lens_calibration.sync_to_ddd_repo_pg")
    def test_single_entry_back_compat(
        self,
        mock_sync: MagicMock,
        client: TestClient,
    ) -> None:
        """The legacy single-entry body persists exactly one entry, unchanged."""
        resp = client.post(
            SAVE_URL,
            json={
                "camera_id": "cam-legacy",
                "zoom": 1.5,
                "coefficients": {"k1": -0.05, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0},
                "intrinsics": {"fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0},
                "validation_rmse": 0.9,
                "num_lines": 5,
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"success": True, "camera_id": "cam-legacy"}

        mock_sync.assert_called_once()
        table = mock_sync.call_args.args[0]
        assert len(table.entries) == 1
        entry = table.entries[1.5]
        assert entry.k1 == -0.05
        assert entry.fx == 1500.0
        assert entry.num_lines_used == 5

    @patch("api.routers.lens_calibration.sync_to_ddd_repo_pg")
    def test_duplicate_zoom_last_wins(
        self,
        mock_sync: MagicMock,
        client: TestClient,
    ) -> None:
        """Two entries at the same zoom collapse to one — the last one wins."""
        resp = client.post(
            SAVE_URL,
            json={
                "camera_id": "cam-dup",
                "zoom_entries": [
                    {
                        "zoom": 2.0,
                        "coefficients": {"k1": -0.1, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0},
                        "validation_rmse": 1.0,
                        "num_lines": 3,
                    },
                    {
                        "zoom": 2.0,
                        "coefficients": {"k1": -0.9, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0},
                        "validation_rmse": 4.0,
                        "num_lines": 7,
                    },
                ],
            },
        )

        assert resp.status_code == 200
        mock_sync.assert_called_once()
        table = mock_sync.call_args.args[0]
        assert len(table.entries) == 1
        entry = table.entries[2.0]
        assert entry.k1 == -0.9
        assert entry.validation_rmse == 4.0
        assert entry.num_lines_used == 7
