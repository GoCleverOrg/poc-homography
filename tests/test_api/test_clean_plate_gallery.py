"""Integration tests for the clean-plate gallery router (frames + runs).

The DB repository, the MinIO store, and the imgproxy signer are all mocked, so
these tests run fully offline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from tests.test_api.conftest import make_basic_auth_header

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Test data factory
# ---------------------------------------------------------------------------


def _make_row(frame_id: str = "f1", run_id: str = "run-1") -> SimpleNamespace:
    """A stand-in for a ``CleanPlateFrameModel`` row (attribute access only)."""
    return SimpleNamespace(
        id=frame_id,
        run_id=run_id,
        camera_id="cam-1",
        phase="clean_plate",
        pose_id="pose-1",
        commanded_pan=10.0,
        commanded_tilt=20.0,
        commanded_zoom=1.5,
        burst_id=None,
        frame_index=0,
        captured_at=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
        minio_bucket="cleanplate-frames",
        minio_object_key=f"run-1/clean_plate/pose-1/{frame_id}.jpg",
        checksum_sha256="abc123",
        record={"optics": {"fov_deg": 60.0}, "pose_id": "pose-1"},
    )


# ---------------------------------------------------------------------------
# GET /clean-plate/frames
# ---------------------------------------------------------------------------


class TestGetFrames:
    """Tests for ``GET /clean-plate/frames``."""

    @patch("api.routers.clean_plate_gallery.ImgproxySigner")
    @patch("api.routers.clean_plate_gallery.MinioFrameStore")
    @patch("api.routers.clean_plate_gallery.RepoPostgresCleanPlateFrame")
    def test_returns_frames_with_image_and_thumbnail_urls(
        self,
        mock_repo_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_signer_cls: MagicMock,
        client: TestClient,
    ) -> None:
        mock_repo_cls.return_value.query_frames.return_value = ([_make_row()], 1)
        mock_store_cls.from_env.return_value.presign_get.return_value = (
            "https://minio/presigned/f1.jpg"
        )
        mock_signer_cls.from_env.return_value.thumbnail_url.return_value = (
            "https://imgproxy/sig/rs:fit:320:320:0/abc"
        )

        resp = client.get("/clean-plate/frames")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["limit"] == 100
        assert body["offset"] == 0
        assert len(body["frames"]) == 1
        frame = body["frames"][0]
        assert frame["id"] == "f1"
        assert frame["image_url"] == "https://minio/presigned/f1.jpg"
        assert frame["thumbnail_url"] == "https://imgproxy/sig/rs:fit:320:320:0/abc"
        assert frame["record"]["optics"]["fov_deg"] == 60.0
        # imgproxy source is the s3:// URL of the frame object.
        mock_signer_cls.from_env.return_value.thumbnail_url.assert_called_once_with(
            "s3://cleanplate-frames/run-1/clean_plate/pose-1/f1.jpg"
        )

    @patch("api.routers.clean_plate_gallery.ImgproxySigner")
    @patch("api.routers.clean_plate_gallery.MinioFrameStore")
    @patch("api.routers.clean_plate_gallery.RepoPostgresCleanPlateFrame")
    def test_thumbnail_falls_back_to_image_when_imgproxy_unset(
        self,
        mock_repo_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_signer_cls: MagicMock,
        client: TestClient,
    ) -> None:
        mock_repo_cls.return_value.query_frames.return_value = ([_make_row()], 1)
        mock_store_cls.from_env.return_value.presign_get.return_value = "https://minio/full.jpg"
        mock_signer_cls.from_env.return_value = None  # imgproxy not configured

        resp = client.get("/clean-plate/frames")

        assert resp.status_code == 200
        frame = resp.json()["frames"][0]
        assert frame["thumbnail_url"] == "https://minio/full.jpg"
        assert frame["image_url"] == "https://minio/full.jpg"

    @patch("api.routers.clean_plate_gallery.MinioFrameStore")
    @patch("api.routers.clean_plate_gallery.RepoPostgresCleanPlateFrame")
    def test_empty_result_does_not_touch_minio(
        self,
        mock_repo_cls: MagicMock,
        mock_store_cls: MagicMock,
        client: TestClient,
    ) -> None:
        mock_repo_cls.return_value.query_frames.return_value = ([], 0)

        resp = client.get("/clean-plate/frames")

        assert resp.status_code == 200
        assert resp.json() == {"frames": [], "total": 0, "limit": 100, "offset": 0}
        mock_store_cls.from_env.assert_not_called()

    @patch("api.routers.clean_plate_gallery.MinioFrameStore")
    @patch("api.routers.clean_plate_gallery.RepoPostgresCleanPlateFrame")
    def test_filters_and_pagination_passed_to_repo(
        self,
        mock_repo_cls: MagicMock,
        mock_store_cls: MagicMock,
        client: TestClient,
    ) -> None:
        mock_repo_cls.return_value.query_frames.return_value = ([], 0)

        resp = client.get(
            "/clean-plate/frames",
            params={
                "run_id": "run-7",
                "pose_id": "pose-9",
                "camera_id": "cam-2",
                "phase": "clean_plate",
                "limit": 25,
                "offset": 50,
            },
        )

        assert resp.status_code == 200
        kwargs = mock_repo_cls.return_value.query_frames.call_args.kwargs
        assert kwargs["run_id"] == "run-7"
        assert kwargs["pose_id"] == "pose-9"
        assert kwargs["camera_id"] == "cam-2"
        assert kwargs["phase"] == "clean_plate"
        assert kwargs["limit"] == 25
        assert kwargs["offset"] == 50

    def test_limit_out_of_range_is_rejected(self, client: TestClient) -> None:
        resp = client.get("/clean-plate/frames", params={"limit": 9999})
        assert resp.status_code == 422

    def test_requires_authentication(self, client_no_auth_override: TestClient) -> None:
        resp = client_no_auth_override.get(
            "/clean-plate/frames",
            headers=make_basic_auth_header("nobody", "wrong"),
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /clean-plate/runs
# ---------------------------------------------------------------------------


class TestGetRuns:
    """Tests for ``GET /clean-plate/runs``."""

    @patch("api.routers.clean_plate_gallery.RepoPostgresCleanPlateFrame")
    def test_returns_runs(self, mock_repo_cls: MagicMock, client: TestClient) -> None:
        mock_repo_cls.return_value.list_runs.return_value = [
            {
                "run_id": "run-2",
                "frame_count": 3,
                "first_captured_at": datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc),
                "last_captured_at": datetime(2026, 6, 2, 10, 0, tzinfo=timezone.utc),
            },
            {
                "run_id": "run-1",
                "frame_count": 5,
                "first_captured_at": datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc),
                "last_captured_at": datetime(2026, 6, 1, 11, 0, tzinfo=timezone.utc),
            },
        ]

        resp = client.get("/clean-plate/runs")

        assert resp.status_code == 200
        runs = resp.json()["runs"]
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run-2"
        assert runs[0]["frame_count"] == 3
        assert runs[1]["run_id"] == "run-1"

    @patch("api.routers.clean_plate_gallery.RepoPostgresCleanPlateFrame")
    def test_returns_empty_list(self, mock_repo_cls: MagicMock, client: TestClient) -> None:
        mock_repo_cls.return_value.list_runs.return_value = []

        resp = client.get("/clean-plate/runs")

        assert resp.status_code == 200
        assert resp.json() == {"runs": []}
