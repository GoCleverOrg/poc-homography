"""Unit tests for the map-asset upload pipeline (fake store, tmp maps dir)."""

from __future__ import annotations

import hashlib
import textwrap
from typing import TYPE_CHECKING

from poc_homography.infrastructure.clients.minio_frame_store import PutResult
from poc_homography.maps import upload_map_assets

if TYPE_CHECKING:
    from pathlib import Path


class FakeMapStore:
    """Records ``put_map`` and ``ensure_bucket`` calls."""

    def __init__(self) -> None:
        self.bucket = "map-assets"
        self.puts: list[tuple[bytes, str]] = []
        self.ensure_calls = 0

    def ensure_bucket(self) -> None:
        self.ensure_calls += 1

    def put_map(self, data: bytes, key: str, content_type: str = "image/tiff") -> PutResult:
        self.puts.append((data, key))
        return PutResult(bucket=self.bucket, key=key, sha256=hashlib.sha256(data).hexdigest())


def _write_sidecar(maps_dir: Path, name: str, map_id: str, tenant: str, path: str) -> None:
    (maps_dir / name).write_text(
        textwrap.dedent(
            f"""\
            id: {map_id}
            tenant_id: {tenant}
            photo:
              path: {path}
            """
        )
    )


def _make_maps_dir(tmp_path: Path) -> Path:
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    _write_sidecar(maps_dir, "icozee.yaml", "icozee_cropped", "icozee", "icozee-cropped.tif")
    _write_sidecar(
        maps_dir, "valte.yaml", "Cartografia_valencia", "valte", "Cartografia_valencia.tif"
    )
    return maps_dir


def test_uploads_with_tenant_scoped_keys(tmp_path: Path) -> None:
    maps_dir = _make_maps_dir(tmp_path)
    (maps_dir / "icozee-cropped.tif").write_bytes(b"icozee tif")
    (maps_dir / "Cartografia_valencia.tif").write_bytes(b"valte tif")
    store = FakeMapStore()

    outcomes = upload_map_assets(maps_dir, store)

    keys = {o.key for o in outcomes}
    assert keys == {"icozee/icozee-cropped.tif", "valte/Cartografia_valencia.tif"}
    assert all(o.status == "uploaded" for o in outcomes)
    put_keys = {key for _, key in store.puts}
    assert put_keys == keys
    assert (b"icozee tif", "icozee/icozee-cropped.tif") in store.puts
    assert store.ensure_calls == 1


def test_missing_tif_is_skipped(tmp_path: Path) -> None:
    maps_dir = _make_maps_dir(tmp_path)
    # Only one tif present on disk; the other is absent (not materialized locally).
    (maps_dir / "icozee-cropped.tif").write_bytes(b"icozee tif")
    store = FakeMapStore()

    outcomes = upload_map_assets(maps_dir, store)

    by_key = {o.key: o for o in outcomes}
    assert by_key["icozee/icozee-cropped.tif"].status == "uploaded"
    assert by_key["valte/Cartografia_valencia.tif"].status == "missing"
    assert by_key["valte/Cartografia_valencia.tif"].result is None
    # No put for the missing one.
    assert [key for _, key in store.puts] == ["icozee/icozee-cropped.tif"]


def test_idempotent_rerun(tmp_path: Path) -> None:
    maps_dir = _make_maps_dir(tmp_path)
    (maps_dir / "icozee-cropped.tif").write_bytes(b"icozee tif")
    (maps_dir / "Cartografia_valencia.tif").write_bytes(b"valte tif")
    store = FakeMapStore()

    first = upload_map_assets(maps_dir, store)
    second = upload_map_assets(maps_dir, store)

    assert [o.key for o in first] == [o.key for o in second]
    # ensure_bucket called once per run.
    assert store.ensure_calls == 2


def test_ensure_bucket_called_once_per_run(tmp_path: Path) -> None:
    maps_dir = _make_maps_dir(tmp_path)
    store = FakeMapStore()

    upload_map_assets(maps_dir, store)

    assert store.ensure_calls == 1


def test_ensure_bucket_disabled(tmp_path: Path) -> None:
    maps_dir = _make_maps_dir(tmp_path)
    store = FakeMapStore()

    upload_map_assets(maps_dir, store, ensure_bucket=False)

    assert store.ensure_calls == 0
