"""Unit tests for :class:`MinioMapStore` (mocked S3 client)."""

from __future__ import annotations

import hashlib

import pytest
from botocore.exceptions import ClientError

from poc_homography.infrastructure.clients.minio_map_store import (
    DEFAULT_BUCKET,
    MinioMapStore,
)


class FakeS3:
    """Records boto3 S3 client calls; ``head_missing`` makes head_bucket 404."""

    def __init__(self, *, head_missing: bool = False) -> None:
        self.head_missing = head_missing
        self.puts: list[dict] = []
        self.created: list[str] = []
        self.heads: list[str] = []

    def head_bucket(self, Bucket: str) -> None:
        self.heads.append(Bucket)
        if self.head_missing:
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    def create_bucket(self, Bucket: str) -> None:
        self.created.append(Bucket)

    def put_object(self, **kwargs: object) -> None:
        self.puts.append(kwargs)

    def generate_presigned_url(self, op: str, Params: dict, ExpiresIn: int) -> str:
        return f"https://minio/{op}/{Params['Bucket']}/{Params['Key']}?e={ExpiresIn}"


def _store(client: FakeS3, bucket: str = "map-assets") -> MinioMapStore:
    return MinioMapStore(
        endpoint_url="http://minio:9000",
        access_key="k",
        secret_key="s",
        bucket=bucket,
        client=client,
    )


def test_put_map_uploads_and_returns_sha256() -> None:
    client = FakeS3()
    store = _store(client)
    data = b"II*\x00 tiff bytes"

    result = store.put_map(data, "icozee/icozee-cropped.tif")

    assert result.bucket == "map-assets"
    assert result.key == "icozee/icozee-cropped.tif"
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    assert len(client.puts) == 1
    put = client.puts[0]
    assert put["Bucket"] == "map-assets"
    assert put["Key"] == "icozee/icozee-cropped.tif"
    assert put["Body"] == data
    assert put["ContentType"] == "image/tiff"
    assert put["Metadata"]["sha256"] == result.sha256


def test_ensure_bucket_creates_when_missing() -> None:
    client = FakeS3(head_missing=True)
    _store(client).ensure_bucket()
    assert client.created == ["map-assets"]


def test_ensure_bucket_noop_when_present() -> None:
    client = FakeS3(head_missing=False)
    _store(client).ensure_bucket()
    assert client.created == []


def test_presign_get_delegates_to_client() -> None:
    client = FakeS3()
    url = _store(client).presign_get("valte/Cartografia_valencia.tif", expires_in=120)
    assert url == "https://minio/get_object/map-assets/valte/Cartografia_valencia.tif?e=120"


def test_from_env_reads_minio_map_bucket() -> None:
    env = {
        "MINIO_ENDPOINT": "http://minio:9000",
        "MINIO_ACCESS_KEY": "k",
        "MINIO_SECRET_KEY": "s",
    }
    store = MinioMapStore.from_env(env)
    assert store.bucket == DEFAULT_BUCKET

    custom = MinioMapStore.from_env({**env, "MINIO_MAP_BUCKET": "other-maps"})
    assert custom.bucket == "other-maps"


def test_from_env_missing_raises() -> None:
    with pytest.raises(RuntimeError, match="MINIO_ACCESS_KEY"):
        MinioMapStore.from_env({"MINIO_ENDPOINT": "http://minio:9000"})
