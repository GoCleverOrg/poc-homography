"""Unit tests for :class:`MinioFrameStore` (mocked S3 client)."""

from __future__ import annotations

import hashlib

import pytest
from botocore.exceptions import ClientError

from poc_homography.infrastructure.clients.minio_frame_store import (
    DEFAULT_BUCKET,
    MinioFrameStore,
)


class FakeS3:
    """Records boto3 S3 client calls; ``head_missing`` makes head_bucket 404."""

    def __init__(self, *, head_missing: bool = False) -> None:
        self.head_missing = head_missing
        self.puts: list[dict] = []
        self.created: list[str] = []
        self.heads: list[str] = []
        self.objects: dict[tuple[str, str], bytes] = {}

    def head_bucket(self, Bucket: str) -> None:
        self.heads.append(Bucket)
        if self.head_missing:
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    def create_bucket(self, Bucket: str) -> None:
        self.created.append(Bucket)

    def put_object(self, **kwargs: object) -> None:
        self.puts.append(kwargs)
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]

    def get_object(self, Bucket: str, Key: str) -> dict:
        try:
            data = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject") from exc
        return {"Body": _Body(data)}

    def generate_presigned_url(self, op: str, Params: dict, ExpiresIn: int) -> str:
        return f"https://minio/{op}/{Params['Bucket']}/{Params['Key']}?e={ExpiresIn}"


class _Body:
    """Minimal stand-in for a boto3 streaming body (``.read()``)."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


def _store(client: FakeS3, bucket: str = "cleanplate-frames") -> MinioFrameStore:
    return MinioFrameStore(
        endpoint_url="http://minio:9000",
        access_key="k",
        secret_key="s",
        bucket=bucket,
        client=client,
    )


def test_put_frame_uploads_and_returns_sha256() -> None:
    client = FakeS3()
    store = _store(client)
    data = b"\xff\xd8\xff jpeg bytes"

    result = store.put_frame(data, "run1/main/p1/cap1.jpg")

    assert result.bucket == "cleanplate-frames"
    assert result.key == "run1/main/p1/cap1.jpg"
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    assert len(client.puts) == 1
    put = client.puts[0]
    assert put["Bucket"] == "cleanplate-frames"
    assert put["Key"] == "run1/main/p1/cap1.jpg"
    assert put["Body"] == data
    assert put["ContentType"] == "image/jpeg"
    assert put["Metadata"]["sha256"] == result.sha256


def test_ensure_bucket_creates_when_missing() -> None:
    client = FakeS3(head_missing=True)
    _store(client).ensure_bucket()
    assert client.created == ["cleanplate-frames"]


def test_ensure_bucket_noop_when_present() -> None:
    client = FakeS3(head_missing=False)
    _store(client).ensure_bucket()
    assert client.created == []


def test_get_frame_round_trips_put_bytes() -> None:
    client = FakeS3()
    store = _store(client)
    data = b"\xff\xd8\xff jpeg bytes"
    store.put_frame(data, "run1/main/p1/cap1.jpg")

    assert store.get_frame("run1/main/p1/cap1.jpg") == data


def test_get_frame_bucket_override_reads_from_named_bucket() -> None:
    client = FakeS3()
    # Store configured for one bucket; the frame lives in another.
    store = _store(client, bucket="store-default")
    data = b"\x89PNG other-bucket bytes"
    client.objects[("frame-bucket", "run9/cap9.jpg")] = data

    assert store.get_frame("run9/cap9.jpg", bucket="frame-bucket") == data


def test_get_frame_missing_raises_client_error() -> None:
    store = _store(FakeS3())
    with pytest.raises(ClientError):
        store.get_frame("nope.jpg")


def test_presign_get_delegates_to_client() -> None:
    client = FakeS3()
    url = _store(client).presign_get("run1/x.jpg", expires_in=120)
    assert url == "https://minio/get_object/cleanplate-frames/run1/x.jpg?e=120"


def test_from_env_reads_minio_vars() -> None:
    env = {
        "MINIO_ENDPOINT": "http://minio:9000",
        "MINIO_ACCESS_KEY": "k",
        "MINIO_SECRET_KEY": "s",
    }
    store = MinioFrameStore.from_env(env)
    assert store.bucket == DEFAULT_BUCKET


def test_from_env_missing_raises() -> None:
    with pytest.raises(RuntimeError, match="MINIO_ACCESS_KEY"):
        MinioFrameStore.from_env({"MINIO_ENDPOINT": "http://minio:9000"})
