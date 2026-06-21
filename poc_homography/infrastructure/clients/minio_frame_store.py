"""S3 / MinIO object store for survey frame images.

The clean-plate capture writes frame *images* to a MinIO bucket (on maglar's
k8s-lab MinIO) and frame *metadata* to Postgres (Neon). This module owns the
image side: a thin S3 client (MinIO speaks the S3 API) that uploads JPEG bytes
under a deterministic object key and, for the gallery, mints presigned GET URLs.

Config comes from the environment so the same code runs on maglor (capture) and
in the API (gallery):

- ``MINIO_ENDPOINT``      e.g. ``http://s3.10-121-15-59.sslip.io:9000``
- ``MINIO_ACCESS_KEY``    MinIO access key
- ``MINIO_SECRET_KEY``    MinIO secret key
- ``MINIO_BUCKET``        bucket name (default ``cleanplate-frames``)
- ``MINIO_REGION``        S3 region label (default ``us-east-1``; MinIO ignores it)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_BUCKET = "cleanplate-frames"
DEFAULT_REGION = "us-east-1"


@dataclass(frozen=True)
class PutResult:
    """Where an uploaded object landed, plus its content hash."""

    bucket: str
    key: str
    sha256: str


class MinioFrameStore:
    """Uploads (and presigns) survey frame images in an S3/MinIO bucket."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str = DEFAULT_BUCKET,
        region: str = DEFAULT_REGION,
        client: Any | None = None,
    ) -> None:
        """Build the store.

        Args:
            endpoint_url: MinIO S3 endpoint (scheme + host + port).
            access_key: MinIO access key.
            secret_key: MinIO secret key.
            bucket: Target bucket name.
            region: S3 region label (MinIO ignores it but boto3 requires one).
            client: Optional pre-built boto3 S3 client (dependency injection for
                tests); when given, the credential/endpoint args are unused.
        """
        self.bucket = bucket
        self._client = client or boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MinioFrameStore:
        """Build a store from ``MINIO_*`` environment variables.

        Raises:
            RuntimeError: If a required variable is missing.
        """
        src = env if env is not None else os.environ
        endpoint = src.get("MINIO_ENDPOINT", "")
        access = src.get("MINIO_ACCESS_KEY", "")
        secret = src.get("MINIO_SECRET_KEY", "")
        missing = [
            name
            for name, value in (
                ("MINIO_ENDPOINT", endpoint),
                ("MINIO_ACCESS_KEY", access),
                ("MINIO_SECRET_KEY", secret),
            )
            if not value
        ]
        if missing:
            msg = f"MinIO config missing: {', '.join(missing)}"
            raise RuntimeError(msg)
        return cls(
            endpoint_url=endpoint,
            access_key=access,
            secret_key=secret,
            bucket=src.get("MINIO_BUCKET", DEFAULT_BUCKET),
            region=src.get("MINIO_REGION", DEFAULT_REGION),
        )

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not already exist (idempotent)."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self.bucket)

    def put_frame(self, data: bytes, key: str, content_type: str = "image/jpeg") -> PutResult:
        """Upload ``data`` under ``key`` and return its location + sha256."""
        sha256 = hashlib.sha256(data).hexdigest()
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={"sha256": sha256},
        )
        return PutResult(bucket=self.bucket, key=key, sha256=sha256)

    def get_frame(self, key: str, *, bucket: str | None = None) -> bytes:
        """Download and return the raw frame-image bytes stored under ``key``.

        ``bucket`` overrides the store's configured bucket so callers can read
        against the bucket recorded on a frame row (the authoritative location
        of that image), rather than assuming the store's env-configured bucket.
        Used by the offline survey-frame loader to materialise frames for lens
        calibration. Propagates whatever the underlying S3 client raises when
        the object is absent (boto3 ``ClientError`` with code ``NoSuchKey``).
        """
        resp = self._client.get_object(Bucket=bucket or self.bucket, Key=key)
        return resp["Body"].read()

    def presign_get(self, key: str, expires_in: int = 3600, *, bucket: str | None = None) -> str:
        """Return a presigned GET URL for ``key`` (used by the gallery).

        ``bucket`` overrides the store's configured bucket so callers can presign
        against the bucket recorded on a frame row (the authoritative location of
        that image), rather than assuming the store's env-configured bucket.
        """
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket or self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )


__all__ = ["DEFAULT_BUCKET", "MinioFrameStore", "PutResult"]
