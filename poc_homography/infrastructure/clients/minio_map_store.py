"""S3 / MinIO object store for map-asset GeoTIFFs.

The map-asset pipeline uploads ``data/maps/*.tif`` GeoTIFFs to a MinIO bucket
(on maglar's k8s-lab MinIO) under tenant/map-scoped object keys. This module
owns that upload side: a thin S3 client (MinIO speaks the S3 API) that uploads
GeoTIFF bytes under a deterministic object key and, for downstream consumers,
mints presigned GET URLs.

Config comes from the environment so the same code runs wherever the pipeline
is invoked. The MinIO endpoint/credentials are shared with
:mod:`poc_homography.infrastructure.clients.minio_frame_store`; only the bucket
differs:

- ``MINIO_ENDPOINT``      e.g. ``http://s3.10-121-15-59.sslip.io:9000``
- ``MINIO_ACCESS_KEY``    MinIO access key
- ``MINIO_SECRET_KEY``    MinIO secret key
- ``MINIO_MAP_BUCKET``    bucket name (default ``map-assets``)
- ``MINIO_REGION``        S3 region label (default ``us-east-1``; MinIO ignores it)
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from poc_homography.infrastructure.clients.minio_frame_store import PutResult

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_BUCKET = "map-assets"
DEFAULT_REGION = "us-east-1"


class MinioMapStore:
    """Uploads (and presigns) map-asset GeoTIFFs in an S3/MinIO bucket."""

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
    def from_env(cls, env: Mapping[str, str] | None = None) -> MinioMapStore:
        """Build a store from ``MINIO_*`` environment variables.

        Reuses the shared ``MINIO_ENDPOINT``/``MINIO_ACCESS_KEY``/
        ``MINIO_SECRET_KEY`` credentials and reads the bucket from
        ``MINIO_MAP_BUCKET`` (default ``map-assets``).

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
            bucket=src.get("MINIO_MAP_BUCKET", DEFAULT_BUCKET),
            region=src.get("MINIO_REGION", DEFAULT_REGION),
        )

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not already exist (idempotent)."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self.bucket)

    def put_map(self, data: bytes, key: str, content_type: str = "image/tiff") -> PutResult:
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

    def presign_get(self, key: str, expires_in: int = 3600) -> str:
        """Return a presigned GET URL for ``key`` (used by downstream consumers)."""
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def get_map(self, key: str) -> bytes:
        """Download and return the raw GeoTIFF bytes stored under ``key``.

        Used by the API to materialise a map asset locally (e.g. under ``/tmp``)
        before tiling. Propagates whatever the underlying S3 client raises when
        the object is absent (boto3 ``ClientError`` with code ``NoSuchKey``).
        """
        resp = self._client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()


__all__ = ["DEFAULT_BUCKET", "MinioMapStore", "PutResult"]
