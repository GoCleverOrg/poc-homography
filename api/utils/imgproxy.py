"""imgproxy signed-URL builder for clean-plate gallery thumbnails.

imgproxy serves on-the-fly resized thumbnails from the MinIO ``cleanplate-frames``
bucket. Each thumbnail URL is signed with an HMAC-SHA256 keyed on imgproxy's
configured key/salt so imgproxy will only honour URLs this API minted. The
source is referenced as an ``s3://bucket/key`` URL, which imgproxy fetches
directly from MinIO (imgproxy must be deployed with S3 integration pointed at
the same bucket — see the infra deployment issue).

Config comes from the environment (all required to enable thumbnails):

- ``IMGPROXY_BASE_URL``  e.g. ``https://imgproxy.vasco-dev.pedweb.link``
- ``IMGPROXY_KEY``       hex-encoded HMAC key (imgproxy ``IMGPROXY_KEY``)
- ``IMGPROXY_SALT``      hex-encoded HMAC salt (imgproxy ``IMGPROXY_SALT``)

When any variable is unset, :meth:`ImgproxySigner.from_env` returns ``None`` and
the gallery falls back to the presigned full-resolution image for thumbnails.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_THUMB_SIZE = 320


class ImgproxySigner:
    """Builds signed imgproxy URLs for resized thumbnails of MinIO objects."""

    def __init__(self, *, base_url: str, key_hex: str, salt_hex: str) -> None:
        """Build the signer.

        Args:
            base_url: imgproxy public base URL (scheme + host, no trailing path).
            key_hex: Hex-encoded HMAC key (imgproxy ``IMGPROXY_KEY``).
            salt_hex: Hex-encoded HMAC salt (imgproxy ``IMGPROXY_SALT``).

        Raises:
            ValueError: If ``key_hex`` or ``salt_hex`` is not valid hex.
        """
        self._base_url = base_url.rstrip("/")
        self._key = bytes.fromhex(key_hex)
        self._salt = bytes.fromhex(salt_hex)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ImgproxySigner | None:
        """Build a signer from ``IMGPROXY_*`` env vars, or ``None`` if unset.

        Returns ``None`` (thumbnails disabled) when any required variable is
        missing, so the gallery degrades gracefully to full-image previews.
        """
        src = env if env is not None else os.environ
        base = src.get("IMGPROXY_BASE_URL", "")
        key = src.get("IMGPROXY_KEY", "")
        salt = src.get("IMGPROXY_SALT", "")
        if not (base and key and salt):
            return None
        return cls(base_url=base, key_hex=key, salt_hex=salt)

    def thumbnail_url(
        self,
        source_url: str,
        *,
        width: int = DEFAULT_THUMB_SIZE,
        height: int = DEFAULT_THUMB_SIZE,
    ) -> str:
        """Return a signed imgproxy URL that resizes ``source_url`` to fit.

        Args:
            source_url: Source the thumbnail is generated from — an
                ``s3://bucket/key`` URL imgproxy fetches from MinIO.
            width: Target bounding-box width in pixels.
            height: Target bounding-box height in pixels.

        Returns:
            A fully-qualified, signed imgproxy URL.
        """
        encoded_source = base64.urlsafe_b64encode(source_url.encode()).decode().rstrip("=")
        # rs:fit:W:H:0 — resize to fit within WxH without enlarging.
        path = f"/rs:fit:{width}:{height}:0/{encoded_source}"
        digest = hmac.new(self._key, self._salt + path.encode(), hashlib.sha256).digest()
        signature = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        return f"{self._base_url}/{signature}{path}"


__all__ = ["DEFAULT_THUMB_SIZE", "ImgproxySigner"]
