"""SHA-256 checksum helpers for survey image/video assets.

Pure helpers: ``sha256_file`` performs a single read-only file read;
``sha256_bytes`` is fully in-memory. Both return a lowercase hex digest.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_CHUNK_SIZE = 65536  # 64 KiB streaming read


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``.

    Args:
        data: Raw bytes to hash.

    Returns:
        Deterministic lowercase hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at ``path``.

    The file is read in fixed-size chunks so arbitrarily large images and
    video segments hash with bounded memory. The read is the only side
    effect.

    Args:
        path: Path to the file to hash.

    Returns:
        Deterministic lowercase hex digest of the file bytes.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
