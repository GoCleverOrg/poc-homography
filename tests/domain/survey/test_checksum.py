"""Tests for the SHA-256 checksum utility."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from poc_homography.domain.utils.checksum import sha256_bytes, sha256_file

if TYPE_CHECKING:
    from pathlib import Path


class TestSha256Bytes:
    def test_known_digest(self) -> None:
        assert sha256_bytes(b"") == hashlib.sha256(b"").hexdigest()

    def test_deterministic(self) -> None:
        payload = b"survey-frame-bytes"
        assert sha256_bytes(payload) == sha256_bytes(payload)

    def test_distinct_inputs_differ(self) -> None:
        assert sha256_bytes(b"a") != sha256_bytes(b"b")


class TestSha256File:
    def test_matches_in_memory_digest(self, tmp_path: Path) -> None:
        payload = b"\x00\x01\x02image-bytes" * 10000
        image = tmp_path / "frame.jpg"
        image.write_bytes(payload)
        assert sha256_file(image) == hashlib.sha256(payload).hexdigest()

    def test_deterministic(self, tmp_path: Path) -> None:
        image = tmp_path / "frame.jpg"
        image.write_bytes(b"deterministic")
        assert sha256_file(image) == sha256_file(image)

    def test_is_hex_digest(self, tmp_path: Path) -> None:
        image = tmp_path / "frame.jpg"
        image.write_bytes(b"abc")
        digest = sha256_file(image)
        assert len(digest) == 64
        int(digest, 16)  # raises ValueError if not hex
