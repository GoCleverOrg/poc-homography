"""Unit tests for the imgproxy signed-URL builder."""

from __future__ import annotations

import base64
import hashlib
import hmac

from api.utils.imgproxy import ImgproxySigner

# imgproxy keys/salts are hex-encoded; "0011..." decodes to raw bytes.
_KEY_HEX = "0011223344556677"
_SALT_HEX = "8899aabbccddeeff"
_SOURCE = "s3://cleanplate-frames/run-1/clean_plate/pose-1/f1.jpg"


def _decode_segment(b64url: str) -> bytes:
    padding = "=" * (-len(b64url) % 4)
    return base64.urlsafe_b64decode(b64url + padding)


class TestFromEnv:
    """``ImgproxySigner.from_env`` enable/disable behaviour."""

    def test_returns_none_when_any_var_missing(self) -> None:
        assert ImgproxySigner.from_env({"IMGPROXY_BASE_URL": "https://x"}) is None
        assert ImgproxySigner.from_env({}) is None

    def test_builds_signer_when_all_vars_present(self) -> None:
        signer = ImgproxySigner.from_env(
            {
                "IMGPROXY_BASE_URL": "https://imgproxy.example",
                "IMGPROXY_KEY": _KEY_HEX,
                "IMGPROXY_SALT": _SALT_HEX,
            }
        )
        assert isinstance(signer, ImgproxySigner)


class TestThumbnailUrl:
    """``ImgproxySigner.thumbnail_url`` structure + signature correctness."""

    def _signer(self, base_url: str = "https://imgproxy.example") -> ImgproxySigner:
        return ImgproxySigner(base_url=base_url, key_hex=_KEY_HEX, salt_hex=_SALT_HEX)

    def test_url_structure_and_encoded_source_roundtrip(self) -> None:
        url = self._signer().thumbnail_url(_SOURCE, width=320, height=240)

        prefix = "https://imgproxy.example/"
        assert url.startswith(prefix)
        signature, processing, encoded_source = url[len(prefix) :].split("/")
        assert processing == "rs:fit:320:240:0"
        assert _decode_segment(encoded_source) == _SOURCE.encode()
        # Signature is a non-empty url-safe base64 segment.
        assert signature and "=" not in signature

    def test_signature_matches_hmac_over_salt_plus_path(self) -> None:
        url = self._signer().thumbnail_url(_SOURCE)
        signature, processing, encoded_source = url.split("/")[-3:]

        path = f"/{processing}/{encoded_source}".encode()
        expected = (
            base64.urlsafe_b64encode(
                hmac.new(
                    bytes.fromhex(_KEY_HEX), bytes.fromhex(_SALT_HEX) + path, hashlib.sha256
                ).digest()
            )
            .decode()
            .rstrip("=")
        )
        assert signature == expected

    def test_different_key_yields_different_signature(self) -> None:
        url_a = self._signer().thumbnail_url(_SOURCE)
        other = ImgproxySigner(
            base_url="https://imgproxy.example", key_hex="ffff", salt_hex=_SALT_HEX
        )
        url_b = other.thumbnail_url(_SOURCE)
        assert url_a.split("/")[3] != url_b.split("/")[3]

    def test_trailing_slash_in_base_url_is_normalised(self) -> None:
        url = self._signer(base_url="https://imgproxy.example/").thumbnail_url(_SOURCE)
        assert "https://imgproxy.example//" not in url
