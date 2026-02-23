"""Tests for Sam3ApiClient — API key in header, not URL (Security #9)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from poc_homography.domain.protocols import Sam3ApiError
from poc_homography.infrastructure.clients.sam3_api_client import (
    SAM3_API_URL,
    Sam3ApiClient,
)

API_KEY = "test-key-123"


@pytest.fixture
def client() -> Sam3ApiClient:
    return Sam3ApiClient(api_key=API_KEY)


def _mock_success_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"prompt_results": []}
    return resp


class TestSam3ApiKeyInHeader:
    @patch("poc_homography.infrastructure.clients.sam3_api_client.requests.post")
    def test_api_key_in_x_api_key_header(self, mock_post: MagicMock, client: Sam3ApiClient) -> None:
        mock_post.return_value = _mock_success_response()
        client.segment("base64img", "road")

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["x-api-key"] == API_KEY

    @patch("poc_homography.infrastructure.clients.sam3_api_client.requests.post")
    def test_api_key_not_in_url(self, mock_post: MagicMock, client: Sam3ApiClient) -> None:
        mock_post.return_value = _mock_success_response()
        client.segment("base64img", "road")

        url = mock_post.call_args[0][0]
        assert API_KEY not in url
        assert url == SAM3_API_URL

    @patch("poc_homography.infrastructure.clients.sam3_api_client.requests.post")
    def test_api_key_not_in_body(self, mock_post: MagicMock, client: Sam3ApiClient) -> None:
        mock_post.return_value = _mock_success_response()
        client.segment("base64img", "road")

        body = mock_post.call_args[1]["json"]
        assert API_KEY not in str(body)

    @patch("poc_homography.infrastructure.clients.sam3_api_client.requests.post")
    def test_content_type_json(self, mock_post: MagicMock, client: Sam3ApiClient) -> None:
        mock_post.return_value = _mock_success_response()
        client.segment("base64img", "road")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Content-Type"] == "application/json"


class TestSam3ErrorHandling:
    @patch("poc_homography.infrastructure.clients.sam3_api_client.requests.post")
    def test_non_200_raises(self, mock_post: MagicMock, client: Sam3ApiClient) -> None:
        resp = MagicMock()
        resp.status_code = 500
        mock_post.return_value = resp

        with pytest.raises(Sam3ApiError, match="500"):
            client.segment("base64img", "road")

    @patch("poc_homography.infrastructure.clients.sam3_api_client.requests.post")
    def test_request_exception_raises(self, mock_post: MagicMock, client: Sam3ApiClient) -> None:
        mock_post.side_effect = requests.RequestException("connection timeout")

        with pytest.raises(Sam3ApiError, match="request failed"):
            client.segment("base64img", "road")
