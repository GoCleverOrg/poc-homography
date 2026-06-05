"""HTTP transport for the Hikvision ISAPI adapter.

Wraps a persistent ``requests.Session`` with HTTP digest authentication and a
fixed timeout. Every request goes through this module, which maps low-level
failures onto the adapter exception hierarchy:

- ``requests.RequestException`` -> ``HikvisionTransportError``
- XML parse failure -> ``HikvisionParseError``
- non-2xx response -> ``HikvisionHTTPError`` / ``HikvisionUnsupportedError``

XML is parsed with ``defusedxml`` to avoid entity-expansion attacks. Element
typing uses ``xml.etree.ElementTree.Element`` while parsing stays on defusedxml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import requests
from defusedxml import ElementTree as ET
from requests.auth import HTTPDigestAuth

from poc_homography.infrastructure.clients.hikvision.isapi_endpoints import NS
from poc_homography.infrastructure.clients.hikvision.isapi_errors import (
    HikvisionError,
    HikvisionHTTPError,
    HikvisionParseError,
    HikvisionTransportError,
    HikvisionUnsupportedError,
)

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element


def find(elem: Element, path: str) -> Element | None:
    """Namespace-aware ``find`` using the Hikvision namespace map.

    Args:
        elem: Element to search from.
        path: Namespaced path (e.g. ``".//h:azimuth"``).

    Returns:
        The first matching element, or ``None`` if not found.
    """
    return elem.find(path, NS)


def findtext(elem: Element, path: str) -> str | None:
    """Namespace-aware ``findtext`` using the Hikvision namespace map.

    Args:
        elem: Element to search from.
        path: Namespaced path (e.g. ``".//h:statusString"``).

    Returns:
        The matched element's text, or ``None`` if not found.
    """
    return elem.findtext(path, default=None, namespaces=NS)


def _parse_xml(text: str) -> Element:
    """Parse ``text`` into an Element, raising ``HikvisionParseError`` on failure."""
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        raise HikvisionParseError(f"XML parse error: {exc}") from exc


def classify_error_response(status_code: int, body_text: str) -> HikvisionError:
    """Classify a non-2xx response body into a Hikvision error.

    This is a pure function (no socket) so error mapping is unit-testable. The
    body is best-effort parsed for ``statusString`` / ``subStatusCode``; a
    malformed body is tolerated and yields a plain ``HikvisionHTTPError``.

    Args:
        status_code: HTTP status code of the response.
        body_text: Raw response body text.

    Returns:
        ``HikvisionUnsupportedError`` for a 403 whose ``subStatusCode`` is
        ``"methodNotAllowed"``; otherwise ``HikvisionHTTPError``.
    """
    status_string: str | None = None
    sub_status_code: str | None = None
    try:
        root = ET.fromstring(body_text)
    except ET.ParseError:
        root = None
    if root is not None:
        status_string = findtext(root, ".//h:statusString") or findtext(root, "h:statusString")
        sub_status_code = findtext(root, ".//h:subStatusCode") or findtext(root, "h:subStatusCode")

    if status_code == 403 and sub_status_code == "methodNotAllowed":
        return HikvisionUnsupportedError(
            status_code=status_code,
            status_string=status_string,
            sub_status_code=sub_status_code,
        )
    return HikvisionHTTPError(
        status_code=status_code,
        status_string=status_string,
        sub_status_code=sub_status_code,
    )


class IsapiTransport:
    """Persistent HTTP transport for ISAPI requests against one camera.

    Attributes:
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        *,
        timeout: float = 5.0,
        scheme: str = "http",
    ) -> None:
        """Initialize the transport.

        Args:
            host: Camera host or IP address (no scheme).
            username: ISAPI account username.
            password: ISAPI account password.
            timeout: Per-request timeout in seconds.
            scheme: URL scheme, ``"http"`` or ``"https"``.
        """
        self._base_url = f"{scheme}://{host}"
        self.timeout = timeout
        self._session = requests.Session()
        self._session.auth = HTTPDigestAuth(username, password)

    def _url(self, path: str) -> str:
        """Join the base URL with an ISAPI ``path``."""
        return f"{self._base_url}{path}"

    def get_xml(self, path: str) -> Element:
        """GET ``path`` and parse the response body as XML.

        Args:
            path: ISAPI path (from ``isapi_endpoints``).

        Returns:
            Parsed root element.

        Raises:
            HikvisionTransportError: On network failure.
            HikvisionHTTPError: On a non-2xx response.
            HikvisionParseError: On a malformed XML body.
        """
        response = self._request("GET", path)
        return _parse_xml(response.text)

    def put_xml(self, path: str, body: Element | str) -> Element:
        """PUT an XML ``body`` to ``path`` and parse the response.

        Args:
            path: ISAPI path (from ``isapi_endpoints``).
            body: XML payload as a string or an Element.

        Returns:
            Parsed response root element (often a ``ResponseStatus``).

        Raises:
            HikvisionTransportError: On network failure.
            HikvisionHTTPError: On a non-2xx response.
            HikvisionParseError: On a malformed XML body.
        """
        if isinstance(body, str):
            data = body
        else:
            data = ET.tostring(body, encoding="unicode")
        response = self._request(
            "PUT",
            path,
            data=data,
            headers={"Content-Type": "application/xml"},
        )
        return _parse_xml(response.text)

    def get_bytes(self, path: str) -> bytes:
        """GET ``path`` and return the raw response body (e.g. snapshot JPEG).

        Args:
            path: ISAPI path (from ``isapi_endpoints``).

        Returns:
            Raw response bytes.

        Raises:
            HikvisionTransportError: On network failure.
            HikvisionHTTPError: On a non-2xx response.
        """
        response = self._request("GET", path)
        return response.content

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """Send a request and raise on transport or HTTP errors.

        Args:
            method: HTTP method.
            path: ISAPI path.
            data: Optional request body.
            headers: Optional request headers.

        Returns:
            The successful ``requests.Response``.

        Raises:
            HikvisionTransportError: On network failure.
            HikvisionHTTPError: On a non-2xx response.
        """
        try:
            response = self._session.request(
                method,
                self._url(path),
                data=data,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise HikvisionTransportError(f"Request to {path} failed: {exc}") from exc

        if not response.ok:
            raise classify_error_response(response.status_code, response.text)
        return response
