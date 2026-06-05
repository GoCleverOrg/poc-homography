"""Hikvision ISAPI adapter package.

Public surface: the error hierarchy, the HTTP transport, the endpoint /
unit-conversion helper modules, and the ``HikvisionISAPIClient`` adapter.
"""

from __future__ import annotations

from poc_homography.infrastructure.clients.hikvision import (
    isapi_endpoints,
    isapi_units,
)
from poc_homography.infrastructure.clients.hikvision.isapi_client import (
    HikvisionISAPIClient,
)
from poc_homography.infrastructure.clients.hikvision.isapi_errors import (
    HikvisionError,
    HikvisionHTTPError,
    HikvisionParseError,
    HikvisionTransportError,
    HikvisionUnsupportedError,
)
from poc_homography.infrastructure.clients.hikvision.isapi_transport import (
    IsapiTransport,
    classify_error_response,
    find,
    findtext,
)

__all__ = [
    "HikvisionISAPIClient",
    "HikvisionError",
    "HikvisionHTTPError",
    "HikvisionParseError",
    "HikvisionTransportError",
    "HikvisionUnsupportedError",
    "IsapiTransport",
    "classify_error_response",
    "find",
    "findtext",
    "isapi_endpoints",
    "isapi_units",
]
