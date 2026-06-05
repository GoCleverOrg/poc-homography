"""Hikvision ISAPI adapter package.

Public surface for Phase A: the error hierarchy, the HTTP transport, and the
endpoint / unit-conversion helper modules. The ``HikvisionISAPIClient`` adapter
is added in Phase C and will be exported here then.
"""

from __future__ import annotations

from poc_homography.infrastructure.clients.hikvision import (
    isapi_endpoints,
    isapi_units,
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
