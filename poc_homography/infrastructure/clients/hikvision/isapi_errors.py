"""Exception hierarchy for the Hikvision ISAPI adapter.

``HikvisionError`` is a subclass of the domain-level ``CameraControllerError``
so existing ``except CameraControllerError`` consumers keep catching adapter
failures. The import flows from this module into
``poc_homography.domain.protocols.camera_controller``; that module does not
import this package, so there is no import cycle.
"""

from __future__ import annotations

from poc_homography.domain.protocols.camera_controller import CameraControllerError


class HikvisionError(CameraControllerError):
    """Base error for all Hikvision ISAPI adapter failures."""


class HikvisionTransportError(HikvisionError):
    """Network-level failure (connection refused, timeout, DNS, etc.)."""


class HikvisionHTTPError(HikvisionError):
    """Non-2xx HTTP response from the camera.

    Attributes:
        status_code: HTTP status code of the response.
        status_string: ISAPI ``statusString`` value, if present in the body.
        sub_status_code: ISAPI ``subStatusCode`` value, if present in the body.
    """

    def __init__(
        self,
        status_code: int,
        status_string: str | None = None,
        sub_status_code: str | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize the HTTP error.

        Args:
            status_code: HTTP status code of the response.
            status_string: ISAPI ``statusString`` value, if present.
            sub_status_code: ISAPI ``subStatusCode`` value, if present.
            message: Optional override for the exception message.
        """
        self.status_code = status_code
        self.status_string = status_string
        self.sub_status_code = sub_status_code
        if message is None:
            message = f"HTTP {status_code}"
            if status_string:
                message += f" ({status_string})"
            if sub_status_code:
                message += f": {sub_status_code}"
        super().__init__(message)


class HikvisionParseError(HikvisionError):
    """XML parse failure on a response body."""


class HikvisionUnsupportedError(HikvisionHTTPError):
    """403 response with ``subStatusCode == "methodNotAllowed"``.

    Raised when the camera firmware does not support the requested endpoint or
    method (e.g. a PUT-only setter that the hardware does not implement).
    """
