"""Custom Django middleware for the homography web application."""

from __future__ import annotations

from django.http import JsonResponse

from homography_web.frame_utils import TenantIdError


class TenantIdMiddleware:
    """Return 400 JSON for TenantIdError exceptions."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_exception(self, request, exception):
        if isinstance(exception, TenantIdError):
            return JsonResponse({"error": str(exception)}, status=400)
        return None
