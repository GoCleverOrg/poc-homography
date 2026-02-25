"""Custom Django middleware for the homography web application."""

from __future__ import annotations

from django.http import JsonResponse


class TenantIdMiddleware:
    """Return 400 JSON for ValueError exceptions related to tenant_id."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_exception(self, request, exception):
        if isinstance(exception, ValueError) and "tenant_id" in str(exception):
            return JsonResponse({"error": str(exception)}, status=400)
        return None
