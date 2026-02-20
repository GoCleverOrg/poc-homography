"""
URL configuration for GCP app.

Separates page views from API endpoints for cleaner routing.
"""

from django.urls import path

from . import views

app_name = "gcp"

# Page views - render HTML templates
page_patterns = [
    path("", views.index, name="index"),
    path("debug/", views.debug_map, name="debug_map"),
]

# API endpoints
api_patterns = [
    path("api/tenants/", views.api_tenants, name="api_tenants"),
    path(
        "api/tenants/<str:tenant_id>/maps/",
        views.api_tenant_maps,
        name="api_tenant_maps",
    ),
    path("api/map-ids/", views.api_map_ids, name="api_map_ids"),
]

urlpatterns = page_patterns + api_patterns
