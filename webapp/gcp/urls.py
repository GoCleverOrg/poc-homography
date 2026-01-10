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
    path("capture/", views.gcp_capture, name="gcp_capture"),
    path("debug/", views.debug_map, name="debug_map"),
]

# API endpoints - return JSON
api_patterns = [
    path("api/gcps/", views.api_get_gcps, name="api_get_gcps"),
    path("api/gcps/save/", views.api_save_gcps, name="api_save_gcps"),
    path("api/gcps/delete/<int:gcp_id>/", views.api_delete_gcp, name="api_delete_gcp"),
]

urlpatterns = page_patterns + api_patterns
