"""
URL configuration for GCP app.
"""

from django.urls import path

from . import views

app_name = "gcp"

urlpatterns = [
    # Main pages
    path("", views.index, name="index"),
    path("capture/", views.gcp_capture, name="gcp_capture"),
    path("debug/", views.debug_map, name="debug_map"),
    # API endpoints
    path("api/gcps/", views.api_get_gcps, name="api_get_gcps"),
    path("api/gcps/save/", views.api_save_gcps, name="api_save_gcps"),
    path("api/gcps/delete/<int:gcp_id>/", views.api_delete_gcp, name="api_delete_gcp"),
]
