"""
URL configuration for homography_precision app.

Separates page views from API endpoints for cleaner routing.
"""

from django.urls import path

from . import views

app_name = "homography_precision"

# Page views - render HTML templates
page_patterns = [
    path("", views.index, name="index"),
]

# API endpoints - return JSON
api_patterns = [
    path("api/test-cases/", views.api_test_cases, name="api_test_cases"),
    path("api/test-cases/<str:name>/", views.api_test_case_detail, name="api_test_case_detail"),
    path("api/gcp-registry/", views.api_gcp_registry, name="api_gcp_registry"),
    path("api/compute-homography/", views.api_compute_homography, name="api_compute_homography"),
    # Line validation endpoints
    path("api/line-test-cases/", views.api_line_test_cases, name="api_line_test_cases"),
    path(
        "api/line-test-cases/<str:name>/",
        views.api_line_test_case_detail,
        name="api_line_test_case_detail",
    ),
    path("api/line-registry/", views.api_line_registry, name="api_line_registry"),
    path(
        "api/compute-homography-from-lines/",
        views.api_compute_homography_from_lines,
        name="api_compute_homography_from_lines",
    ),
    path("api/compute-line-errors/", views.api_compute_line_errors, name="api_compute_line_errors"),
    # Image info endpoints
    path("api/camera-info/", views.api_camera_info, name="api_camera_info"),
    path("api/map-info/", views.api_map_info, name="api_map_info"),
    # Tile serving endpoints
    path("api/camera-tile/", views.api_camera_tile, name="api_camera_tile"),
    path("api/map-tile/", views.api_map_tile, name="api_map_tile"),
]

urlpatterns = page_patterns + api_patterns
