"""URL patterns for line picker app."""

from django.urls import path

from . import views

app_name = "line_picker"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # Image API (reuse same pattern as point_picker for consistency)
    path("api/image/info/", views.api_image_info, name="api_image_info"),
    path("api/image/tile/", views.api_image_tile, name="api_image_tile"),
    path("api/image/full/", views.api_image_full, name="api_image_full"),
    # Lines API
    path("api/lines/", views.api_lines, name="api_lines"),
    path("api/lines/next-id/", views.api_next_line_id, name="api_next_line_id"),
    path("api/lines/<str:line_id>/", views.api_line_detail, name="api_line_detail"),
    # Geo-coordinate conversion
    path("api/geo-coords/", views.api_geo_coords, name="api_geo_coords"),
    # Import/Export API
    path("api/export/", views.api_export, name="api_export"),
    path("api/import/", views.api_import, name="api_import"),
]
