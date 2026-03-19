"""URL patterns for point picker app."""

from django.urls import path

from . import views

app_name = "point_picker"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # Image API
    path("api/image/info/", views.api_image_info, name="api_image_info"),
    path("api/image/tile/", views.api_image_tile, name="api_image_tile"),
    path("api/image/full/", views.api_image_full, name="api_image_full"),
    # Points API
    path("api/points/", views.api_points, name="api_points"),
    path("api/points/next-id/", views.api_next_id, name="api_next_id"),
    path("api/points/<str:point_id>/", views.api_point_detail, name="api_point_detail"),
    # Coordinate API
    path("api/geo-coords/", views.api_geo_coords, name="api_geo_coords"),
]
