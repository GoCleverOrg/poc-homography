"""URL patterns for distortion validator app."""

from django.urls import path

from . import views

app_name = "distortion_validator"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # API endpoints
    path("api/calibration-files/", views.api_calibration_files, name="api_calibration_files"),
    path("api/load-calibration/", views.api_load_calibration, name="api_load_calibration"),
    path("api/images/", views.api_images, name="api_images"),
    path("api/undistort/", views.api_undistort, name="api_undistort"),
    path("api/measure-straightness/", views.api_measure_straightness, name="api_measure_straightness"),
    path("api/transform-points/", views.api_transform_points, name="api_transform_points"),
    path("api/result-image/<str:filename>", views.api_serve_result_image, name="api_serve_result_image"),
    path("api/compute-intrinsics/", views.api_compute_intrinsics, name="api_compute_intrinsics"),
]
