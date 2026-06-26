"""URL patterns for lens calibration app."""

from django.urls import path

from . import views

app_name = "lens_calibration"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # API endpoints
    path("api/validate/", views.api_validate, name="api_validate"),
    path("api/save/", views.api_save, name="api_save"),
    path("api/load/", views.api_load, name="api_load"),
    path("api/calibration-ids/", views.api_calibration_ids, name="api_calibration_ids"),
    path("api/compute-intrinsics/", views.api_compute_intrinsics, name="api_compute_intrinsics"),
    path("api/line-trace-sets/", views.api_line_trace_sets, name="api_line_trace_sets"),
    path(
        "api/line-trace-set-detail/",
        views.api_line_trace_set_detail,
        name="api_line_trace_set_detail",
    ),
]
