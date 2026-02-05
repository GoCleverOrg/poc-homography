"""URL patterns for lens calibration app."""

from django.urls import path

from . import views

app_name = "lens_calibration"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # API endpoints
    path("api/calibrate/", views.api_calibrate, name="api_calibrate"),
    path("api/calibrate-from-files/", views.api_calibrate_from_calibration_files, name="api_calibrate_from_files"),
    path("api/calibrate-annotated-lines/", views.api_calibrate_annotated_lines, name="api_calibrate_annotated_lines"),
    path("api/validate/", views.api_validate, name="api_validate"),
    path("api/save/", views.api_save, name="api_save"),
    path("api/load/", views.api_load, name="api_load"),
    path("api/survey-sessions/", views.api_survey_sessions, name="api_survey_sessions"),
    path("api/compute-intrinsics/", views.api_compute_intrinsics, name="api_compute_intrinsics"),
    path("api/test-data-files/", views.api_test_data_files, name="api_test_data_files"),
    path("api/test-data-file-content/", views.api_test_data_file_content, name="api_test_data_file_content"),
]
