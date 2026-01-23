"""URL patterns for lens calibration app."""

from django.urls import path

from . import views

app_name = "lens_calibration"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # API endpoints
    path("api/calibrate/", views.api_calibrate, name="api_calibrate"),
    path("api/validate/", views.api_validate, name="api_validate"),
    path("api/save/", views.api_save, name="api_save"),
    path("api/load/", views.api_load, name="api_load"),
    path("api/survey-sessions/", views.api_survey_sessions, name="api_survey_sessions"),
]
