"""URL patterns for camera survey app."""

from django.urls import path

from . import views

app_name = "camera_survey"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # Tenant and camera selection
    path("api/tenants/", views.api_tenants, name="api_tenants"),
    path("api/cameras/", views.api_cameras, name="api_cameras"),
    # Survey control
    path("api/survey/start/", views.api_start_survey, name="api_start_survey"),
    path("api/survey/ptz-status/", views.api_ptz_status, name="api_ptz_status"),
    path("api/survey/<str:session_id>/status/", views.api_survey_status, name="api_survey_status"),
    path("api/survey/<str:session_id>/abort/", views.api_abort_survey, name="api_abort_survey"),
    # Session management
    path("api/survey/sessions/", views.api_sessions_list, name="api_sessions_list"),
    path("api/survey/sessions/<str:session_id>/", views.api_session_detail, name="api_session_detail"),
    path("api/survey/sessions/<str:session_id>/delete/", views.api_delete_session, name="api_delete_session"),
    path("api/survey/sessions/<str:session_id>/manifest/", views.api_session_manifest, name="api_session_manifest"),
    path("api/survey/sessions/<str:session_id>/images/<str:filename>", views.api_session_image, name="api_session_image"),
    # Presets
    path("api/survey/presets/", views.api_presets, name="api_presets"),
]
