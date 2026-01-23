"""URL patterns for Camera Evaluation Tool app.

Note: Stress test endpoints have been moved to camera_diagnostic app.
This app now only handles survey functionality.
"""

from django.urls import path

from . import views

app_name = "camera_evaluation"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # Common API endpoints
    path("api/tenants/", views.api_tenants, name="api_tenants"),
    path("api/cameras/", views.api_cameras, name="api_cameras"),
    # Video streaming (still needed for survey preview)
    path("api/video-stream/<str:camera_id>/", views.api_video_stream, name="api_video_stream"),
    # Survey API endpoints
    path("api/survey/presets/", views.api_survey_presets, name="api_survey_presets"),
    path("api/survey/start/", views.api_survey_start, name="api_survey_start"),
    path("api/survey/<str:session_id>/status/", views.api_survey_status, name="api_survey_status"),
    path("api/survey/<str:session_id>/abort/", views.api_survey_abort, name="api_survey_abort"),
    path("api/survey/sessions/", views.api_survey_sessions, name="api_survey_sessions"),
    path("api/survey/sessions/<str:session_id>/", views.api_survey_session_detail, name="api_survey_session_detail"),
    path("api/survey/sessions/<str:session_id>/manifest/", views.api_survey_session_manifest, name="api_survey_session_manifest"),
    path("api/survey/sessions/<str:session_id>/images/<str:filename>/", views.api_survey_session_image, name="api_survey_session_image"),
    path("api/survey/sessions/<str:session_id>/delete/", views.api_survey_delete_session, name="api_survey_delete_session"),
]
