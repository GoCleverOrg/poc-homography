"""URL patterns for camera diagnostic app."""

from django.urls import path

from . import views

app_name = "camera_diagnostic"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # API endpoints - Tenants and Cameras
    path("api/tenants/", views.api_tenants, name="api_tenants"),
    path("api/cameras/", views.api_cameras, name="api_cameras"),
    # RTSP streaming endpoints (camera_name is now camera_id like "valte_cam01")
    path("api/video-stream/<str:camera_name>/", views.api_video_stream, name="api_video_stream"),
    path("api/test-rtsp/<str:camera_name>/", views.api_test_rtsp, name="api_test_rtsp"),
    path(
        "api/capture-snapshot/<str:camera_name>/",
        views.api_capture_snapshot,
        name="api_capture_snapshot",
    ),
    # Web UI testing endpoint
    path("api/test-webui/<str:camera_name>/", views.api_test_webui, name="api_test_webui"),
    # PTZ API testing endpoint
    path("api/test-ptz/<str:camera_name>/", views.api_test_ptz, name="api_test_ptz"),
    # Diagnostic session endpoints
    path("api/diagnostic/run/", views.api_run_diagnostic, name="api_run_diagnostic"),
    path("api/diagnostic/sessions/", views.api_list_sessions, name="api_list_sessions"),
    path(
        "api/diagnostic/sessions/<str:session_id>/",
        views.api_get_session,
        name="api_get_session",
    ),
    path(
        "api/diagnostic/sessions/<str:session_id>/delete/",
        views.api_delete_session,
        name="api_delete_session",
    ),
    # Stress test endpoints
    path("api/stress-test/presets/", views.api_stress_test_presets, name="api_stress_test_presets"),
    path("api/stress-test/start/", views.api_stress_test_start, name="api_stress_test_start"),
    path(
        "api/stress-test/status/<str:session_id>/",
        views.api_stress_test_status,
        name="api_stress_test_status",
    ),
    path(
        "api/stress-test/abort/<str:session_id>/",
        views.api_stress_test_abort,
        name="api_stress_test_abort",
    ),
    path("api/stress-test/sessions/", views.api_stress_test_sessions, name="api_stress_test_sessions"),
    path(
        "api/stress-test/sessions/<str:session_id>/",
        views.api_stress_test_session_detail,
        name="api_stress_test_session_detail",
    ),
    path(
        "api/stress-test/sessions/<str:session_id>/evaluate/",
        views.api_stress_test_evaluate,
        name="api_stress_test_evaluate",
    ),
    path(
        "api/stress-test/sessions/<str:session_id>/delete/",
        views.api_stress_test_delete,
        name="api_stress_test_delete",
    ),
    path(
        "api/stress-test/video-stream/<str:camera_id>/",
        views.api_stress_video_stream,
        name="api_stress_video_stream",
    ),
]
