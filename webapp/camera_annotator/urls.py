"""URL patterns for camera annotator app."""

from django.urls import path

from . import views

app_name = "camera_annotator"

urlpatterns = [
    # Main page
    path("", views.index, name="index"),
    # API endpoints
    path("api/gcps/", views.api_gcps, name="api_gcps"),
    path("api/annotations/", views.api_annotations, name="api_annotations"),
    path("api/images/", views.api_images, name="api_images"),
    path("api/switch-image/", views.api_switch_image, name="api_switch_image"),
    path("api/save-annotations/", views.api_save_annotations, name="api_save_annotations"),
    # Image serving
    path("image/", views.serve_image, name="serve_image"),
]
