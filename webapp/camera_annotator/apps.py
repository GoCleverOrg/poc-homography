"""Django app configuration for camera_annotator."""

from django.apps import AppConfig


class CameraAnnotatorConfig(AppConfig):
    """Configuration for the Camera Frame Annotator Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "camera_annotator"
    verbose_name = "Camera Frame Annotator"
