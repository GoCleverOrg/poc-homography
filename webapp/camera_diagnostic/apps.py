"""Django app configuration for camera_diagnostic."""

from django.apps import AppConfig


class CameraDiagnosticConfig(AppConfig):
    """Configuration for the Camera Diagnostic Tool Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "camera_diagnostic"
    verbose_name = "Camera Diagnostic Tool"
