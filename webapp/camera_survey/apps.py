"""Django app configuration for camera_survey."""

from django.apps import AppConfig


class CameraSurveyConfig(AppConfig):
    """Configuration for the Camera Survey app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "camera_survey"
    verbose_name = "Camera Survey Tool"
