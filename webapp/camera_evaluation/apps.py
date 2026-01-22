"""Django app configuration for camera_evaluation."""

from django.apps import AppConfig


class CameraEvaluationConfig(AppConfig):
    """Configuration for the Camera Evaluation Tool Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "camera_evaluation"
    verbose_name = "Camera Evaluation Tool"
