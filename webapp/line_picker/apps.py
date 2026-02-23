"""Django app configuration for line_picker."""

from django.apps import AppConfig
from homography_web.frame_utils import get_default_map_id, get_map_entity, get_map_image_path


class LinePickerConfig(AppConfig):
    """Configuration for the Line Picker Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "line_picker"
    verbose_name = "Line Picker"

    def ready(self) -> None:
        """Initialize line picker state when app loads."""
        from .state import get_state, initialize_state

        # Check if already initialized (avoid double initialization in runserver)
        try:
            get_state()
            return  # Already initialized
        except RuntimeError:
            pass

        # Use the default map
        default_map = get_default_map_id()
        if default_map is None:
            return  # No map configured for default tenant

        try:
            map_file = get_map_image_path(default_map)
            map_entity = get_map_entity(default_map)
        except FileNotFoundError:
            return

        initialize_state(
            map_file,
            default_map,
            width=int(map_entity.photo.width),
            height=int(map_entity.photo.height),
        )
