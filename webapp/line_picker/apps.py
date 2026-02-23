"""Django app configuration for line_picker."""

from django.apps import AppConfig
from homography_web.frame_utils import DATA_MAPS_DIR, get_default_map


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
        map_entity = get_default_map()
        if map_entity is None:
            return  # No map configured for default tenant

        map_file = DATA_MAPS_DIR / map_entity.photo.path
        if not map_file.exists():
            return

        initialize_state(
            map_file,
            map_entity.id,
            width=int(map_entity.photo.width),
            height=int(map_entity.photo.height),
        )
