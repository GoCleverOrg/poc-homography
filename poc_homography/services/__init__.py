"""Domain services for the poc_homography system.

Services contain domain logic that doesn't naturally belong to an entity
or value object. They are stateless and take domain objects as arguments.

Key design decisions:
- All services are stateless - they receive dependencies as constructor args
- Services take separate domain objects (config, calibration, ptz_state)
  rather than fetching them from repositories
- PTZState is always passed as an argument, never stored

Note on CoordinateTransformService:
    The document originally planned a CoordinateTransformService, but the
    GeoTiff VO already has pixel_to_geo() and geo_to_pixel() methods.
    Since coordinate transforms are inherently tied to the GeoTiff data,
    having them on the VO is appropriate and a separate service is not needed.
"""

from poc_homography.services.homography import (
    HomographyStrategy,
    IntrinsicExtrinsicStrategy,
)
from poc_homography.services.orientation import (
    AdditiveOrientationStrategy,
    OrientationStrategy,
    RotationMatrixStrategy,
)

__all__ = [
    "AdditiveOrientationStrategy",
    "HomographyStrategy",
    "IntrinsicExtrinsicStrategy",
    "OrientationStrategy",
    "RotationMatrixStrategy",
]
