"""OrthoLine value object — a line segment detected on a georeferenced orthophoto.

Each detected segment carries BOTH map-pixel endpoints (raster column/row) and
their UTM world projection (easting/northing), so downstream consumers can work
in whichever frame they need without re-running the affine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from poc_homography.types import Degrees, Easting, Meters, Northing, PixelsFloat


@dataclass(frozen=True)
class OrthoLine:
    """A straight-line segment detected on an orthophoto, in pixel + UTM frames.

    Attributes:
        start_pixel: (x, y) start endpoint in map-pixel coordinates.
        end_pixel: (x, y) end endpoint in map-pixel coordinates.
        start_utm: (easting, northing) start endpoint in CRS units.
        end_utm: (easting, northing) end endpoint in CRS units.
        angle_deg: Segment orientation in the pixel frame, degrees, 0 = horizontal.
        length_px: Segment length in pixels.
        length_m: Segment length in ground meters (from the UTM endpoints).
        confidence: Detector confidence (0.0 to 1.0).
        cluster_id: Parallel-line cluster id (-1 if unclustered).
    """

    start_pixel: tuple[PixelsFloat, PixelsFloat]
    end_pixel: tuple[PixelsFloat, PixelsFloat]
    start_utm: tuple[Easting, Northing]
    end_utm: tuple[Easting, Northing]
    angle_deg: Degrees
    length_px: PixelsFloat
    length_m: Meters
    confidence: float = 1.0
    cluster_id: int = -1

    @classmethod
    def from_endpoints(
        cls,
        start_pixel: tuple[float, float],
        end_pixel: tuple[float, float],
        start_utm: tuple[float, float],
        end_utm: tuple[float, float],
        *,
        confidence: float = 1.0,
        cluster_id: int = -1,
    ) -> OrthoLine:
        """Build an :class:`OrthoLine`, deriving angle and lengths from endpoints.

        Args:
            start_pixel: (x, y) start endpoint in pixels.
            end_pixel: (x, y) end endpoint in pixels.
            start_utm: (easting, northing) start endpoint.
            end_utm: (easting, northing) end endpoint.
            confidence: Detector confidence.
            cluster_id: Parallel-line cluster id.
        """
        dx = end_pixel[0] - start_pixel[0]
        dy = end_pixel[1] - start_pixel[1]
        length_px = math.hypot(dx, dy)
        angle_deg = math.degrees(math.atan2(dy, dx))

        de = end_utm[0] - start_utm[0]
        dn = end_utm[1] - start_utm[1]
        length_m = math.hypot(de, dn)

        return cls(
            start_pixel=(PixelsFloat(start_pixel[0]), PixelsFloat(start_pixel[1])),
            end_pixel=(PixelsFloat(end_pixel[0]), PixelsFloat(end_pixel[1])),
            start_utm=(Easting(start_utm[0]), Northing(start_utm[1])),
            end_utm=(Easting(end_utm[0]), Northing(end_utm[1])),
            angle_deg=Degrees(angle_deg),
            length_px=PixelsFloat(length_px),
            length_m=Meters(length_m),
            confidence=confidence,
            cluster_id=cluster_id,
        )


__all__ = ["OrthoLine"]
