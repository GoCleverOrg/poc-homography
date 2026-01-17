"""Intrinsic-extrinsic homography computation strategy.

This strategy computes homography using the classical computer vision approach:
- Build intrinsic matrix K from camera spec and zoom
- Build extrinsic matrix from camera position and orientation
- Compute H = K @ [r1, r2, t] for ground plane (Z=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.domain.vo import CameraIntrinsics, Homography, Vector3

if TYPE_CHECKING:
    from poc_homography.domain.entities import CameraCalibration, CameraConfig, Map
    from poc_homography.domain.vo import Orientation, PTZState


class StrategyIntrinsicExtrinsic:
    """Strategy that computes homography using intrinsic/extrinsic decomposition.

    This strategy uses the CameraGeometry.compute_from_vo() method with domain VOs.

    The approach:
    1. Build CameraIntrinsics from CameraSpec and PTZ zoom
    2. Convert calibration position (map pixels) to world coordinates (meters)
    3. Use the pre-computed final orientation (yaw, pitch, roll)
    4. Delegate to CameraGeometry.compute_from_vo() for the actual homography computation
    """

    def compute(
        self,
        config: CameraConfig,
        calibration: CameraCalibration,
        ptz_state: PTZState,
        map_entity: Map,
        final_orientation: Orientation,
    ) -> Homography:
        """Compute homography using intrinsic-extrinsic decomposition.

        Args:
            config: Camera configuration with hardware spec.
            calibration: Camera calibration data.
            ptz_state: Current PTZ state (used for zoom).
            map_entity: Map entity with GeoTiff metadata.
            final_orientation: Pre-computed final orientation.

        Returns:
            Homography VO with matrices and projection methods.
        """
        spec = config.spec

        # Build intrinsics from spec and current zoom
        focal_length = spec.focal_length_at_zoom(ptz_state.zoom)
        intrinsics = CameraIntrinsics.create(
            sensor_width=spec.sensor_width,
            base_focal_length=spec.base_focal_length,
            image_width=spec.image_width,
            image_height=spec.image_height,
            focal_length=focal_length,
        )

        # Convert camera position from map pixels to world coordinates (meters)
        # Using GeoTiff to get real-world position
        geotiff = map_entity.geotiff
        pos = calibration.position

        # Get world coordinates from pixel position
        easting, northing = geotiff.pixel_to_geo(pos.x, pos.y)

        # Camera position as VO
        camera_position = Vector3.create(float(easting), float(northing), float(calibration.height))

        # Get distortion if present
        distortion = calibration.distortion if calibration.distortion.has_distortion else None

        # Delegate to CameraGeometry for homography computation
        return CameraGeometry.compute_from_vo(
            intrinsics=intrinsics,
            camera_position=camera_position,
            orientation=final_orientation,
            distortion=distortion,
        )
