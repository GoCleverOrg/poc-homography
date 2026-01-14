"""Intrinsic-extrinsic homography computation strategy.

This strategy computes homography using the classical computer vision approach:
- Build intrinsic matrix K from camera spec and zoom
- Build extrinsic matrix from camera position and orientation
- Compute H = K @ [r1, r2, t] for ground plane (Z=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from poc_homography.camera_geometry import CameraGeometry
from poc_homography.camera_parameters import CameraParameters, DistortionCoefficients
from poc_homography.domain.vo import CameraIntrinsics
from poc_homography.types import Degrees, Pixels, Unitless

if TYPE_CHECKING:
    from poc_homography.camera_parameters import CameraGeometryResult
    from poc_homography.domain.entities import CameraCalibration, CameraConfig, Map
    from poc_homography.domain.vo import Orientation, PTZState


class IntrinsicExtrinsicStrategy:
    """Strategy that computes homography using intrinsic/extrinsic decomposition.

    This strategy wraps the existing CameraGeometry.compute() method, translating
    the new domain model objects into the legacy CameraParameters format.

    The approach:
    1. Build CameraIntrinsics from CameraSpec and PTZ zoom
    2. Convert calibration position (map pixels) to world coordinates (meters)
    3. Use the pre-computed final orientation (yaw, pitch, roll)
    4. Delegate to CameraGeometry.compute() for the actual homography computation
    """

    def compute(
        self,
        config: CameraConfig,
        calibration: CameraCalibration,
        ptz_state: PTZState,
        map_entity: Map,
        final_orientation: Orientation,
    ) -> CameraGeometryResult:
        """Compute homography using intrinsic-extrinsic decomposition.

        Args:
            config: Camera configuration with hardware spec.
            calibration: Camera calibration data.
            ptz_state: Current PTZ state (used for zoom).
            map_entity: Map entity with GeoTiff metadata.
            final_orientation: Pre-computed final orientation.

        Returns:
            CameraGeometryResult with homography matrix and validation state.
        """
        spec = config.spec

        # Build intrinsics from spec and current zoom
        focal_length = spec.focal_length_at_zoom(ptz_state.zoom)
        intrinsics = CameraIntrinsics(
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

        # Camera position in world frame: [X, Y, Z] where Z is height
        camera_position = np.array([float(easting), float(northing), float(calibration.height)])

        # Build distortion coefficients
        dist = calibration.distortion
        distortion = DistortionCoefficients(
            k1=Unitless(float(dist.k1)),
            k2=Unitless(float(dist.k2)),
            p1=Unitless(float(dist.p1)),
            p2=Unitless(float(dist.p2)),
            k3=Unitless(0.0),
        )

        # Get map dimensions from photo
        map_width = Pixels(map_entity.photo.width)
        map_height = Pixels(map_entity.photo.height)

        # Calculate pixels per meter from GeoTiff (assuming square pixels)
        pixel_size_m = abs(float(geotiff.geotransform.pixel_width))  # meters per pixel
        pixels_per_meter = Unitless(1.0 / pixel_size_m) if pixel_size_m > 0 else Unitless(1.0)

        # Build CameraParameters for legacy CameraGeometry
        params = CameraParameters.create(
            image_width=intrinsics.image_width,
            image_height=intrinsics.image_height,
            intrinsic_matrix=intrinsics.K,
            camera_position=camera_position,
            pan_deg=Degrees(float(final_orientation.yaw)),
            tilt_deg=Degrees(float(final_orientation.pitch)),
            roll_deg=Degrees(float(final_orientation.roll)),
            map_width=map_width,
            map_height=map_height,
            pixels_per_meter=pixels_per_meter,
            distortion=distortion if dist.has_distortion else None,
        )

        # Delegate to existing CameraGeometry for homography computation
        return CameraGeometry.compute(params)
