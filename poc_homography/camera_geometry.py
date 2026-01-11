from __future__ import annotations

# camera_geometry.py
import logging
import math
import warnings
from typing import Any

import numpy as np

from poc_homography.camera_parameters import (
    CameraGeometryResult,
    CameraParameters,
    DistortionCoefficients,
    HeightUncertainty,
)
from poc_homography.types import Degrees, Meters, Millimeters, Pixels, Unitless

logger = logging.getLogger(__name__)


class CameraGeometry:
    """
    Handles all spatial and projection calculations for a PTZ camera.
    Calculates the homography matrix H to map image points to the world ground plane.

    COORDINATE SYSTEM CONVENTIONS:
    ===============================
    World Frame (Right-Handed):
      - Origin: Arbitrary reference point (typically camera location or scene center)
      - X-axis: East (positive = East, negative = West)
      - Y-axis: North (positive = North, negative = South)
      - Z-axis: Up (positive = Up, height above ground)
      - Ground plane: Z = 0

    Camera Frame (Right-Handed, standard computer vision):
      - Origin: Camera optical center
      - X-axis: Right (in image)
      - Y-axis: Down (in image)
      - Z-axis: Forward (along optical axis, into the scene)

    Image Frame:
      - Origin: Top-left corner
      - u-axis: Right (width)
      - v-axis: Down (height)
      - Units: Pixels

    HOMOGRAPHY:
    ===========
    The homography H maps world ground plane points (Z=0) to image pixels:
      [u]       [X_world]
      [v]  ∝ H  [Y_world]
      [1]       [1      ]

    For inverse (image to world):
      [X_world]           [u]
      [Y_world]  ∝ H^-1  [v]
      [1      ]           [1]

    CAMERA PARAMETERS:
    ==================
    - K: 3x3 intrinsic matrix (focal length, principal point)
    - w_pos: Camera position [X, Y, Z] in world coordinates (meters)
    - pan_deg: Horizontal rotation (degrees, positive = right/clockwise from above)
    - tilt_deg: Vertical rotation (degrees, positive = down, Hikvision convention)
    """

    # Validation constants
    ZOOM_MIN = 1.0
    ZOOM_MAX = 25.0
    ZOOM_WARN_HIGH = 20.0

    TILT_MIN = 0.0  # Must be positive (pointing down)
    TILT_MAX = 90.0
    TILT_WARN_LOW = 10.0
    TILT_WARN_HIGH = 80.0

    HEIGHT_MIN = 1.0  # meters
    HEIGHT_MAX = 50.0
    HEIGHT_WARN_LOW = 2.0
    HEIGHT_WARN_HIGH = 30.0

    FOV_MIN_DEG = 2.0
    FOV_MAX_DEG = 120.0
    FOV_WARN_MIN_DEG = 10.0
    FOV_WARN_MAX_DEG = 90.0

    CONDITION_WARN = 1e6
    CONDITION_ERROR = 1e10

    # Maximum reasonable distance ratio for ground projection validation
    # Set to 20x camera height as a heuristic for typical PTZ camera scenarios
    # Beyond this, projections likely indicate near-horizontal viewing angles
    MAX_DISTANCE_HEIGHT_RATIO = 20.0

    def __init__(self, w: Pixels, h: Pixels):
        """
        Initializes geometry class with image dimensions.

        Args:
            w: Image width (pixels).
            h: Image height (pixels).
        """
        self.w, self.h = w, h
        self.H = np.eye(3)  # Homography matrix (Image -> Map)
        self.H_inv = np.eye(3)  # Inverse Homography (Map -> Image)
        self.map_width = 640  # Default scale for the map view (pixels)
        self.map_height = 640  # Default scale for the map view (pixels)

        # Pixels per meter for mapping world coords to the side-panel map
        self.PPM = 100

        # Affine transformation matrix A from reference image pixels to world coordinates
        # For georeferenced ortho imagery (GeoTIFF), transforms reference pixels to UTM meters
        # A = [[pixel_size_x, 0, t_x], [0, pixel_size_y, t_y], [0, 0, 1]]
        # where t_x = origin_easting - camera_easting, t_y = origin_northing - camera_northing
        # Defaults to identity when geotiff parameters not provided
        self.A = np.eye(3)

        # Placeholder for real-world geometry parameters (to be set by user)
        self.K: np.ndarray | None = None  # Intrinsic Matrix
        self.w_pos: np.ndarray | None = None  # Camera World Position [Xw, Yw, Zw]
        self.height_m = 5.0  # Camera Height (m)
        self.pan_deg = 0.0  # Camera Pan (Yaw)
        self.tilt_deg = 0.0  # Camera Tilt (Pitch)
        self.roll_deg = 0.0  # Camera Roll (rotation around optical axis)

        # Height uncertainty for propagation (set via set_height_uncertainty)
        self.height_uncertainty_lower: Meters | None = (
            None  # Lower bound of height confidence interval
        )
        self.height_uncertainty_upper: Meters | None = (
            None  # Upper bound of height confidence interval
        )

        # Lens distortion coefficients (OpenCV model)
        # Radial: k1, k2, k3 (k3 often 0 for simple lenses)
        # Tangential: p1, p2
        self._dist_coeffs: np.ndarray | None = None  # np.array([k1, k2, p1, p2, k3]) when set
        self._use_distortion = False

    @staticmethod
    def get_intrinsics(
        zoom_factor: Unitless,
        W_px: Pixels = Pixels(1920),
        H_px: Pixels = Pixels(1080),
        sensor_width_mm: Millimeters = Millimeters(6.78),
    ) -> np.ndarray:
        """
        Calculates the 3x3 Intrinsic Matrix K based on camera specifications and zoom factor.

        Args:
            zoom_factor (float): Digital or optical zoom multiplier (e.g., 1.0 for no zoom).
            W_px (int): Image width in pixels.
            H_px (int): Image height in pixels.
            sensor_width_mm (float): Sensor width in millimeters.

        Returns:
            K (3x3): Intrinsic camera matrix.

        Raises:
            ValueError: If zoom_factor is outside valid range [ZOOM_MIN, ZOOM_MAX].
        """
        # Validate zoom factor
        if zoom_factor < CameraGeometry.ZOOM_MIN or zoom_factor > CameraGeometry.ZOOM_MAX:
            raise ValueError(
                f"Zoom factor {zoom_factor} is out of valid range "
                f"[{CameraGeometry.ZOOM_MIN}, {CameraGeometry.ZOOM_MAX}]"
            )

        if zoom_factor > CameraGeometry.ZOOM_WARN_HIGH:
            logger.warning(
                f"Zoom factor {zoom_factor} is very high (>{CameraGeometry.ZOOM_WARN_HIGH}). "
                f"Results may be less accurate at extreme zoom levels."
            )

        # 1. Calculate focal length in mm
        # Based on Hikvision DS-2DF8425IX-AELW datasheet:
        #   - Focal length: 5.9mm (wide) to 147.5mm (tele)
        #   - Optical zoom: 25x (linear: f = 5.9mm × zoom_factor)
        f_mm = 5.9 * zoom_factor

        f_px = f_mm * (W_px / sensor_width_mm)

        # 3. Construct K
        cx, cy = W_px / 2.0, H_px / 2.0
        K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])
        return K

    def set_geotiff_params(
        self, geotiff_params: dict[str, Any] | None, camera_utm_position: tuple[float, float] | None
    ) -> None:
        """
        Set GeoTIFF parameters to compute the affine transformation matrix A.

        The A matrix transforms reference image pixels to world ground plane coordinates
        (UTM meters). Supports GDAL's 6-parameter GeoTransform for full affine transformation
        including rotation (e.g., for rotated rasters).

        GDAL GeoTransform Formula:
            Xgeo = GT[0] + P*GT[1] + L*GT[2]
            Ygeo = GT[3] + P*GT[4] + L*GT[5]

        A Matrix Construction:
            A = [[GT[1], GT[2], t_x],
                 [GT[4], GT[5], t_y],
                 [0,     0,     1  ]]

        Where:
            GT[0]: X-coordinate of upper-left corner (origin easting)
            GT[1]: Pixel width (meters per pixel in X direction)
            GT[2]: Row rotation (typically 0 for north-up images)
            GT[3]: Y-coordinate of upper-left corner (origin northing)
            GT[4]: Column rotation (typically 0 for north-up images)
            GT[5]: Pixel height (meters per pixel in Y direction, typically negative)
            t_x = GT[0] - camera_easting
            t_y = GT[3] - camera_northing

        Pixel Origin Convention:
            GDAL GeoTransform references the UPPER-LEFT CORNER of a pixel.
            This is the standard convention for raster data.

        Args:
            geotiff_params: Dictionary containing GeoTIFF metadata. Supports two formats:

                **New format (recommended):**
                {
                    'geotransform': [GT0, GT1, GT2, GT3, GT4, GT5]
                }

                **Legacy format (deprecated):**
                {
                    'pixel_size_x': float,
                    'pixel_size_y': float,
                    'origin_easting': float,
                    'origin_northing': float
                }
                Note: Legacy format assumes no rotation (GT[2]=0, GT[4]=0).
                A deprecation warning will be issued if legacy format is used.

                If None, A matrix is set to identity (backward compatibility).

            camera_utm_position: Tuple of (easting, northing) for camera position in UTM (meters).
                If None, A matrix is set to identity (backward compatibility).

        Raises:
            ValueError: If geotransform has incorrect length or missing required keys.
            TypeError: If geotransform elements or camera_utm_position are not numeric.
            ValueError: If geotransform elements are non-finite or pixel sizes are zero.

        Examples:
            >>> geo = CameraGeometry(1920, 1080)

            >>> # New format: north-up raster
            >>> geotiff_params = {
            ...     'geotransform': [737575.05, 0.15, 0, 4391595.45, 0, -0.15]
            ... }
            >>> camera_utm_position = (737575.0, 4391595.0)
            >>> geo.set_geotiff_params(geotiff_params, camera_utm_position)

            >>> # New format: rotated raster (22.5° clockwise)
            >>> geotiff_params_rotated = {
            ...     'geotransform': [500000, 0.1387, 0.0574, 4400000, 0.0574, -0.1387]
            ... }
            >>> camera_utm_position = (500000, 4400000)
            >>> geo.set_geotiff_params(geotiff_params_rotated, camera_utm_position)
        """
        # Backward compatibility: if either parameter is None, set A to identity
        if geotiff_params is None or camera_utm_position is None:
            self.A = np.eye(3)
            return

        # Validate camera_utm_position is a tuple
        if not isinstance(camera_utm_position, tuple):
            raise TypeError(
                f"camera_utm_position must be a tuple, got {type(camera_utm_position).__name__}"
            )

        # Validate camera_utm_position has exactly 2 elements
        if len(camera_utm_position) != 2:
            raise ValueError(
                f"camera_utm_position must be a 2-tuple (easting, northing), "
                f"got {len(camera_utm_position)} elements"
            )

        camera_easting, camera_northing = camera_utm_position

        # Validate camera_utm_position values are numeric and finite
        for i, (name, value) in enumerate(
            [("easting", camera_easting), ("northing", camera_northing)]
        ):
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"camera_utm_position[{i}] ({name}) must be numeric (int or float), "
                    f"got {type(value).__name__}"
                )
            if not np.isfinite(value):
                raise ValueError(f"camera_utm_position[{i}] ({name}) must be finite, got {value}")

        # Check format: new geotransform format vs legacy 4-parameter format
        if "geotransform" in geotiff_params:
            # New format: 6-parameter GDAL GeoTransform
            gt = geotiff_params["geotransform"]

            # Validate geotransform is a list/tuple
            if not isinstance(gt, (list, tuple)):
                raise TypeError(f"geotransform must be a list or tuple, got {type(gt).__name__}")

            # Validate geotransform has exactly 6 elements
            if len(gt) != 6:
                raise ValueError(
                    f"geotransform must have exactly 6 elements (GDAL GeoTransform), "
                    f"got {len(gt)} elements"
                )

            # Validate all elements are numeric and finite
            for i, val in enumerate(gt):
                if not isinstance(val, (int, float)):
                    raise TypeError(
                        f"geotransform[{i}] must be numeric (int or float), "
                        f"got {type(val).__name__}"
                    )
                if not np.isfinite(val):
                    raise ValueError(f"geotransform[{i}] must be finite, got {val}")

            # Validate pixel sizes (GT[1] and GT[5]) are non-zero
            if gt[1] == 0:
                raise ValueError(f"GT[1] (pixel width) must be non-zero, got {gt[1]}")
            if gt[5] == 0:
                raise ValueError(f"GT[5] (pixel height) must be non-zero, got {gt[5]}")

            # Extract GeoTransform parameters
            origin_easting = gt[0]
            pixel_width = gt[1]
            row_rotation = gt[2]
            origin_northing = gt[3]
            col_rotation = gt[4]
            pixel_height = gt[5]

            # Compute translation components
            t_x = origin_easting - camera_easting
            t_y = origin_northing - camera_northing

            # Construct A matrix with rotation terms
            self.A = np.array(
                [
                    [pixel_width, row_rotation, t_x],
                    [col_rotation, pixel_height, t_y],
                    [0.0, 0.0, 1.0],
                ]
            )

            # Log A matrix computation at INFO level
            rotation_info = ""
            if row_rotation != 0 or col_rotation != 0:
                rotation_info = (
                    f"\n  Rotation: GT[2]={row_rotation}, GT[4]={col_rotation} (rotated raster)"
                )

            logger.info(
                f"Computed affine transformation matrix A from GeoTransform:\n"
                f"  GeoTransform: {gt}{rotation_info}\n"
                f"  Reference origin: ({origin_easting}, {origin_northing}) UTM meters\n"
                f"  Camera position: ({camera_easting}, {camera_northing}) UTM meters\n"
                f"  Translation: ({t_x}, {t_y}) meters\n"
                f"  A matrix:\n{self.A}"
            )

        else:
            # Legacy format: 4-parameter (pixel_size_x, pixel_size_y, origin_easting, origin_northing)
            # Issue deprecation warning
            warnings.warn(
                "Using legacy geotiff_params format with separate keys "
                "(pixel_size_x, pixel_size_y, origin_easting, origin_northing) is deprecated. "
                "Please use the new format with 'geotransform' key containing a 6-element array. "
                "Legacy format assumes no rotation (GT[2]=0, GT[4]=0). "
                "To convert: geotiff_params = {'geotransform': [origin_easting, pixel_size_x, 0, "
                "origin_northing, 0, pixel_size_y]}",
                DeprecationWarning,
                stacklevel=2,
            )

            # Validate legacy format contains required keys
            required_keys = ["pixel_size_x", "pixel_size_y", "origin_easting", "origin_northing"]
            for key in required_keys:
                if key not in geotiff_params:
                    raise ValueError(
                        f"geotiff_params missing required key: '{key}'. "
                        f"Required keys: {required_keys}"
                    )

            # Extract legacy parameters
            pixel_size_x = geotiff_params["pixel_size_x"]
            pixel_size_y = geotiff_params["pixel_size_y"]
            origin_easting = geotiff_params["origin_easting"]
            origin_northing = geotiff_params["origin_northing"]

            # Validate numeric types and values
            for name, value in [
                ("pixel_size_x", pixel_size_x),
                ("pixel_size_y", pixel_size_y),
                ("origin_easting", origin_easting),
                ("origin_northing", origin_northing),
            ]:
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"geotiff_params['{name}'] must be numeric (int or float), "
                        f"got {type(value).__name__}"
                    )
                if not np.isfinite(value):
                    raise ValueError(f"geotiff_params['{name}'] must be finite, got {value}")

            # Validate pixel sizes are non-zero
            if pixel_size_x == 0:
                raise ValueError(f"pixel_size_x must be non-zero, got {pixel_size_x}")
            if pixel_size_y == 0:
                raise ValueError(f"pixel_size_y must be non-zero, got {pixel_size_y}")

            # Compute translation components
            t_x = origin_easting - camera_easting
            t_y = origin_northing - camera_northing

            # Construct A matrix (no rotation)
            self.A = np.array([[pixel_size_x, 0.0, t_x], [0.0, pixel_size_y, t_y], [0.0, 0.0, 1.0]])

            # Log A matrix computation at INFO level
            logger.info(
                f"Computed affine transformation matrix A from legacy GeoTIFF parameters:\n"
                f"  Pixel size: ({pixel_size_x}, {pixel_size_y}) meters/pixel\n"
                f"  Reference origin: ({origin_easting}, {origin_northing}) UTM meters\n"
                f"  Camera position: ({camera_easting}, {camera_northing}) UTM meters\n"
                f"  Translation: ({t_x}, {t_y}) meters\n"
                f"  A matrix:\n{self.A}"
            )

    def set_distortion_coefficients(
        self,
        k1: Unitless = Unitless(0.0),
        k2: Unitless = Unitless(0.0),
        p1: Unitless = Unitless(0.0),
        p2: Unitless = Unitless(0.0),
        k3: Unitless = Unitless(0.0),
    ) -> None:
        """
        Set lens distortion coefficients using the OpenCV distortion model.

        The distortion model corrects for radial and tangential lens distortion:
        - Radial distortion (barrel/pincushion): controlled by k1, k2, k3
        - Tangential distortion (decentering): controlled by p1, p2

        For most PTZ cameras, only k1 (and sometimes k2) are significant.
        Positive k1 = barrel distortion (edges curve outward)
        Negative k1 = pincushion distortion (edges curve inward)

        Args:
            k1: First radial distortion coefficient (most significant)
            k2: Second radial distortion coefficient
            p1: First tangential distortion coefficient
            p2: Second tangential distortion coefficient
            k3: Third radial distortion coefficient (usually 0)

        Example:
            >>> geo = CameraGeometry(1920, 1080)
            >>> # Typical barrel distortion for wide-angle PTZ
            >>> geo.set_distortion_coefficients(k1=-0.1, k2=0.01)
        """
        self._dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        self._use_distortion = not np.allclose(self._dist_coeffs, 0.0)

        if self._use_distortion:
            logger.info(
                f"Distortion coefficients set: k1={k1:.6f}, k2={k2:.6f}, "
                f"p1={p1:.6f}, p2={p2:.6f}, k3={k3:.6f}"
            )
        else:
            logger.info("Distortion coefficients cleared (all zero)")

    def get_distortion_coefficients(self) -> np.ndarray | None:
        """
        Get current distortion coefficients.

        Returns:
            numpy array [k1, k2, p1, p2, k3] or None if not set
        """
        return self._dist_coeffs.copy() if self._dist_coeffs is not None else None

    def undistort_point(self, u: float, v: float, iterations: int = 10) -> tuple[float, float]:
        """
        Undistort a single image point to remove lens distortion.

        Converts a distorted pixel coordinate to the corresponding undistorted
        coordinate that would be observed with an ideal pinhole camera.

        This uses an iterative method to invert the distortion model.

        Args:
            u: Distorted x pixel coordinate
            v: Distorted y pixel coordinate
            iterations: Number of iterations for undistortion (default: 10)

        Returns:
            Tuple (u_undistorted, v_undistorted) in pixel coordinates

        Note:
            If no distortion coefficients are set, returns the input unchanged.
        """
        if not self._use_distortion or self.K is None or self._dist_coeffs is None:
            return (u, v)

        # Get intrinsic parameters
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        k1, k2, p1, p2, k3 = self._dist_coeffs

        # Convert to normalized camera coordinates
        x_d = (u - cx) / fx
        y_d = (v - cy) / fy

        # Iterative undistortion (Newton-Raphson style)
        x = x_d
        y = y_d

        for _ in range(iterations):
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2

            # Radial distortion factor
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

            # Tangential distortion
            dx_tangential = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

            # Update estimate (inverse iteration)
            x = (x_d - dx_tangential) / radial
            y = (y_d - dy_tangential) / radial

        # Convert back to pixel coordinates
        u_undist = x * fx + cx
        v_undist = y * fy + cy

        return (u_undist, v_undist)

    def undistort_points(
        self, points: list[tuple[float, float]], iterations: int = 10
    ) -> list[tuple[float, float]]:
        """
        Undistort multiple image points.

        Args:
            points: List of (u, v) distorted pixel coordinates
            iterations: Number of iterations for undistortion

        Returns:
            List of (u, v) undistorted pixel coordinates
        """
        return [self.undistort_point(u, v, iterations) for u, v in points]

    def distort_point(self, u: float, v: float) -> tuple[float, float]:
        """
        Apply lens distortion to an undistorted point.

        Converts an ideal pinhole camera coordinate to the corresponding
        distorted coordinate as observed through the actual lens.

        Args:
            u: Undistorted x pixel coordinate
            v: Undistorted y pixel coordinate

        Returns:
            Tuple (u_distorted, v_distorted) in pixel coordinates

        Note:
            If no distortion coefficients are set, returns the input unchanged.
        """
        if not self._use_distortion or self.K is None or self._dist_coeffs is None:
            return (u, v)

        # Get intrinsic parameters
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        k1, k2, p1, p2, k3 = self._dist_coeffs

        # Convert to normalized camera coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        # Radial distortion
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        # Tangential distortion
        dx_tangential = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        # Apply distortion
        x_d = x * radial + dx_tangential
        y_d = y * radial + dy_tangential

        # Convert back to pixel coordinates
        u_dist = x_d * fx + cx
        v_dist = y_d * fy + cy

        return (u_dist, v_dist)

    # Roll validation thresholds
    ROLL_WARN_THRESHOLD = 5.0  # Warning when |roll_deg| > 5.0
    ROLL_ERROR_THRESHOLD = 15.0  # Error when |roll_deg| > 15.0

    @classmethod
    def compute(cls, params: CameraParameters) -> CameraGeometryResult:
        """Compute homography from camera parameters (pure function).

        This is a pure classmethod that computes the ground plane homography
        without modifying any instance state. It takes an immutable
        CameraParameters config and returns an immutable CameraGeometryResult.

        This enables functional-style programming patterns:
        - Deterministic: same inputs always produce same outputs
        - Parallelizable: no shared mutable state
        - Testable: easy to test with known inputs/outputs
        - Cacheable: results can be cached based on input parameters

        Args:
            params: Immutable camera configuration containing all required
                parameters for homography computation.

        Returns:
            CameraGeometryResult containing the computed homography matrices
            and validation state.

        Raises:
            ValueError: If parameters fail validation checks.

        Example:
            >>> params = CameraParameters.create(
            ...     image_width=Pixels(1920),
            ...     image_height=Pixels(1080),
            ...     intrinsic_matrix=K,
            ...     camera_position=np.array([0.0, 0.0, 10.0]),
            ...     pan_deg=Degrees(0.0),
            ...     tilt_deg=Degrees(45.0),
            ...     roll_deg=Degrees(0.0),
            ...     map_width=Pixels(640),
            ...     map_height=Pixels(640),
            ...     pixels_per_meter=Unitless(100.0),
            ... )
            >>> result = CameraGeometry.compute(params)
            >>> if result.is_valid:
            ...     H = result.homography_matrix
        """
        # Extract parameters from immutable config
        K = params.intrinsic_matrix
        w_pos = params.camera_position
        pan_deg = params.pan_deg
        tilt_deg = params.tilt_deg
        roll_deg = params.roll_deg
        image_width = params.image_width
        image_height = params.image_height

        # Validate parameters (using class validation constants)
        cls._validate_parameters_static(K, w_pos, pan_deg, tilt_deg, roll_deg)

        # Log parameters at INFO level
        logger.info("Computing homography from CameraParameters:")
        logger.info(f"  Camera position: [{w_pos[0]:.2f}, {w_pos[1]:.2f}, {w_pos[2]:.2f}] meters")
        logger.info(f"  Pan: {pan_deg:.1f}°, Tilt: {tilt_deg:.1f}°, Roll: {roll_deg:.1f}°")
        logger.info(f"  Image dimensions: {image_width}x{image_height} pixels")

        # Compute rotation matrix
        R = cls._get_rotation_matrix_static(pan_deg, tilt_deg, roll_deg)

        # Compute homography matrix
        H = cls._calculate_ground_homography_static(K, w_pos, R)

        # Collect validation messages
        validation_messages: list[str] = []
        is_valid = True

        # Validate and compute inverse homography with condition number checks
        det_H = float(np.linalg.det(H))
        if abs(det_H) < 1e-10:
            validation_messages.append(
                f"Homography matrix is singular (det={det_H:.2e}). Cannot compute inverse."
            )
            is_valid = False
            # Return invalid result with identity inverse
            return CameraGeometryResult.create(
                homography_matrix=H,
                inverse_homography_matrix=np.eye(3),
                condition_number=float("inf"),
                determinant=det_H,
                is_valid=False,
                validation_messages=tuple(validation_messages),
                center_projection_distance=None,
            )

        # Compute condition number
        cond_H = float(np.linalg.cond(H))
        if cond_H > cls.CONDITION_ERROR:
            validation_messages.append(
                f"Homography condition number {cond_H:.2e} exceeds error threshold ({cls.CONDITION_ERROR:.2e}). "
                f"Matrix is ill-conditioned."
            )
            is_valid = False
        elif cond_H > cls.CONDITION_WARN:
            validation_messages.append(
                f"Homography condition number {cond_H:.2e} exceeds warning threshold ({cls.CONDITION_WARN:.2e}). "
                f"Inverse may be numerically unstable."
            )

        # Compute inverse
        H_inv = np.asarray(np.linalg.inv(H))

        # Validate projection and compute center distance
        center_distance = cls._compute_center_projection_distance(
            H_inv, w_pos, image_width, image_height, validation_messages
        )

        # Check for unreasonable projection distance
        height_m = w_pos[2]
        if center_distance is not None:
            max_reasonable_distance = height_m * cls.MAX_DISTANCE_HEIGHT_RATIO
            if center_distance > max_reasonable_distance:
                validation_messages.append(
                    f"Projected ground distance {center_distance:.2f}m is very large "
                    f"(>{max_reasonable_distance:.2f}m). Near-horizontal tilt angle suspected."
                )

        logger.info("Homography computation complete.")
        logger.info(f"  Homography det(H): {det_H:.2e}")
        logger.info(f"  Homography condition number: {cond_H:.2e}")
        if center_distance is not None:
            logger.info(f"  Image center projects to ground at distance: {center_distance:.2f}m")

        return CameraGeometryResult.create(
            homography_matrix=H,
            inverse_homography_matrix=H_inv,
            condition_number=cond_H,
            determinant=det_H,
            is_valid=is_valid,
            validation_messages=tuple(validation_messages),
            center_projection_distance=Meters(center_distance)
            if center_distance is not None
            else None,
        )

    def compute_and_update(self, params: CameraParameters) -> CameraGeometryResult:
        """Compute homography and update instance state for backward compatibility.

        This method provides a bridge between the new immutable pattern and
        the existing mutable CameraGeometry API. It:
        1. Calls compute(params) to get the immutable result
        2. Updates instance state (H, H_inv, K, w_pos, etc.)
        3. Returns the CameraGeometryResult

        This enables gradual migration: new code can use the immutable result,
        while existing code continues to work with mutable instance state.

        Args:
            params: Immutable camera configuration.

        Returns:
            CameraGeometryResult containing the computed homography matrices
            and validation state.

        Raises:
            ValueError: If parameters fail validation or homography is invalid.

        Example:
            >>> geo = CameraGeometry(1920, 1080)
            >>> params = CameraParameters.create(...)
            >>> result = geo.compute_and_update(params)
            >>> # Instance state is updated
            >>> assert np.allclose(geo.H, result.homography_matrix)
            >>> # Result can also be used immutably
            >>> H = result.homography_matrix
        """
        # Compute using the pure function
        result = CameraGeometry.compute(params)

        # Check if result is valid before updating state
        if not result.is_valid:
            raise ValueError(
                f"Homography computation failed: {'; '.join(result.validation_messages)}"
            )

        # Update instance state for backward compatibility
        self.K = params.intrinsic_matrix.copy()
        self.w_pos = params.camera_position.copy()
        self.height_m = params.camera_height
        self.pan_deg = float(params.pan_deg)
        self.tilt_deg = float(params.tilt_deg)
        self.roll_deg = float(params.roll_deg)
        self.map_width = int(params.map_width)
        self.map_height = int(params.map_height)
        self.PPM = int(params.pixels_per_meter)

        # Update homography matrices
        self.H = result.homography_matrix.copy()
        self.H_inv = result.inverse_homography_matrix.copy()

        # Update distortion if provided
        if params.distortion is not None and not params.distortion.is_zero():
            self._dist_coeffs = params.distortion.to_array()
            self._use_distortion = True
        else:
            self._dist_coeffs = None
            self._use_distortion = False

        # Update height uncertainty if provided
        if params.height_uncertainty is not None:
            self.height_uncertainty_lower = params.height_uncertainty.lower
            self.height_uncertainty_upper = params.height_uncertainty.upper
        else:
            self.height_uncertainty_lower = None
            self.height_uncertainty_upper = None

        # Update affine matrix if provided
        if params.affine_matrix is not None:
            self.A = params.affine_matrix.copy()
        else:
            self.A = np.eye(3)

        logger.info("Instance state updated from CameraGeometryResult.")

        return result

    @staticmethod
    def _validate_parameters_static(
        K: np.ndarray,
        w_pos: np.ndarray,
        _pan_deg: Degrees,  # Currently unused but kept for API consistency
        tilt_deg: Degrees,
        roll_deg: Degrees = Degrees(0.0),
    ) -> None:
        """Static version of parameter validation for use in compute().

        This is a static method to enable pure function computation without
        requiring an instance.

        Args:
            K: 3x3 intrinsic matrix
            w_pos: Camera position in world coordinates [X, Y, Z] (meters)
            _pan_deg: Pan angle in degrees (currently unused, kept for API consistency)
            tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention)
            roll_deg: Roll angle in degrees (positive = clockwise, default = 0.0)

        Raises:
            ValueError: If any parameter is invalid or out of acceptable range.
        """
        # Use class constants
        cls = CameraGeometry

        # Validate K matrix shape
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got shape {K.shape}")

        # Validate K matrix for NaN/Infinity
        if not np.all(np.isfinite(K)):
            raise ValueError("K matrix contains NaN or Infinity values")

        # Validate w_pos structure
        if len(w_pos) != 3:
            raise ValueError(f"w_pos must have 3 elements [X, Y, Z], got {len(w_pos)}")

        # Validate w_pos for NaN/Infinity
        if not np.all(np.isfinite(w_pos)):
            raise ValueError("w_pos contains NaN or Infinity values")

        # Camera height validation (w_pos[2])
        height = w_pos[2]
        if height < cls.HEIGHT_MIN or height > cls.HEIGHT_MAX:
            raise ValueError(
                f"Camera height {height:.2f}m is out of valid range "
                f"[{cls.HEIGHT_MIN}, {cls.HEIGHT_MAX}]m"
            )

        if height < cls.HEIGHT_WARN_LOW:
            logger.warning(
                f"Camera height {height:.2f}m is very low (<{cls.HEIGHT_WARN_LOW}m). "
                f"Ground projection accuracy may be reduced."
            )
        elif height > cls.HEIGHT_WARN_HIGH:
            logger.warning(
                f"Camera height {height:.2f}m is very high (>{cls.HEIGHT_WARN_HIGH}m). "
                f"Ground projection accuracy may be reduced at extreme distances."
            )

        # Tilt angle validation
        if tilt_deg <= cls.TILT_MIN or tilt_deg > cls.TILT_MAX:
            raise ValueError(
                f"Tilt angle {tilt_deg:.1f}° is out of valid range "
                f"({cls.TILT_MIN}, {cls.TILT_MAX}]°. "
                f"Camera must point downward (positive tilt) for ground plane projection."
            )

        if tilt_deg < cls.TILT_WARN_LOW:
            logger.debug(
                f"Tilt angle {tilt_deg:.1f}° is near horizontal (<{cls.TILT_WARN_LOW}°). "
                f"Ground projection may be unstable or extend to very large distances."
            )
        elif tilt_deg > cls.TILT_WARN_HIGH:
            logger.debug(
                f"Tilt angle {tilt_deg:.1f}° is very steep (>{cls.TILT_WARN_HIGH}°). "
                f"Ground coverage area will be very limited."
            )

        # FOV validation (calculated from K)
        focal_length = K[0, 0]  # Assuming square pixels (fx = fy)
        sensor_width_px = 2.0 * K[0, 2]  # Principal point is at center
        fov_rad = 2.0 * math.atan(sensor_width_px / (2.0 * focal_length))
        fov_deg = math.degrees(fov_rad)

        if fov_deg < cls.FOV_MIN_DEG or fov_deg > cls.FOV_MAX_DEG:
            raise ValueError(
                f"Calculated FOV {fov_deg:.1f}° is out of reasonable range "
                f"[{cls.FOV_MIN_DEG}, {cls.FOV_MAX_DEG}]°. "
                f"Check intrinsic matrix K and zoom factor."
            )

        if fov_deg < cls.FOV_WARN_MIN_DEG or fov_deg > cls.FOV_WARN_MAX_DEG:
            logger.warning(
                f"FOV {fov_deg:.1f}° is unusual. Typical PTZ cameras have FOV between "
                f"{cls.FOV_WARN_MIN_DEG}° and {cls.FOV_WARN_MAX_DEG}°."
            )

        # Roll angle validation
        if abs(roll_deg) > cls.ROLL_ERROR_THRESHOLD:
            raise ValueError(
                f"Roll angle {roll_deg:.1f}° is outside valid range "
                f"[-{cls.ROLL_ERROR_THRESHOLD}°, {cls.ROLL_ERROR_THRESHOLD}°]. "
                f"Check camera mount alignment."
            )

        if abs(roll_deg) > cls.ROLL_WARN_THRESHOLD:
            warnings.warn(
                f"Roll angle {roll_deg:.1f}° is unusually large (>{cls.ROLL_WARN_THRESHOLD}°). "
                f"Typical camera mount roll is ±2°. Verify configuration.",
                UserWarning,
            )

    @staticmethod
    def _get_rotation_matrix_static(
        pan_deg: Degrees, tilt_deg: Degrees, roll_deg: Degrees
    ) -> np.ndarray:
        """Static version of rotation matrix calculation for use in compute().

        Calculates the 3x3 rotation matrix R from world to camera coordinates
        based on pan (Yaw), tilt (Pitch), and roll.

        Args:
            pan_deg: Pan angle in degrees
            tilt_deg: Tilt angle in degrees
            roll_deg: Roll angle in degrees

        Returns:
            R: 3x3 rotation matrix transforming world coordinates to camera frame
        """
        pan_rad = math.radians(pan_deg)
        tilt_rad = math.radians(tilt_deg)
        roll_rad = math.radians(roll_deg)

        # Base transformation from World to Camera when pan=0, tilt=0
        R_base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        # Pan rotation around world Z-axis (yaw)
        Rz_pan = np.array(
            [
                [math.cos(pan_rad), -math.sin(pan_rad), 0],
                [math.sin(pan_rad), math.cos(pan_rad), 0],
                [0, 0, 1],
            ]
        )

        # Roll rotation around camera Z-axis (optical axis)
        Rz_roll = np.array(
            [
                [math.cos(roll_rad), -math.sin(roll_rad), 0],
                [math.sin(roll_rad), math.cos(roll_rad), 0],
                [0, 0, 1],
            ]
        )

        # Tilt rotation around camera X-axis (pitch)
        Rx_tilt = np.array(
            [
                [1, 0, 0],
                [0, math.cos(tilt_rad), -math.sin(tilt_rad)],
                [0, math.sin(tilt_rad), math.cos(tilt_rad)],
            ]
        )

        # Full rotation: R = R_tilt @ R_roll @ R_base @ R_pan
        R: np.ndarray = Rx_tilt @ Rz_roll @ R_base @ Rz_pan
        return R

    @staticmethod
    def _calculate_ground_homography_static(
        K: np.ndarray, w_pos: np.ndarray, R: np.ndarray
    ) -> np.ndarray:
        """Static version of ground homography calculation for use in compute().

        Calculates the Homography matrix H that maps world ground plane (Z=0)
        to image pixels.

        Args:
            K: 3x3 intrinsic matrix
            w_pos: Camera position [X, Y, Z] in world coordinates
            R: 3x3 rotation matrix from world to camera frame

        Returns:
            H: 3x3 normalized homography matrix
        """
        # Translation from camera to world origin: t = -R @ C
        t = -R @ w_pos

        # Build homography: H = K @ [r1, r2, t]
        r1 = R[:, 0]  # Column 0: world X-axis in camera frame
        r2 = R[:, 1]  # Column 1: world Y-axis in camera frame

        # Construct 3x3 extrinsic homography matrix
        H_extrinsic = np.column_stack([r1, r2, t])
        if H_extrinsic.shape != (3, 3):
            raise ValueError(f"H_extrinsic must be 3x3, got {H_extrinsic.shape}")

        H = K @ H_extrinsic
        if H.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got {H.shape}")

        # Normalize so H[2, 2] = 1 for consistent scale
        if abs(H[2, 2]) < 1e-10:
            logger.warning(
                "Homography normalization failed (H[2,2] near zero). Returning identity."
            )
            return np.eye(3)

        H_normalized: np.ndarray = H / H[2, 2]
        return H_normalized

    @staticmethod
    def _compute_center_projection_distance(
        H_inv: np.ndarray,
        w_pos: np.ndarray,
        image_width: int,
        image_height: int,
        validation_messages: list[str],
    ) -> float | None:
        """Compute distance from camera to where image center projects on ground.

        Args:
            H_inv: 3x3 inverse homography matrix
            w_pos: Camera position [X, Y, Z] in world coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            validation_messages: List to append validation messages to

        Returns:
            Distance in meters, or None if projection is invalid
        """
        # Project image center to ground plane
        image_center = np.array([image_width / 2.0, image_height / 2.0, 1.0])
        world_point = H_inv @ image_center

        # Normalize homogeneous coordinates
        if abs(world_point[2]) < 1e-10:
            validation_messages.append(
                "Projection of image center yields invalid homogeneous coordinate (w near 0). "
                "Check tilt angle and camera height."
            )
            return None

        Xw = world_point[0] / world_point[2]
        Yw = world_point[1] / world_point[2]

        # Calculate distance from camera position to projected point
        distance = math.sqrt((Xw - w_pos[0]) ** 2 + (Yw - w_pos[1]) ** 2)

        # Validate distance
        if not math.isfinite(distance):
            validation_messages.append(
                "Projected ground distance is infinite or NaN. "
                "Check tilt angle - camera may be pointing too close to horizontal."
            )
            return None

        if distance < 0:
            validation_messages.append(
                f"Projected ground distance is negative ({distance:.2f}m). "
                f"This should not occur. Check homography calculation."
            )
            return None

        return distance

    def _validate_parameters(
        self,
        K: np.ndarray,
        w_pos: np.ndarray,
        pan_deg: Degrees,
        tilt_deg: Degrees,
        roll_deg: Degrees = Degrees(0.0),
    ) -> None:
        """
        Validates all camera parameters before setting them.

        Args:
            K: 3x3 intrinsic matrix
            w_pos: Camera position in world coordinates [X, Y, Z] (meters)
            pan_deg: Pan angle in degrees (positive = right)
            tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention)
            roll_deg: Roll angle in degrees (positive = clockwise, default = 0.0)

        Raises:
            ValueError: If any parameter is invalid or out of acceptable range.
        """
        # Validate K matrix shape
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got shape {K.shape}")

        # Validate K matrix for NaN/Infinity
        if not np.all(np.isfinite(K)):
            raise ValueError("K matrix contains NaN or Infinity values")

        # Validate w_pos structure
        if len(w_pos) != 3:
            raise ValueError(f"w_pos must have 3 elements [X, Y, Z], got {len(w_pos)}")

        # Validate w_pos for NaN/Infinity
        if not np.all(np.isfinite(w_pos)):
            raise ValueError("w_pos contains NaN or Infinity values")

        # Camera height validation (w_pos[2])
        height = w_pos[2]
        if height < self.HEIGHT_MIN or height > self.HEIGHT_MAX:
            raise ValueError(
                f"Camera height {height:.2f}m is out of valid range "
                f"[{self.HEIGHT_MIN}, {self.HEIGHT_MAX}]m"
            )

        if height < self.HEIGHT_WARN_LOW:
            logger.warning(
                f"Camera height {height:.2f}m is very low (<{self.HEIGHT_WARN_LOW}m). "
                f"Ground projection accuracy may be reduced."
            )
        elif height > self.HEIGHT_WARN_HIGH:
            logger.warning(
                f"Camera height {height:.2f}m is very high (>{self.HEIGHT_WARN_HIGH}m). "
                f"Ground projection accuracy may be reduced at extreme distances."
            )

        # Tilt angle validation
        if tilt_deg <= self.TILT_MIN or tilt_deg > self.TILT_MAX:
            raise ValueError(
                f"Tilt angle {tilt_deg:.1f}° is out of valid range "
                f"({self.TILT_MIN}, {self.TILT_MAX}]°. "
                f"Camera must point downward (positive tilt) for ground plane projection."
            )

        if tilt_deg < self.TILT_WARN_LOW:
            logger.debug(
                f"Tilt angle {tilt_deg:.1f}° is near horizontal (<{self.TILT_WARN_LOW}°). "
                f"Ground projection may be unstable or extend to very large distances."
            )
        elif tilt_deg > self.TILT_WARN_HIGH:
            logger.debug(
                f"Tilt angle {tilt_deg:.1f}° is very steep (>{self.TILT_WARN_HIGH}°). "
                f"Ground coverage area will be very limited."
            )

        # FOV validation (calculated from K)
        focal_length = K[0, 0]  # Assuming square pixels (fx = fy)
        sensor_width_px = 2.0 * K[0, 2]  # Principal point is at center
        fov_rad = 2.0 * math.atan(sensor_width_px / (2.0 * focal_length))
        fov_deg = math.degrees(fov_rad)

        if fov_deg < self.FOV_MIN_DEG or fov_deg > self.FOV_MAX_DEG:
            raise ValueError(
                f"Calculated FOV {fov_deg:.1f}° is out of reasonable range "
                f"[{self.FOV_MIN_DEG}, {self.FOV_MAX_DEG}]°. "
                f"Check intrinsic matrix K and zoom factor."
            )

        if fov_deg < self.FOV_WARN_MIN_DEG or fov_deg > self.FOV_WARN_MAX_DEG:
            logger.warning(
                f"FOV {fov_deg:.1f}° is unusual. Typical PTZ cameras have FOV between "
                f"{self.FOV_WARN_MIN_DEG}° and {self.FOV_WARN_MAX_DEG}°."
            )

        # Roll angle validation
        if abs(roll_deg) > self.ROLL_ERROR_THRESHOLD:
            raise ValueError(
                f"Roll angle {roll_deg:.1f}° is outside valid range "
                f"[-{self.ROLL_ERROR_THRESHOLD}°, {self.ROLL_ERROR_THRESHOLD}°]. "
                f"Check camera mount alignment."
            )

        if abs(roll_deg) > self.ROLL_WARN_THRESHOLD:
            warnings.warn(
                f"Roll angle {roll_deg:.1f}° is unusually large (>{self.ROLL_WARN_THRESHOLD}°). "
                f"Typical camera mount roll is ±2°. Verify configuration.",
                UserWarning,
            )

    def set_camera_parameters(
        self,
        K: np.ndarray,
        w_pos: np.ndarray,
        pan_deg: Degrees,
        tilt_deg: Degrees,
        map_width: Pixels,
        map_height: Pixels,
        roll_deg: Degrees = Degrees(0.0),
    ) -> CameraGeometryResult:
        """
        Sets all required parameters and calculates the Homography matrix H.

        This method maintains full backward compatibility while internally using
        the new immutable CameraParameters/CameraGeometryResult pattern. It:
        1. Creates a CameraParameters instance from the provided arguments
        2. Calls compute_and_update() to compute and update instance state
        3. Returns the CameraGeometryResult (new in this version)

        Args:
            K: 3x3 intrinsic matrix
            w_pos: Camera position in world coordinates [X, Y, Z] (meters)
            pan_deg: Pan angle in degrees (positive = right)
            tilt_deg: Tilt angle in degrees (positive = down, Hikvision convention)
            map_width: Width of output map in pixels
            map_height: Height of output map in pixels
            roll_deg: Roll angle in degrees (positive = clockwise, default = 0.0)
                      Typical camera mount roll is ±2°

        Returns:
            CameraGeometryResult containing the computed homography matrices
            and validation state. This is a new return value for forward
            compatibility; existing code ignoring the return value will
            continue to work unchanged.

        Raises:
            ValueError: If any parameter is invalid or out of acceptable range.

        Example:
            >>> geo = CameraGeometry(1920, 1080)
            >>> K = CameraGeometry.get_intrinsics(1.0)
            >>> w_pos = np.array([0.0, 0.0, 10.0])
            >>> # Backward compatible usage (ignore return value)
            >>> geo.set_camera_parameters(K, w_pos, 0.0, 45.0, 640, 640)
            >>> # New usage (use the returned result)
            >>> result = geo.set_camera_parameters(K, w_pos, 0.0, 45.0, 640, 640)
            >>> if result.is_valid:
            ...     print(f"Condition number: {result.condition_number}")
        """
        # Log parameters for debugging (mirrors original behavior)
        logger.info("Setting camera parameters (via immutable pattern):")
        logger.info(f"  Intrinsic matrix K:\n{K}")

        # Extract current distortion coefficients if set
        distortion: DistortionCoefficients | None = None
        if self._dist_coeffs is not None and not np.allclose(self._dist_coeffs, 0.0):
            distortion = DistortionCoefficients.from_array(self._dist_coeffs)

        # Extract current height uncertainty if set
        height_uncertainty: HeightUncertainty | None = None
        if self.height_uncertainty_lower is not None and self.height_uncertainty_upper is not None:
            height_uncertainty = HeightUncertainty(
                lower=self.height_uncertainty_lower,
                upper=self.height_uncertainty_upper,
            )

        # Extract current affine matrix if not identity
        affine_matrix: np.ndarray | None = None
        if not np.allclose(self.A, np.eye(3)):
            affine_matrix = self.A.copy()

        # Create immutable CameraParameters
        params = CameraParameters.create(
            image_width=Pixels(self.w),
            image_height=Pixels(self.h),
            intrinsic_matrix=K,
            camera_position=w_pos,
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
            roll_deg=roll_deg,
            map_width=map_width,
            map_height=map_height,
            pixels_per_meter=Unitless(float(self.PPM)),
            distortion=distortion,
            height_uncertainty=height_uncertainty,
            affine_matrix=affine_matrix,
        )

        # Use compute_and_update to compute homography and update instance state
        result = self.compute_and_update(params)

        logger.info("Geometry setup complete. Homography matrix H calculated.")

        return result

    def _validate_projection(self) -> None:
        """
        Validates that the homography produces reasonable ground projections.

        Raises:
            ValueError: If projection yields invalid results (negative or infinite distances).
        """
        if self.w_pos is None:
            raise RuntimeError("Camera position not set. Call set_camera_parameters() first.")

        # Project image center to ground plane
        image_center = np.array([self.w / 2.0, self.h / 2.0, 1.0])
        world_point = self.H_inv @ image_center

        # Normalize homogeneous coordinates
        if abs(world_point[2]) < 1e-10:
            raise ValueError(
                "Projection of image center yields invalid homogeneous coordinate (w ≈ 0). "
                "Check tilt angle and camera height."
            )

        Xw = world_point[0] / world_point[2]
        Yw = world_point[1] / world_point[2]

        # Calculate distance from camera position to projected point
        distance = math.sqrt((Xw - self.w_pos[0]) ** 2 + (Yw - self.w_pos[1]) ** 2)

        # Validate distance
        if not math.isfinite(distance):
            raise ValueError(
                "Projected ground distance is infinite or NaN. "
                "Check tilt angle - camera may be pointing too close to horizontal."
            )

        if distance < 0:
            raise ValueError(
                f"Projected ground distance is negative ({distance:.2f}m). "
                f"This should not occur. Check homography calculation."
            )

        # Log the projected distance for information
        logger.info(f"  Image center projects to ground at distance: {distance:.2f}m")

        # Warn if distance seems unreasonable
        max_reasonable_distance = self.height_m * self.MAX_DISTANCE_HEIGHT_RATIO
        if distance > max_reasonable_distance:
            logger.warning(
                f"Projected ground distance {distance:.2f}m is very large (>{max_reasonable_distance:.2f}m). "
                f"This may indicate a near-horizontal tilt angle."
            )

    # --- Core Geometry Methods (Unchanged from previous output) ---

    def _get_rotation_matrix(self, roll_deg: Degrees | None = None) -> np.ndarray:
        """
        Calculates the 3x3 rotation matrix R from world to camera coordinates
        based on pan (Yaw), tilt (Pitch), and roll.

        Rotation order: R = R_tilt @ R_roll @ R_base @ R_pan

        Coordinate System Convention:
        - World: X=East, Y=North, Z=Up
        - Camera: X=Right, Y=Down, Z=Forward (optical axis)

        The transformation consists of:
        1. Pan rotation: Rotate around world Z-axis (yaw)
        2. Base transform: Convert world coords to camera coords when pan=0, tilt=0
           (camera looking North, horizontal)
        3. Roll rotation: Rotate around camera Z-axis (optical axis)
        4. Tilt rotation: Rotate around camera X-axis (pitch)

        Tilt Convention (Hikvision):
        - Positive tilt_deg = camera pointing downward
        - Negative tilt_deg = camera pointing upward

        Roll Convention:
        - Positive roll_deg = clockwise rotation when looking from behind camera
          (along +Z axis, into the scene)
        - Roll is applied in camera frame after base transformation but before tilt

        Args:
            roll_deg: Roll angle in degrees. If None, uses self.roll_deg (default 0.0)

        Returns:
            R: 3x3 rotation matrix transforming world coordinates to camera frame
        """
        pan_rad = math.radians(self.pan_deg)
        tilt_rad = math.radians(self.tilt_deg)

        # Use provided roll_deg or fall back to instance attribute
        roll = roll_deg if roll_deg is not None else self.roll_deg
        roll_rad = math.radians(roll)

        # Base transformation from World to Camera when pan=0, tilt=0
        # (camera looking North, horizontal):
        # - World X (East)  -> Camera X (Right)
        # - World Y (North) -> Camera Z (Forward)
        # - World Z (Up)    -> Camera -Y (camera Y points Down)
        #
        # This matrix transforms [Xw, Yw, Zw] to [Xc, Yc, Zc]:
        #   Xc = Xw
        #   Yc = -Zw
        #   Zc = Yw
        R_base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        # Pan rotation around world Z-axis (yaw)
        # Positive pan = clockwise from above = camera looks right (towards East when starting North)
        # This is applied in world coordinates before the base transform
        Rz_pan = np.array(
            [
                [math.cos(pan_rad), -math.sin(pan_rad), 0],
                [math.sin(pan_rad), math.cos(pan_rad), 0],
                [0, 0, 1],
            ]
        )

        # Roll rotation around camera Z-axis (optical axis)
        # Positive roll = clockwise when looking from behind camera (along +Z axis)
        # This rotates the image plane around the optical axis
        Rz_roll = np.array(
            [
                [math.cos(roll_rad), -math.sin(roll_rad), 0],
                [math.sin(roll_rad), math.cos(roll_rad), 0],
                [0, 0, 1],
            ]
        )

        # Tilt rotation around camera X-axis (pitch)
        # Positive tilt = camera looks down
        # This is applied in camera coordinates after pan, base transform, and roll
        Rx_tilt = np.array(
            [
                [1, 0, 0],
                [0, math.cos(tilt_rad), -math.sin(tilt_rad)],
                [0, math.sin(tilt_rad), math.cos(tilt_rad)],
            ]
        )

        # Full rotation: first pan in world, then base transform, then roll in camera, then tilt in camera
        # R_world_to_cam = R_tilt @ R_roll @ R_base @ R_pan
        R: np.ndarray = Rx_tilt @ Rz_roll @ R_base @ Rz_pan
        return R

    def _calculate_ground_homography(self) -> np.ndarray:
        """
        Calculates the Homography matrix H that maps world ground plane (Z=0) to image pixels.

        Coordinate Frame Conventions:
            World Frame:
                - Origin: Arbitrary reference point (camera position or scene center)
                - Axes: X=East (meters), Y=North (meters), Z=Up (meters)
                - Ground plane: Z=0
                - Right-handed coordinate system

            Camera Frame:
                - Origin: Camera optical center
                - Axes: X=Right (meters), Y=Down (meters), Z=Forward (meters, optical axis)
                - Transformation: P_camera = R @ P_world + t, where t = -R @ C
                - Right-handed coordinate system (standard computer vision)

            Image Frame:
                - Origin: Top-left corner of image
                - Axes: u=horizontal (pixels, right), v=vertical (pixels, down)
                - Projection: p_image = K @ P_camera

        Mathematical Derivation (Planar Homography for Z=0 Ground Plane):
            Step 1 - General 3D projection:
                P_camera = R @ P_world + t
                p_image ~ K @ P_camera = K @ (R @ P_world + t)

            Step 2 - For ground plane Z=0, P_world = [X, Y, 0]^T:
                Rotation matrix R = [r1, r2, r3] (columns)
                R @ [X, Y, 0]^T = r1*X + r2*Y + r3*0 = r1*X + r2*Y

            Step 3 - Substitute into projection:
                p_image ~ K @ (r1*X + r2*Y + t)
                        = K @ [r1, r2, t] @ [X, Y, 1]^T

            Step 4 - Define homography:
                H = K @ [r1, r2, t]
                This is a 3x3 matrix (NOT 3x4 projection matrix)
                Maps 2D ground plane [X, Y, 1] to 2D image [u, v, 1]

            Normalization: H is normalized so H[2,2] = 1 for numerical stability.

        Returns:
            H (np.ndarray): 3x3 homography matrix mapping [X_world, Y_world, 1] -> [u, v, 1]
        """

        if self.K is None:
            logger.warning("K not set. Returning Identity Homography.")
            return np.eye(3)

        R = self._get_rotation_matrix()

        # Camera position C in world coordinates
        C = self.w_pos  # [Xw, Yw, Zw]

        # Translation from camera to world origin: t = -R @ C
        # This gives the position of the world origin in camera frame
        t = -R @ C

        # Build homography: H = K @ [r1, r2, t]
        # r1, r2 are the first two COLUMNS of R (corresponding to world X and Y axes)
        # This is correct because for Z=0: R @ [X, Y, 0]^T = r1*X + r2*Y
        r1 = R[:, 0]  # Column 0: world X-axis in camera frame
        r2 = R[:, 1]  # Column 1: world Y-axis in camera frame

        # Construct 3x3 extrinsic homography matrix (NOT 3x4 projection matrix)
        H_extrinsic = np.column_stack([r1, r2, t])
        if H_extrinsic.shape != (3, 3):
            raise ValueError(f"H_extrinsic must be 3x3, got {H_extrinsic.shape}")

        H = self.K @ H_extrinsic
        if H.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got {H.shape}")

        # Normalize so H[2, 2] = 1 for consistent scale
        if abs(H[2, 2]) < 1e-10:
            logger.warning(
                "Homography normalization failed (H[2,2] near zero). Returning identity."
            )
            return np.eye(3)

        H_normalized: np.ndarray = H / H[2, 2]

        return H_normalized

    def project_image_to_map(
        self, pts: list[tuple[int, int]], sw: int, sh: int
    ) -> list[tuple[int, int]]:
        """
        Projects image coordinates (pixels) to the world ground plane (meters).

        If distortion coefficients are set, points are undistorted before projection.
        """
        if self.H_inv is None or np.all(self.H_inv == np.eye(3)):
            return [(int(x / 2), int(y / 2)) for x, y in pts]

        # Undistort points if distortion is enabled
        pts_float: list[tuple[float, float]] = [(float(x), float(y)) for x, y in pts]
        if self._use_distortion:
            pts_float = self.undistort_points(pts_float)

        pts_homogeneous = np.array(pts_float, dtype=np.float64).T
        pts_homogeneous = np.vstack([pts_homogeneous, np.ones(pts_homogeneous.shape[1])])

        # Project from Image to World Ground Plane (Xw, Yw, 1)
        pts_world_homogeneous = self.H_inv @ pts_homogeneous

        # Normalize world coordinates (convert [Xw, Yw, W] to [Xw/W, Yw/W, 1])
        Xw = pts_world_homogeneous[0, :] / pts_world_homogeneous[2, :]
        Yw = pts_world_homogeneous[1, :] / pts_world_homogeneous[2, :]

        # --- Mapping Policy (uses self.PPM pixels per meter) ---
        map_center_x = sw // 2
        map_bottom_y = sh

        pts_map_x = (Xw * self.PPM) + map_center_x
        pts_map_y = map_bottom_y - (Yw * self.PPM)

        pts_map = [(int(x), int(y)) for x, y in zip(pts_map_x, pts_map_y)]

        return pts_map

    def world_to_map(
        self, Xw: Meters, Yw: Meters, sw: Pixels | None = None, sh: Pixels | None = None
    ) -> tuple[Pixels, Pixels]:
        """
        Map a world coordinate (meters) to side-panel pixel coordinates.

        Args:
            Xw, Yw: world coordinates in meters
            sw, sh: side-panel width/height in pixels. If omitted, uses self.map_width/self.map_height.

        Returns:
            (x_px, y_px) tuple in pixels relative to the top-left of the side panel image.
        """
        width: int = sw if sw is not None else self.map_width
        height: int = sh if sh is not None else self.map_height

        map_center_x = width // 2
        map_bottom_y = height

        x_px = Pixels(int((Xw * self.PPM) + map_center_x))
        y_px = Pixels(int(map_bottom_y - (Yw * self.PPM)))

        return x_px, y_px

    def set_height_uncertainty(self, confidence_interval: tuple[Meters, Meters]) -> None:
        """
        Set height uncertainty for propagation to future projections.

        This stores the confidence interval bounds that will be used by
        project_to_world_with_uncertainty() to compute uncertainty in
        projected world coordinates.

        Args:
            confidence_interval: Tuple of (lower_bound, upper_bound) for camera
                                height in meters. Typically a 95% confidence interval
                                from height calibration.

        Raises:
            ValueError: If confidence_interval is not a 2-tuple or bounds are invalid

        Example:
            >>> # After height calibration
            >>> result = calibrator.optimize_height_least_squares()
            >>> geo.set_height_uncertainty(result.confidence_interval)
        """
        if not isinstance(confidence_interval, tuple) or len(confidence_interval) != 2:
            raise ValueError(
                f"confidence_interval must be a 2-tuple, got {type(confidence_interval).__name__}"
            )

        lower, upper = confidence_interval

        if lower <= 0 or upper <= 0:
            raise ValueError(f"Height bounds must be positive, got lower={lower}, upper={upper}")

        if lower > upper:
            raise ValueError(f"Lower bound ({lower}) cannot exceed upper bound ({upper})")

        self.height_uncertainty_lower = lower
        self.height_uncertainty_upper = upper

    def project_to_world_with_uncertainty(self, pixel_x: float, pixel_y: float) -> dict[str, Any]:
        """
        Project pixel to world coordinates with uncertainty propagation.

        This method projects an image pixel to world coordinates and computes
        uncertainty bounds based on the height uncertainty set via
        set_height_uncertainty(). The uncertainty is computed by projecting
        at the lower and upper height bounds and measuring the variation in
        the resulting world coordinates.

        Physical principle:
            When camera height changes, world coordinates scale proportionally:
            - new_world_x = original_world_x * (new_height / original_height)
            - new_world_y = original_world_y * (new_height / original_height)

        Args:
            pixel_x: Horizontal pixel coordinate in image
            pixel_y: Vertical pixel coordinate in image

        Returns:
            Dictionary with the following keys:
                - world_x (float): Mean X coordinate in meters (East-West)
                - world_y (float): Mean Y coordinate in meters (North-South)
                - distance (float): Mean distance from camera in meters
                - x_uncertainty (float): Half-width of X uncertainty (± value in meters)
                - y_uncertainty (float): Half-width of Y uncertainty (± value in meters)
                - distance_uncertainty (float): Half-width of distance uncertainty (± value in meters)
                - confidence_level (float): The confidence level (0.95)

        Raises:
            ValueError: If pixel is near horizon and cannot be projected

        Note:
            - If set_height_uncertainty() has not been called, all uncertainty
              values will be 0.0
            - The mean coordinates are computed from the current height
            - Uncertainty represents the range due to height uncertainty only
            - Does not account for other error sources (camera calibration,
              lens distortion, etc.)

        Example:
            >>> geo.set_height_uncertainty((4.8, 5.2))
            >>> result = geo.project_to_world_with_uncertainty(1280, 720)
            >>> print(f"Position: ({result['world_x']:.2f} ± {result['x_uncertainty']:.2f}, "
            ...       f"{result['world_y']:.2f} ± {result['y_uncertainty']:.2f}) meters")
        """
        # Project at current height to get mean coordinates
        pt_img = np.array([[pixel_x], [pixel_y], [1.0]])
        pt_world = self.H_inv @ pt_img

        # Check for valid projection (not near horizon)
        if abs(pt_world[2, 0]) < 1e-6:
            raise ValueError(
                f"Invalid point at pixel ({pixel_x}, {pixel_y}): "
                "Point is too close to horizon and cannot be projected"
            )

        # Normalize to get world coordinates at current height
        world_x_mean = pt_world[0, 0] / pt_world[2, 0]
        world_y_mean = pt_world[1, 0] / pt_world[2, 0]
        distance_mean = np.sqrt(world_x_mean**2 + world_y_mean**2)

        # Initialize uncertainty to zero
        x_uncertainty = 0.0
        y_uncertainty = 0.0
        distance_uncertainty = 0.0

        # If height uncertainty has been set, compute uncertainty bounds
        if self.height_uncertainty_lower is not None and self.height_uncertainty_upper is not None:
            # World coordinates scale proportionally with height
            # new_coord = original_coord * (new_height / original_height)

            # Project at lower height bound
            height_ratio_lower = self.height_uncertainty_lower / self.height_m
            world_x_lower = world_x_mean * height_ratio_lower
            world_y_lower = world_y_mean * height_ratio_lower
            distance_lower = distance_mean * height_ratio_lower

            # Project at upper height bound
            height_ratio_upper = self.height_uncertainty_upper / self.height_m
            world_x_upper = world_x_mean * height_ratio_upper
            world_y_upper = world_y_mean * height_ratio_upper
            distance_upper = distance_mean * height_ratio_upper

            # Compute half-width of uncertainty range (± value)
            x_uncertainty = abs(world_x_upper - world_x_lower) / 2.0
            y_uncertainty = abs(world_y_upper - world_y_lower) / 2.0
            distance_uncertainty = abs(distance_upper - distance_lower) / 2.0

        return {
            "world_x": world_x_mean,
            "world_y": world_y_mean,
            "distance": distance_mean,
            "x_uncertainty": x_uncertainty,
            "y_uncertainty": y_uncertainty,
            "distance_uncertainty": distance_uncertainty,
            "confidence_level": 0.95,
        }
