#!/usr/bin/env python3
"""
Web-based GCP Capture Tool - Interactive Ground Control Point collection.

This utility provides a browser-based interface for capturing Ground Control Points
(GCPs) from a camera frame with:
- High-precision zoom and pan capabilities
- Real-time distribution quality feedback
- Quadrant coverage visualization
- Interactive GPS coordinate entry
- Drag-to-reposition existing GCPs for fine-tuning
- Load existing YAML configs and continue adding points

Usage:
    # Capture from live camera
    python tools/capture_gcps_web.py Valte

    # Use existing frame image
    python tools/capture_gcps_web.py --frame path/to/frame.jpg

    # Load existing YAML and continue editing
    python tools/capture_gcps_web.py --frame path/to/frame.jpg --load existing_gcps.yaml

    # Specify output file
    python tools/capture_gcps_web.py Valte --output my_gcps.yaml
"""

import argparse
import base64
import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import yaml

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Camera config is optional
try:
    from poc_homography.camera_config import (
        get_camera_by_name, get_rtsp_url, USERNAME, PASSWORD, CAMERAS
    )
    CAMERA_CONFIG_AVAILABLE = True
except (ValueError, ImportError) as e:
    CAMERA_CONFIG_AVAILABLE = False
    print(f"Note: Camera config not available ({e})")
    print("  Use --frame to load an existing image file.\n")

# Import the intrinsics utility if available
try:
    from tools.get_camera_intrinsics import get_ptz_status, compute_intrinsics
    INTRINSICS_AVAILABLE = True
except ImportError:
    INTRINSICS_AVAILABLE = False

# Default camera parameters
DEFAULT_SENSOR_WIDTH_MM = 7.18
DEFAULT_BASE_FOCAL_LENGTH_MM = 5.9

# Reprojection error thresholds for feedback
REPROJ_ERROR_GOOD = 5.0  # pixels - considered good fit
REPROJ_ERROR_WARNING = 10.0  # pixels - warning threshold
REPROJ_ERROR_BAD = 20.0  # pixels - likely outlier

# Outlier auto-remove threshold
OUTLIER_AUTO_REMOVE_THRESHOLD = 15.0  # pixels

# Import for reprojection error calculation
try:
    from poc_homography.coordinate_converter import gps_to_local_xy
    COORDINATE_CONVERTER_AVAILABLE = True
except ImportError:
    COORDINATE_CONVERTER_AVAILABLE = False

# Import for GPS precision analysis and duplicate detection
try:
    from poc_homography.gcp_validation import analyze_gps_precision, detect_duplicate_gcps
    GCP_VALIDATION_AVAILABLE = True
except ImportError:
    GCP_VALIDATION_AVAILABLE = False


class GCPCaptureWebSession:
    """Web-based GCP capture session with distribution feedback."""

    # Distribution thresholds (matching feature_match_homography.py)
    MIN_COVERAGE_RATIO = 0.15
    GOOD_COVERAGE_RATIO = 0.35
    MIN_QUADRANT_COVERAGE = 2
    GOOD_QUADRANT_COVERAGE = 3

    def __init__(
        self,
        frame: np.ndarray,
        camera_name: str = None,
        ptz_status: dict = None,
        sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
        base_focal_length_mm: float = DEFAULT_BASE_FOCAL_LENGTH_MM
    ):
        self.frame = frame
        self.camera_name = camera_name or "Unknown"
        self.ptz_status = ptz_status
        self.sensor_width_mm = sensor_width_mm
        self.base_focal_length_mm = base_focal_length_mm

        self.frame_height, self.frame_width = frame.shape[:2]
        self.gcps: List[Dict] = []
        self.capture_timestamp = datetime.now().isoformat()
        # Coordinate system: 'image_v' (V=0 at top) or None (legacy leaflet_y format)
        self.coordinate_system = 'image_v'

        # Homography and reprojection error state
        self.current_homography = None
        self.reference_lat = None
        self.reference_lon = None
        self.last_reproj_errors = []  # Per-GCP reprojection errors
        self.inlier_mask = None  # RANSAC inlier mask

        # Calculate intrinsics if possible
        self.intrinsics = None
        if INTRINSICS_AVAILABLE and ptz_status:
            self.intrinsics = compute_intrinsics(
                zoom=ptz_status.get('zoom', 1.0),
                image_width=self.frame_width,
                image_height=self.frame_height,
                sensor_width_mm=sensor_width_mm,
                base_focal_length_mm=base_focal_length_mm,
            )

    def calculate_distribution(self) -> Dict:
        """Calculate distribution metrics for current GCPs."""
        n_points = len(self.gcps)

        if n_points < 3:
            return {
                'coverage_ratio': 0.0,
                'quadrants_covered': 0,
                'quadrants': [False, False, False, False],
                'spread_x': 0.0,
                'spread_y': 0.0,
                'distribution_score': 0.0,
                'quality': 'Insufficient',
                'warnings': ['Need at least 3 GCPs for distribution analysis']
            }

        # Extract image points
        image_points = np.array([[gcp['image']['u'], gcp['image']['v']] for gcp in self.gcps])

        # Calculate convex hull coverage
        try:
            hull = cv2.convexHull(image_points.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            image_area = self.frame_width * self.frame_height
            coverage_ratio = hull_area / image_area if image_area > 0 else 0.0
        except Exception:
            coverage_ratio = 0.0

        # Calculate quadrant coverage
        center_u = self.frame_width / 2.0
        center_v = self.frame_height / 2.0
        quadrants = [False, False, False, False]  # TL, TR, BL, BR
        for u, v in image_points:
            q = 0
            if u >= center_u:
                q += 1
            if v >= center_v:
                q += 2
            quadrants[q] = True
        quadrants_covered = sum(quadrants)

        # Calculate spread
        spread_x = np.std(image_points[:, 0]) / self.frame_width if self.frame_width > 0 else 0.0
        spread_y = np.std(image_points[:, 1]) / self.frame_height if self.frame_height > 0 else 0.0

        # Generate warnings
        warnings = []
        if coverage_ratio < self.MIN_COVERAGE_RATIO:
            warnings.append(
                f'GCPs are clustered (coverage {coverage_ratio:.1%} < {self.MIN_COVERAGE_RATIO:.0%}). '
                'Add GCPs in different areas.'
            )
        if quadrants_covered < self.MIN_QUADRANT_COVERAGE:
            warnings.append(
                f'GCPs only cover {quadrants_covered}/4 quadrants. '
                'Add GCPs to cover more of the image.'
            )
        if spread_x < 0.15 or spread_y < 0.15:
            warnings.append('GCPs have low spatial variance. Spread points across the image.')

        # Check for GPS precision issues and duplicates
        if GCP_VALIDATION_AVAILABLE and len(self.gcps) >= 2:
            # Analyze GPS precision
            try:
                precision_result = analyze_gps_precision(self.gcps)
                if precision_result.get('warnings'):
                    warnings.extend(precision_result['warnings'])
            except Exception as e:
                pass  # Don't fail if precision analysis fails

            # Check for duplicate GCPs
            try:
                detect_duplicate_gcps(self.gcps)
            except ValueError as e:
                # ValueError is raised when duplicates are detected
                warnings.append(str(e))

        # Calculate overall score
        coverage_score = min(1.0, coverage_ratio / self.GOOD_COVERAGE_RATIO)
        quadrant_score = quadrants_covered / 4.0
        spread_score = min(1.0, (spread_x + spread_y) / 0.5)
        distribution_score = 0.4 * coverage_score + 0.3 * quadrant_score + 0.3 * spread_score

        # Quality label
        if distribution_score > 0.7:
            quality = 'Good'
        elif distribution_score > 0.5:
            quality = 'Fair'
        else:
            quality = 'Poor'

        return {
            'coverage_ratio': coverage_ratio,
            'quadrants_covered': quadrants_covered,
            'quadrants': quadrants,
            'spread_x': spread_x,
            'spread_y': spread_y,
            'distribution_score': distribution_score,
            'quality': quality,
            'warnings': warnings
        }

    def update_homography(self) -> Dict:
        """
        Compute homography from current GCPs and calculate per-GCP reprojection errors.

        Returns:
            Dictionary with homography quality metrics and per-GCP errors.
        """
        if not COORDINATE_CONVERTER_AVAILABLE:
            return {
                'available': False,
                'message': 'Coordinate converter not available'
            }

        if len(self.gcps) < 4:
            self.current_homography = None
            self.last_reproj_errors = []
            self.inlier_mask = None
            return {
                'available': True,
                'num_gcps': len(self.gcps),
                'errors': [],
                'message': 'Need at least 4 GCPs for homography'
            }

        # Set reference point from camera GPS or GCP centroid
        if self.reference_lat is None:
            if self.ptz_status:
                # Try to get camera GPS from config
                try:
                    cam_info = get_camera_by_name(self.camera_name)
                    if cam_info and 'gps' in cam_info:
                        self.reference_lat = cam_info['gps'].get('latitude')
                        self.reference_lon = cam_info['gps'].get('longitude')
                except Exception:
                    pass

            # Fall back to GCP centroid
            if self.reference_lat is None:
                lats = [g['gps']['latitude'] for g in self.gcps]
                lons = [g['gps']['longitude'] for g in self.gcps]
                self.reference_lat = sum(lats) / len(lats)
                self.reference_lon = sum(lons) / len(lons)

        # Extract points
        image_points = []
        local_points = []

        for gcp in self.gcps:
            u, v = gcp['image']['u'], gcp['image']['v']
            lat, lon = gcp['gps']['latitude'], gcp['gps']['longitude']

            image_points.append([u, v])
            x, y = gps_to_local_xy(self.reference_lat, self.reference_lon, lat, lon)
            local_points.append([x, y])

        image_points = np.array(image_points, dtype=np.float32)
        local_points = np.array(local_points, dtype=np.float32)

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(local_points, image_points, cv2.RANSAC, 5.0)

        if H is None:
            self.current_homography = None
            self.last_reproj_errors = []
            self.inlier_mask = None
            return {
                'available': True,
                'num_gcps': len(self.gcps),
                'errors': [],
                'message': 'Failed to compute homography'
            }

        self.current_homography = H
        self.inlier_mask = mask.ravel().tolist() if mask is not None else [True] * len(self.gcps)

        # Calculate reprojection errors for ALL points
        projected = cv2.perspectiveTransform(
            local_points.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        errors = np.linalg.norm(projected - image_points, axis=1)
        self.last_reproj_errors = errors.tolist()

        # Calculate metrics
        num_inliers = int(np.sum(mask)) if mask is not None else len(self.gcps)
        inlier_errors = errors[mask.ravel() == 1] if mask is not None else errors

        # Build per-GCP error info
        gcp_errors = []
        for i, (gcp, error, is_inlier) in enumerate(zip(self.gcps, errors, self.inlier_mask)):
            status = 'good' if error < REPROJ_ERROR_GOOD else 'warning' if error < REPROJ_ERROR_WARNING else 'bad'
            gcp_errors.append({
                'index': i,
                'description': gcp.get('metadata', {}).get('description', f'GCP {i+1}'),
                'error_px': float(error),
                'is_inlier': bool(is_inlier),
                'status': status
            })

        # Sort by error (highest first) for outlier identification
        sorted_by_error = sorted(gcp_errors, key=lambda x: x['error_px'], reverse=True)

        return {
            'available': True,
            'num_gcps': len(self.gcps),
            'num_inliers': num_inliers,
            'inlier_ratio': num_inliers / len(self.gcps),
            'mean_error_px': float(np.mean(errors)),
            'max_error_px': float(np.max(errors)),
            'inlier_mean_error_px': float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else 0,
            'errors': gcp_errors,
            'outliers': [e for e in sorted_by_error if not e['is_inlier']],
            'worst_gcps': sorted_by_error[:3],
            'thresholds': {
                'good': REPROJ_ERROR_GOOD,
                'warning': REPROJ_ERROR_WARNING,
                'bad': REPROJ_ERROR_BAD,
                'auto_remove': OUTLIER_AUTO_REMOVE_THRESHOLD
            }
        }

    def get_outliers(self, threshold: float = None) -> List[int]:
        """
        Get indices of GCPs with reprojection error above threshold.

        Args:
            threshold: Error threshold in pixels. Defaults to OUTLIER_AUTO_REMOVE_THRESHOLD.

        Returns:
            List of GCP indices that are outliers.
        """
        if threshold is None:
            threshold = OUTLIER_AUTO_REMOVE_THRESHOLD

        if not self.last_reproj_errors:
            self.update_homography()

        if not self.last_reproj_errors:
            return []

        outliers = []
        for i, error in enumerate(self.last_reproj_errors):
            if error > threshold:
                outliers.append(i)

        return outliers

    def remove_outliers(self, threshold: float = None) -> Dict:
        """
        Remove all GCPs with reprojection error above threshold.

        Args:
            threshold: Error threshold in pixels. Defaults to OUTLIER_AUTO_REMOVE_THRESHOLD.

        Returns:
            Dictionary with removal results.
        """
        outlier_indices = self.get_outliers(threshold)

        if not outlier_indices:
            return {
                'removed_count': 0,
                'removed_indices': [],
                'remaining_gcps': len(self.gcps)
            }

        # Remove in reverse order to preserve indices
        removed_descriptions = []
        for i in sorted(outlier_indices, reverse=True):
            if i < len(self.gcps):
                desc = self.gcps[i].get('metadata', {}).get('description', f'GCP {i+1}')
                removed_descriptions.append(desc)
                self.gcps.pop(i)

        # Recalculate homography
        self.update_homography()

        return {
            'removed_count': len(outlier_indices),
            'removed_indices': outlier_indices,
            'removed_descriptions': removed_descriptions,
            'remaining_gcps': len(self.gcps)
        }

    def predict_new_gcp_error(self, u: float, v: float, lat: float, lon: float) -> Dict:
        """
        Predict reprojection error for a potential new GCP before adding it.

        This helps users identify if a GCP would be an outlier before committing.

        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            lat: GPS latitude
            lon: GPS longitude

        Returns:
            Dictionary with prediction results:
                - available: Whether prediction is available
                - predicted_error_px: Predicted reprojection error
                - status: 'good', 'warning', or 'bad'
                - message: Human-readable assessment
        """
        if not COORDINATE_CONVERTER_AVAILABLE:
            return {
                'available': False,
                'message': 'Coordinate converter not available'
            }

        if self.current_homography is None:
            return {
                'available': False,
                'message': 'Need at least 4 GCPs to predict error'
            }

        if self.reference_lat is None or self.reference_lon is None:
            return {
                'available': False,
                'message': 'Reference point not set'
            }

        try:
            # Convert GPS to local coordinates
            x, y = gps_to_local_xy(self.reference_lat, self.reference_lon, lat, lon)

            # Project through current homography
            local_pt = np.array([[x, y]], dtype=np.float32)
            projected = cv2.perspectiveTransform(
                local_pt.reshape(-1, 1, 2), self.current_homography
            ).reshape(2)

            # Calculate error
            error = float(np.sqrt((projected[0] - u)**2 + (projected[1] - v)**2))

            # Determine status
            if error < REPROJ_ERROR_GOOD:
                status = 'good'
                message = f'Good fit ({error:.1f}px)'
            elif error < REPROJ_ERROR_WARNING:
                status = 'warning'
                message = f'Moderate error ({error:.1f}px) - consider verifying coordinates'
            else:
                status = 'bad'
                message = f'High error ({error:.1f}px) - likely outlier, verify GPS and pixel position'

            return {
                'available': True,
                'predicted_error_px': error,
                'status': status,
                'message': message,
                'thresholds': {
                    'good': REPROJ_ERROR_GOOD,
                    'warning': REPROJ_ERROR_WARNING,
                    'bad': REPROJ_ERROR_BAD
                }
            }

        except Exception as e:
            return {
                'available': False,
                'message': f'Prediction failed: {str(e)}'
            }

    def add_gcp(self, u: float, v: float, lat: float, lon: float, description: str = "", accuracy: str = "medium") -> Dict:
        """Add a new GCP."""
        gcp = {
            'gps': {
                'latitude': lat,
                'longitude': lon,
            },
            'image': {
                'u': u,
                'v': v,
            },
            'metadata': {
                'description': description or f"GCP {len(self.gcps) + 1}",
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat(),
            }
        }
        self.gcps.append(gcp)
        return gcp

    def remove_gcp(self, index: int) -> bool:
        """Remove a GCP by index."""
        if 0 <= index < len(self.gcps):
            self.gcps.pop(index)
            return True
        return False

    def update_gcp_position(self, index: int, u: float, v: float) -> bool:
        """Update the pixel position of a GCP (for drag-to-reposition)."""
        if 0 <= index < len(self.gcps):
            self.gcps[index]['image']['u'] = u
            self.gcps[index]['image']['v'] = v
            return True
        return False

    def load_from_yaml(self, yaml_content: str) -> dict:
        """
        Load GCPs from YAML content.

        Returns dict with 'gcps_loaded' count, 'warnings', 'loaded_ptz' position,
        and 'coordinate_system' indicating the V coordinate format.
        """
        warnings = []
        loaded_ptz = None
        loaded_camera_name = None
        coordinate_system = None  # 'image_v' or None (legacy leaflet_y format)

        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            return {'gcps_loaded': 0, 'warnings': [f'YAML parse error: {e}'], 'loaded_ptz': None}

        # Navigate to GCPs
        gcps_data = None
        if 'homography' in data:
            if 'feature_match' in data['homography']:
                fm = data['homography']['feature_match']
                gcps_data = fm.get('ground_control_points', [])

                # Also extract camera context if present
                if 'camera_capture_context' in fm:
                    ctx = fm['camera_capture_context']
                    if ctx.get('camera_name'):
                        loaded_camera_name = ctx['camera_name']
                        self.camera_name = loaded_camera_name
                    if 'ptz_position' in ctx:
                        loaded_ptz = ctx['ptz_position']
                        self.ptz_status = loaded_ptz
                    if ctx.get('capture_timestamp'):
                        self.capture_timestamp = ctx['capture_timestamp']
                    # Check coordinate system - old files won't have this field
                    coordinate_system = ctx.get('coordinate_system')

        if not gcps_data:
            return {'gcps_loaded': 0, 'warnings': ['No GCPs found in YAML'], 'loaded_ptz': None}

        # Set session coordinate system based on loaded data
        self.coordinate_system = coordinate_system  # None for legacy, 'image_v' for new

        # Warn about legacy format
        if coordinate_system is None:
            warnings.append(
                'Legacy format detected (no coordinate_system flag). '
                'V coordinates are treated as Leaflet Y values. '
                'Save will convert to standard image_v format.'
            )

        # Load GCPs
        loaded_count = 0
        for gcp in gcps_data:
            try:
                lat = gcp['gps']['latitude']
                lon = gcp['gps']['longitude']
                u = gcp['image']['u']
                v = gcp['image']['v']
                desc = gcp.get('metadata', {}).get('description', f'GCP {len(self.gcps) + 1}')
                accuracy = gcp.get('metadata', {}).get('accuracy', 'medium')

                self.gcps.append({
                    'gps': {'latitude': lat, 'longitude': lon},
                    'image': {'u': u, 'v': v},
                    'metadata': {
                        'description': desc,
                        'accuracy': accuracy,
                        'timestamp': gcp.get('metadata', {}).get('timestamp', datetime.now().isoformat())
                    }
                })
                loaded_count += 1
            except (KeyError, TypeError) as e:
                warnings.append(f'Skipped invalid GCP: {e}')

        return {
            'gcps_loaded': loaded_count,
            'warnings': warnings,
            'loaded_ptz': loaded_ptz,
            'loaded_camera_name': loaded_camera_name,
            'coordinate_system': coordinate_system
        }

    def move_camera_to_ptz(self, ptz_position: dict, wait_time: float = 3.0) -> dict:
        """
        Move camera to specified PTZ position.

        Args:
            ptz_position: Dict with 'pan', 'tilt', 'zoom' keys
            wait_time: Seconds to wait for camera to reach position

        Returns:
            Dict with 'success' bool and 'message' string
        """
        import time

        if not CAMERA_CONFIG_AVAILABLE:
            return {'success': False, 'message': 'Camera config not available'}

        cam_info = get_camera_by_name(self.camera_name)
        if not cam_info:
            return {'success': False, 'message': f"Camera '{self.camera_name}' not found in config"}

        try:
            from ptz_discovery_and_control.hikvision.hikvision_ptz_discovery import HikvisionPTZ

            ptz = HikvisionPTZ(cam_info['ip'], USERNAME, PASSWORD)

            # Move to absolute position
            ptz.send_ptz_return({
                'pan': ptz_position.get('pan', 0),
                'tilt': ptz_position.get('tilt', 0),
                'zoom': ptz_position.get('zoom', 1.0)
            })

            # Wait for camera to reach position
            time.sleep(wait_time)

            # Update session PTZ status
            self.ptz_status = ptz_position

            return {
                'success': True,
                'message': f"Camera moved to P={ptz_position.get('pan', 0):.1f} T={ptz_position.get('tilt', 0):.1f} Z={ptz_position.get('zoom', 1):.1f}x"
            }

        except ImportError as e:
            return {'success': False, 'message': f'PTZ control not available: {e}'}
        except Exception as e:
            return {'success': False, 'message': f'Failed to move camera: {e}'}

    def generate_yaml(self) -> str:
        """Generate YAML config content."""
        lines = [
            "# GCP Configuration",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Camera: {self.camera_name}",
            f"# GCPs captured: {len(self.gcps)}",
            "",
            "homography:",
            "  approach: feature_match",
            "",
            "  feature_match:",
            "    detector: sift",
            "    min_matches: 4",
            "    ransac_threshold: 5.0",
            "",
            "    # Camera Capture Context",
            "    camera_capture_context:",
            f"      camera_name: \"{self.camera_name}\"",
            f"      image_width: {self.frame_width}",
            f"      image_height: {self.frame_height}",
        ]

        if self.ptz_status:
            lines.extend([
                "      ptz_position:",
                f"        pan: {self.ptz_status.get('pan', 0):.1f}",
                f"        tilt: {self.ptz_status.get('tilt', 0):.1f}",
                f"        zoom: {self.ptz_status.get('zoom', 1.0):.1f}",
            ])

        if self.intrinsics:
            lines.extend([
                "      intrinsics:",
                f"        focal_length_px: {self.intrinsics['focal_length_px']:.2f}",
                "        principal_point:",
                f"          cx: {self.intrinsics['principal_point']['cx']:.1f}",
                f"          cy: {self.intrinsics['principal_point']['cy']:.1f}",
            ])

        lines.extend([
            f"      capture_timestamp: \"{self.capture_timestamp}\"",
            "      # Coordinate system: image_v means V=0 at top (standard image coords)",
            "      # Old files without this field used leaflet_y format (V=0 at bottom)",
            "      coordinate_system: image_v",
            "      notes: \"\"",
            "",
            "    # Ground Control Points",
            "    ground_control_points:",
        ])

        for i, gcp in enumerate(self.gcps):
            lat = gcp['gps']['latitude']
            lon = gcp['gps']['longitude']
            u = gcp['image']['u']
            v = gcp['image']['v']

            # Convert V to image_v format if loaded from legacy (leaflet_y) format
            # Legacy: v was stored as leaflet_y (0 at bottom)
            # image_v: v should be image V (0 at top)
            # Conversion: image_v = frame_height - leaflet_y
            if self.coordinate_system is None:  # Legacy format
                v = self.frame_height - v

            desc = gcp.get('metadata', {}).get('description', f'GCP {i+1}')
            accuracy = gcp.get('metadata', {}).get('accuracy', 'medium')
            timestamp = gcp.get('metadata', {}).get('timestamp', '')

            lines.extend([
                f"      # GCP {i+1}: {desc}",
                "      - gps:",
                f"          latitude: {lat}",
                f"          longitude: {lon}",
                "        image:",
                f"          u: {u}",
                f"          v: {v}",
                "        metadata:",
                f"          description: \"{desc}\"",
                f"          accuracy: {accuracy}",
                f"          timestamp: \"{timestamp}\"",
                "",
            ])

        return "\n".join(lines)


def generate_capture_html(session: GCPCaptureWebSession, frame_path: str) -> str:
    """Generate the HTML interface for GCP capture."""

    gcps_json = json.dumps(session.gcps)
    distribution_json = json.dumps(session.calculate_distribution())
    homography_json = json.dumps(session.update_homography())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GCP Capture - {session.camera_name}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }}

        .container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}

        .header {{
            background: #2d2d2d;
            padding: 12px 20px;
            border-bottom: 1px solid #444;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header h1 {{
            font-size: 18px;
            font-weight: 500;
        }}

        .header-info {{
            font-size: 12px;
            color: #888;
        }}

        .main-content {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        .image-panel {{
            flex: 1;
            position: relative;
            background: #111;
        }}

        #imageMap {{
            width: 100%;
            height: 100%;
            background: #111;
        }}

        .side-panel {{
            width: 350px;
            background: #2d2d2d;
            border-left: 1px solid #444;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .panel-section {{
            padding: 15px;
            border-bottom: 1px solid #444;
        }}

        .panel-section h3 {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #fff;
        }}

        /* Distribution Panel */
        .distribution-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }}

        .metric-box {{
            background: #3a3a3a;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 24px;
            font-weight: 600;
            color: #4CAF50;
        }}

        .metric-value.warning {{
            color: #ff9800;
        }}

        .metric-value.error {{
            color: #f44336;
        }}

        .metric-label {{
            font-size: 11px;
            color: #888;
            margin-top: 4px;
        }}

        /* Quadrant visualization */
        .quadrant-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px;
            width: 80px;
            height: 60px;
            margin: 0 auto;
        }}

        .quadrant {{
            background: #444;
            border-radius: 3px;
            transition: background 0.3s;
        }}

        .quadrant.covered {{
            background: #4CAF50;
        }}

        /* Warnings */
        .warnings {{
            margin-top: 10px;
        }}

        .warning-item {{
            background: rgba(255, 152, 0, 0.15);
            border-left: 3px solid #ff9800;
            padding: 8px 10px;
            margin-bottom: 6px;
            font-size: 12px;
            border-radius: 0 4px 4px 0;
        }}

        /* GCP List */
        .gcp-list {{
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }}

        .gcp-item {{
            background: #3a3a3a;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 8px;
            position: relative;
        }}

        .gcp-item:hover {{
            background: #454545;
        }}

        .gcp-name {{
            font-weight: 600;
            margin-bottom: 4px;
        }}

        .gcp-coords {{
            font-size: 11px;
            color: #888;
            font-family: monospace;
        }}

        .gcp-delete {{
            position: absolute;
            top: 8px;
            right: 8px;
            background: none;
            border: none;
            color: #888;
            cursor: pointer;
            font-size: 16px;
            padding: 4px;
        }}

        .gcp-delete:hover {{
            color: #f44336;
        }}

        /* Actions */
        .actions {{
            padding: 15px;
            border-top: 1px solid #444;
            display: flex;
            gap: 10px;
        }}

        .btn {{
            flex: 1;
            padding: 10px 15px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .btn-primary {{
            background: #4CAF50;
            color: white;
        }}

        .btn-primary:hover {{
            background: #45a049;
        }}

        .btn-primary:disabled {{
            background: #666;
            cursor: not-allowed;
        }}

        .btn-secondary {{
            background: #555;
            color: white;
        }}

        .btn-secondary:hover {{
            background: #666;
        }}

        .btn-batch {{
            background: #666;
            color: white;
        }}

        .btn-batch.active {{
            background: #ff9800;
            color: black;
        }}

        .btn-batch:hover {{
            background: #777;
        }}

        .btn-batch.active:hover {{
            background: #ffb74d;
        }}

        /* Modal */
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 2000;
            justify-content: center;
            align-items: center;
        }}

        .modal.active {{
            display: flex;
        }}

        .modal-content {{
            background: #2d2d2d;
            padding: 25px;
            border-radius: 12px;
            width: 400px;
            max-width: 90%;
        }}

        .modal-header {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
        }}

        .form-group {{
            margin-bottom: 15px;
        }}

        .form-group label {{
            display: block;
            font-size: 12px;
            color: #888;
            margin-bottom: 6px;
        }}

        .form-group input, .form-group select {{
            width: 100%;
            padding: 10px;
            border: 1px solid #555;
            border-radius: 6px;
            background: #3a3a3a;
            color: #e0e0e0;
            font-size: 14px;
        }}

        .form-group input:focus {{
            outline: none;
            border-color: #4CAF50;
        }}

        .form-row {{
            display: flex;
            gap: 10px;
        }}

        .form-row .form-group {{
            flex: 1;
        }}

        .modal-actions {{
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }}

        /* Crosshair */
        .crosshair {{
            position: absolute;
            top: 50%;
            left: 50%;
            pointer-events: none;
            z-index: 1000;
            transform: translate(-50%, -50%);
        }}

        .crosshair-h, .crosshair-v {{
            position: absolute;
            background: rgba(255, 255, 255, 0.3);
        }}

        .crosshair-h {{
            width: 40px;
            height: 1px;
            left: -20px;
            top: 0;
        }}

        .crosshair-v {{
            width: 1px;
            height: 40px;
            left: 0;
            top: -20px;
        }}

        /* GCP markers on map */
        .gcp-marker {{
            background: #4CAF50;
            border: 2px solid white;
            border-radius: 50%;
            width: 12px;
            height: 12px;
            margin-left: -6px;
            margin-top: -6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}

        .gcp-marker.pending {{
            background: #ff9800;
        }}

        /* Instructions */
        .instructions {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 13px;
            z-index: 1000;
            text-align: center;
        }}

        .instructions.capture-mode {{
            background: rgba(76, 175, 80, 0.9);
        }}

        /* Zoom controls */
        .zoom-info {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.7);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            z-index: 1000;
        }}

        /* Leaflet overrides for dark theme */
        .leaflet-container {{
            background: #111;
        }}

        .leaflet-control-zoom a {{
            background: #2d2d2d !important;
            color: #e0e0e0 !important;
            border-color: #444 !important;
        }}

        .leaflet-control-zoom a:hover {{
            background: #3a3a3a !important;
        }}

        /* Homography Quality Panel */
        .homography-panel {{
            margin-bottom: 15px;
        }}

        .error-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }}

        .error-good {{ background: #4CAF50; }}
        .error-warning {{ background: #ff9800; }}
        .error-bad {{ background: #f44336; }}

        .gcp-error {{
            font-size: 11px;
            color: #888;
            font-family: monospace;
        }}

        .gcp-error.good {{ color: #4CAF50; }}
        .gcp-error.warning {{ color: #ff9800; }}
        .gcp-error.bad {{ color: #f44336; }}

        .outlier-badge {{
            background: #f44336;
            color: white;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 10px;
            margin-left: 8px;
        }}

        .btn-danger {{
            background: #c62828;
            color: white;
        }}

        .btn-danger:hover {{
            background: #b71c1c;
        }}

        .btn-danger:disabled {{
            background: #666;
            cursor: not-allowed;
        }}

        .outlier-summary {{
            background: rgba(244, 67, 54, 0.15);
            border-left: 3px solid #f44336;
            padding: 8px 10px;
            margin-top: 10px;
            font-size: 12px;
            border-radius: 0 4px 4px 0;
        }}

        .inlier-stats {{
            background: rgba(76, 175, 80, 0.15);
            border-left: 3px solid #4CAF50;
            padding: 8px 10px;
            margin-top: 10px;
            font-size: 12px;
            border-radius: 0 4px 4px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GCP Capture - {session.camera_name}</h1>
            <div class="header-info">
                Frame: {session.frame_width} x {session.frame_height}
                {f"| PTZ: P={session.ptz_status['pan']:.1f} T={session.ptz_status['tilt']:.1f} Z={session.ptz_status['zoom']:.1f}x" if session.ptz_status else ""}
                <button class="btn btn-batch" id="batchModeBtn" onclick="toggleBatchMode()" style="margin-left: 20px; padding: 4px 12px; font-size: 12px;">Batch Mode: OFF</button>
            </div>
        </div>

        <div class="main-content">
            <div class="image-panel">
                <div id="imageMap"></div>
                <div class="crosshair" id="crosshair">
                    <div class="crosshair-h"></div>
                    <div class="crosshair-v"></div>
                </div>
                <div class="instructions" id="instructions">
                    Scroll to zoom, drag to pan. Click to add GCP.
                </div>
                <div class="zoom-info" id="zoomInfo">Zoom: 1x</div>
            </div>

            <div class="side-panel">
                <div class="panel-section">
                    <h3>Distribution Quality</h3>
                    <div class="distribution-grid">
                        <div class="metric-box">
                            <div class="metric-value" id="scoreValue">0.00</div>
                            <div class="metric-label">Score</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value" id="coverageValue">0%</div>
                            <div class="metric-label">Coverage</div>
                        </div>
                    </div>
                    <div class="metric-box" style="margin-bottom: 15px;">
                        <div style="font-size: 12px; color: #888; margin-bottom: 8px;">Quadrant Coverage</div>
                        <div class="quadrant-grid">
                            <div class="quadrant" id="q0"></div>
                            <div class="quadrant" id="q1"></div>
                            <div class="quadrant" id="q2"></div>
                            <div class="quadrant" id="q3"></div>
                        </div>
                    </div>
                    <div class="warnings" id="warnings"></div>
                </div>

                <div class="panel-section">
                    <h3>Homography Quality</h3>
                    <div id="homographyPanel">
                        <div style="color: #666; font-size: 12px;">Need at least 4 GCPs</div>
                    </div>
                </div>

                <div class="panel-section">
                    <h3>GCPs (<span id="gcpCount">0</span>)</h3>
                </div>

                <div class="gcp-list" id="gcpList">
                    <div style="color: #666; text-align: center; padding: 20px;">
                        Click on the image to add GCPs
                    </div>
                </div>

                <div class="actions" style="flex-wrap: wrap;">
                    <input type="file" id="yamlFileInput" accept=".yaml,.yml" style="display: none;" onchange="handleYamlUpload(event)">
                    <button class="btn btn-secondary" onclick="document.getElementById('yamlFileInput').click()" style="flex: 0 0 auto;">Load YAML</button>
                    <button class="btn btn-secondary" onclick="clearAllGCPs()">Clear All</button>
                    <button class="btn btn-danger" id="removeOutliersBtn" onclick="removeOutliers()" disabled>Remove Outliers</button>
                    <button class="btn btn-primary" id="saveBtn" onclick="saveConfig()" disabled>Save YAML</button>
                </div>
                <div style="padding: 10px 15px; font-size: 11px; color: #666; border-top: 1px solid #444;">
                    Tip: Drag markers to fine-tune. Red markers are outliers.
                </div>
            </div>
        </div>
    </div>

    <!-- Add GCP Modal -->
    <div class="modal" id="addGcpModal">
        <div class="modal-content">
            <div class="modal-header">Add Ground Control Point</div>
            <div class="form-group">
                <label>Pixel Position</label>
                <input type="text" id="pixelPos" readonly>
            </div>
            <div class="form-group">
                <label>GPS Coordinates (paste from Google Maps)</label>
                <input type="text" id="gpsInput" placeholder="39.640296, -0.230037 or 39°38'25.7&quot;N 0°13'48.7&quot;W">
                <div style="font-size: 11px; color: #666; margin-top: 4px;">
                    Accepts: "39.640296, -0.230037" or "39°38'25.7"N 0°13'48.7"W"
                </div>
            </div>
            <div id="parsedCoords" style="display: none; background: #3a3a3a; padding: 8px 10px; border-radius: 4px; margin-bottom: 10px; font-size: 12px;">
                <span style="color: #888;">Parsed:</span> <span id="parsedLat"></span>, <span id="parsedLon"></span>
            </div>
            <div id="predictedError" style="display: none; padding: 8px 10px; border-radius: 4px; margin-bottom: 15px; font-size: 12px; border-left: 3px solid #888;">
                <span style="color: #888;">Predicted error:</span> <span id="predictedErrorValue"></span>
                <div id="predictedErrorMessage" style="margin-top: 4px; font-size: 11px;"></div>
            </div>
            <div class="form-group">
                <label>Description (optional)</label>
                <input type="text" id="descInput" placeholder="e.g., Corner of zebra crossing">
            </div>
            <div class="form-group">
                <label>Accuracy</label>
                <select id="accuracyInput">
                    <option value="high">High</option>
                    <option value="medium" selected>Medium</option>
                    <option value="low">Low</option>
                </select>
            </div>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                <button class="btn btn-primary" onclick="confirmAddGCP()">Add GCP</button>
            </div>
        </div>
    </div>

    <!-- Batch Mode Modal -->
    <div class="modal" id="batchModal">
        <div class="modal-content" style="max-width: 700px; max-height: 80vh; overflow-y: auto;">
            <div class="modal-header">Batch Mode - Finalize GCPs</div>
            <div style="margin-bottom: 15px; color: #888; font-size: 13px;">
                <span id="batchPointCount">0</span> points clicked. Assign names and load a KML file with GPS coordinates.
            </div>
            <div style="margin-bottom: 15px;">
                <input type="file" id="kmlFileInput" accept=".kml" style="display: none;" onchange="handleKmlUpload(event)">
                <button class="btn btn-secondary" onclick="document.getElementById('kmlFileInput').click()" style="padding: 6px 12px; font-size: 13px;">
                    Load KML File
                </button>
                <span id="kmlStatus" style="margin-left: 10px; font-size: 12px; color: #888;"></span>
            </div>
            <div id="batchPointsList" style="max-height: 400px; overflow-y: auto;">
                <!-- Points will be listed here -->
            </div>
            <div class="modal-actions" style="margin-top: 15px;">
                <button class="btn btn-secondary" onclick="closeBatchModal()">Cancel</button>
                <button class="btn btn-primary" id="batchConfirmBtn" onclick="confirmBatchGCPs()" disabled>Add All GCPs</button>
            </div>
        </div>
    </div>

    <script>
        // Configuration
        const imageWidth = {session.frame_width};
        const imageHeight = {session.frame_height};
        const imagePath = "{frame_path}";

        // State
        let gcps = {gcps_json};
        let distribution = {distribution_json};
        let homography = {homography_json};
        let pendingClick = null;
        let gcpMarkers = [];
        // Coordinate system: 'image_v' = standard (V=0 at top), null = legacy leaflet_y format (V=0 at bottom)
        let coordinateSystem = 'image_v';  // Default for new captures

        // Batch mode state
        let batchMode = false;
        let batchPoints = [];  // Array of {{u, v}} pixel positions
        let batchMarkers = []; // Temporary markers shown during batch mode
        let batchGpsCoords = []; // GPS coords loaded from KML

        // Initialize Leaflet map with simple CRS for image
        const map = L.map('imageMap', {{
            crs: L.CRS.Simple,
            minZoom: -3,
            maxZoom: 5,
            zoomSnap: 0.25,
            zoomDelta: 0.5
        }});

        // Calculate bounds for the image
        const bounds = [[0, 0], [imageHeight, imageWidth]];
        const imageBounds = L.latLngBounds([[0, 0], [imageHeight, imageWidth]]);

        // Add the image as an overlay
        L.imageOverlay(imagePath, bounds).addTo(map);

        // Fit the map to show the full image
        map.fitBounds(bounds);

        // Update zoom info display
        function updateZoomInfo() {{
            const zoom = map.getZoom();
            const scale = Math.pow(2, zoom + 1).toFixed(1);
            document.getElementById('zoomInfo').textContent = `Zoom: ${{scale}}x`;
        }}
        map.on('zoom', updateZoomInfo);
        updateZoomInfo();

        // ============================================================
        // GPS Coordinate Parsing
        // ============================================================

        /**
         * Parse GPS coordinates from various formats:
         * - Decimal: "39.640296, -0.230037"
         * - DMS: "39°38'25.7"N 0°13'48.7"W"
         * Returns {{ lat, lon }} or null if parsing fails
         */
        function parseGPSCoordinates(input) {{
            if (!input || typeof input !== 'string') return null;

            // Clean up input
            input = input.trim();

            // Try decimal format first: "39.640296, -0.230037"
            const decimalMatch = input.match(/^(-?\\d+\\.?\\d*)\\s*[,\\s]\\s*(-?\\d+\\.?\\d*)$/);
            if (decimalMatch) {{
                const lat = parseFloat(decimalMatch[1]);
                const lon = parseFloat(decimalMatch[2]);
                if (!isNaN(lat) && !isNaN(lon) && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {{
                    return {{ lat, lon }};
                }}
            }}

            // Try DMS format: "39°38'25.7"N 0°13'48.7"W"
            // Also handles variations like: 39°38'25.7"N, 0°13'48.7"W
            const dmsPattern = /(-?\\d+)[°]\\s*(\\d+)[′']\\s*(\\d+\\.?\\d*)[″"]?\\s*([NSns])?[,\\s]+(-?\\d+)[°]\\s*(\\d+)[′']\\s*(\\d+\\.?\\d*)[″"]?\\s*([EWew])?/;
            const dmsMatch = input.match(dmsPattern);
            if (dmsMatch) {{
                let latDeg = parseFloat(dmsMatch[1]);
                const latMin = parseFloat(dmsMatch[2]);
                const latSec = parseFloat(dmsMatch[3]);
                const latDir = (dmsMatch[4] || 'N').toUpperCase();

                let lonDeg = parseFloat(dmsMatch[5]);
                const lonMin = parseFloat(dmsMatch[6]);
                const lonSec = parseFloat(dmsMatch[7]);
                const lonDir = (dmsMatch[8] || 'E').toUpperCase();

                // Convert to decimal
                let lat = Math.abs(latDeg) + latMin / 60 + latSec / 3600;
                let lon = Math.abs(lonDeg) + lonMin / 60 + lonSec / 3600;

                // Apply direction
                if (latDir === 'S') lat = -lat;
                if (lonDir === 'W') lon = -lon;

                if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {{
                    return {{ lat, lon }};
                }}
            }}

            // Try simpler DMS without seconds: "39°38'N 0°13'W"
            const dmsSimplePattern = /(-?\\d+)[°]\\s*(\\d+\\.?\\d*)[′']\\s*([NSns])?[,\\s]+(-?\\d+)[°]\\s*(\\d+\\.?\\d*)[′']\\s*([EWew])?/;
            const dmsSimpleMatch = input.match(dmsSimplePattern);
            if (dmsSimpleMatch) {{
                let latDeg = parseFloat(dmsSimpleMatch[1]);
                const latMin = parseFloat(dmsSimpleMatch[2]);
                const latDir = (dmsSimpleMatch[3] || 'N').toUpperCase();

                let lonDeg = parseFloat(dmsSimpleMatch[4]);
                const lonMin = parseFloat(dmsSimpleMatch[5]);
                const lonDir = (dmsSimpleMatch[6] || 'E').toUpperCase();

                let lat = Math.abs(latDeg) + latMin / 60;
                let lon = Math.abs(lonDeg) + lonMin / 60;

                if (latDir === 'S') lat = -lat;
                if (lonDir === 'W') lon = -lon;

                if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {{
                    return {{ lat, lon }};
                }}
            }}

            return null;
        }}

        // Update parsed coordinates display on input
        function updateParsedDisplay() {{
            const input = document.getElementById('gpsInput').value;
            const parsed = parseGPSCoordinates(input);
            const parsedDiv = document.getElementById('parsedCoords');
            const errorDiv = document.getElementById('predictedError');

            if (parsed) {{
                document.getElementById('parsedLat').textContent = parsed.lat.toFixed(6);
                document.getElementById('parsedLon').textContent = parsed.lon.toFixed(6);
                parsedDiv.style.display = 'block';
                parsedDiv.style.borderLeft = '3px solid #4CAF50';

                // Call predict_error API if we have a pending click position and enough GCPs
                if (pendingClick && gcps.length >= 4) {{
                    fetch('/api/predict_error', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            u: pendingClick.u,
                            v: pendingClick.v,
                            lat: parsed.lat,
                            lon: parsed.lon
                        }})
                    }})
                    .then(r => r.json())
                    .then(data => {{
                        if (data.available) {{
                            const errorValue = document.getElementById('predictedErrorValue');
                            const errorMessage = document.getElementById('predictedErrorMessage');

                            errorValue.textContent = data.predicted_error_px.toFixed(1) + 'px';

                            // Color based on status
                            if (data.status === 'good') {{
                                errorDiv.style.borderLeftColor = '#4CAF50';
                                errorDiv.style.background = 'rgba(76, 175, 80, 0.1)';
                                errorValue.style.color = '#4CAF50';
                            }} else if (data.status === 'warning') {{
                                errorDiv.style.borderLeftColor = '#ff9800';
                                errorDiv.style.background = 'rgba(255, 152, 0, 0.1)';
                                errorValue.style.color = '#ff9800';
                            }} else {{
                                errorDiv.style.borderLeftColor = '#f44336';
                                errorDiv.style.background = 'rgba(244, 67, 54, 0.1)';
                                errorValue.style.color = '#f44336';
                            }}

                            errorMessage.textContent = data.message;
                            errorDiv.style.display = 'block';
                        }} else {{
                            errorDiv.style.display = 'none';
                        }}
                    }})
                    .catch(() => {{
                        errorDiv.style.display = 'none';
                    }});
                }} else {{
                    errorDiv.style.display = 'none';
                }}
            }} else if (input.trim()) {{
                document.getElementById('parsedLat').textContent = '?';
                document.getElementById('parsedLon').textContent = '?';
                parsedDiv.style.display = 'block';
                parsedDiv.style.borderLeft = '3px solid #f44336';
                errorDiv.style.display = 'none';
            }} else {{
                parsedDiv.style.display = 'none';
                errorDiv.style.display = 'none';
            }}
        }}

        // Handle click on map to add GCP
        map.on('click', function(e) {{
            // Convert Leaflet coords to image pixel coords
            // Uses coordinate-system-aware conversion
            const imgCoords = leafletToImage(e.latlng.lat, e.latlng.lng);
            const u = imgCoords.u;
            const v = imgCoords.v;

            // Check bounds
            if (u < 0 || u > imageWidth || v < 0 || v > imageHeight) {{
                return;
            }}

            // Batch mode: just add to list without modal
            if (batchMode) {{
                batchPoints.push({{ u: imgCoords.u, v: imgCoords.v }});
                updateBatchMarkers();
                updateBatchUI();
                return;
            }}

            // Normal mode: Store pending click and show modal
            pendingClick = {{ u: u, v: v }};
            document.getElementById('pixelPos').value = `(${{u.toFixed(1)}}, ${{v.toFixed(1)}})`;
            document.getElementById('gpsInput').value = '';
            document.getElementById('descInput').value = '';
            document.getElementById('parsedCoords').style.display = 'none';
            document.getElementById('addGcpModal').classList.add('active');
            document.getElementById('gpsInput').focus();
        }});

        // Modal functions
        function closeModal() {{
            document.getElementById('addGcpModal').classList.remove('active');
            pendingClick = null;
        }}

        function confirmAddGCP() {{
            if (!pendingClick) return;

            const gpsInput = document.getElementById('gpsInput').value;
            const parsed = parseGPSCoordinates(gpsInput);
            const desc = document.getElementById('descInput').value.trim();
            const accuracy = document.getElementById('accuracyInput').value;

            if (!parsed) {{
                alert('Could not parse GPS coordinates.\\n\\nAccepted formats:\\n• Decimal: 39.640296, -0.230037\\n• DMS: 39°38\\'25.7"N 0°13\\'48.7"W');
                return;
            }}

            const {{ lat, lon }} = parsed;

            // Add GCP via API
            fetch('/api/add_gcp', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    u: pendingClick.u,
                    v: pendingClick.v,
                    lat: lat,
                    lon: lon,
                    description: desc,
                    accuracy: accuracy
                }})
            }})
            .then(r => r.json())
            .then(data => {{
                gcps = data.gcps;
                distribution = data.distribution;
                homography = data.homography;
                updateUI();
                closeModal();
            }});
        }}

        // Delete GCP
        function deleteGCP(index) {{
            fetch('/api/delete_gcp', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ index: index }})
            }})
            .then(r => r.json())
            .then(data => {{
                gcps = data.gcps;
                distribution = data.distribution;
                homography = data.homography;
                updateUI();
            }});
        }}

        // Remove all outliers
        function removeOutliers() {{
            if (!homography || !homography.outliers || homography.outliers.length === 0) return;

            const outlierCount = homography.outliers.length;
            const outlierNames = homography.outliers.map(o => o.description).join(', ');

            if (!confirm(`Remove ${{outlierCount}} outlier(s)?\\n\\n${{outlierNames}}`)) return;

            fetch('/api/remove_outliers', {{ method: 'POST' }})
            .then(r => r.json())
            .then(data => {{
                if (data.error) {{
                    alert('Error removing outliers: ' + data.error);
                    return;
                }}
                gcps = data.gcps;
                distribution = data.distribution;
                homography = data.homography;
                updateUI();

                if (data.removed_count > 0) {{
                    alert(`Removed ${{data.removed_count}} outlier(s):\\n${{data.removed_descriptions.join(', ')}}`);
                }}
            }});
        }}

        // Clear all GCPs
        function clearAllGCPs() {{
            if (gcps.length === 0) return;
            if (!confirm(`Clear all ${{gcps.length}} GCPs?`)) return;

            fetch('/api/clear_gcps', {{ method: 'POST' }})
            .then(r => r.json())
            .then(data => {{
                gcps = data.gcps;
                distribution = data.distribution;
                homography = data.homography;
                updateUI();
            }});
        }}

        // Save config
        function saveConfig() {{
            window.location.href = '/api/save';
        }}

        // Update UI with current state
        function updateUI() {{
            // Update distribution metrics
            const score = distribution.distribution_score;
            const scoreEl = document.getElementById('scoreValue');
            scoreEl.textContent = score.toFixed(2);
            scoreEl.className = 'metric-value ' + (score > 0.7 ? '' : score > 0.5 ? 'warning' : 'error');

            const coverage = distribution.coverage_ratio * 100;
            const coverageEl = document.getElementById('coverageValue');
            coverageEl.textContent = coverage.toFixed(0) + '%';
            coverageEl.className = 'metric-value ' + (coverage >= 35 ? '' : coverage >= 15 ? 'warning' : 'error');

            // Update quadrant visualization
            for (let i = 0; i < 4; i++) {{
                const el = document.getElementById('q' + i);
                if (distribution.quadrants && distribution.quadrants[i]) {{
                    el.classList.add('covered');
                }} else {{
                    el.classList.remove('covered');
                }}
            }}

            // Update warnings
            const warningsEl = document.getElementById('warnings');
            if (distribution.warnings && distribution.warnings.length > 0) {{
                warningsEl.innerHTML = distribution.warnings
                    .map(w => `<div class="warning-item">${{w}}</div>`)
                    .join('');
            }} else {{
                warningsEl.innerHTML = '';
            }}

            // Update homography panel
            updateHomographyPanel();

            // Update GCP count
            document.getElementById('gcpCount').textContent = gcps.length;

            // Update GCP list with reprojection errors
            const listEl = document.getElementById('gcpList');
            if (gcps.length === 0) {{
                listEl.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">Click on the image to add GCPs</div>';
            }} else {{
                listEl.innerHTML = gcps.map((gcp, i) => {{
                    // Get error info from homography if available
                    let errorHtml = '';
                    let outlierBadge = '';
                    if (homography && homography.errors && homography.errors[i]) {{
                        const errInfo = homography.errors[i];
                        const statusClass = errInfo.status || 'good';
                        errorHtml = `<div class="gcp-error ${{statusClass}}">
                            <span class="error-indicator error-${{statusClass}}"></span>
                            Error: ${{errInfo.error_px.toFixed(1)}}px
                            ${{!errInfo.is_inlier ? ' (outlier)' : ''}}
                        </div>`;
                        if (!errInfo.is_inlier) {{
                            outlierBadge = '<span class="outlier-badge">OUTLIER</span>';
                        }}
                    }}

                    return `
                        <div class="gcp-item" onmouseover="highlightGCP(${{i}})" onmouseout="unhighlightGCP(${{i}})">
                            <button class="gcp-delete" onclick="deleteGCP(${{i}})">&times;</button>
                            <div class="gcp-name">${{gcp.metadata?.description || 'GCP ' + (i+1)}}${{outlierBadge}}</div>
                            <div class="gcp-coords">
                                Pixel: (${{gcp.image.u.toFixed(1)}}, ${{gcp.image.v.toFixed(1)}})<br>
                                GPS: (${{gcp.gps.latitude.toFixed(6)}}, ${{gcp.gps.longitude.toFixed(6)}})
                            </div>
                            ${{errorHtml}}
                        </div>
                    `;
                }}).join('');
            }}

            // Update markers on map
            updateMarkers();

            // Enable/disable save button
            document.getElementById('saveBtn').disabled = gcps.length < 4;

            // Enable/disable remove outliers button
            const outlierCount = homography && homography.outliers ? homography.outliers.length : 0;
            document.getElementById('removeOutliersBtn').disabled = outlierCount === 0;
            document.getElementById('removeOutliersBtn').textContent = outlierCount > 0
                ? `Remove Outliers (${{outlierCount}})`
                : 'Remove Outliers';
        }}

        // Update the homography quality panel
        function updateHomographyPanel() {{
            const panel = document.getElementById('homographyPanel');

            if (!homography || !homography.available) {{
                panel.innerHTML = '<div style="color: #666; font-size: 12px;">Homography calculation not available</div>';
                return;
            }}

            if (gcps.length < 4) {{
                panel.innerHTML = '<div style="color: #666; font-size: 12px;">Need at least 4 GCPs for homography</div>';
                return;
            }}

            const numInliers = homography.num_inliers || 0;
            const numGcps = homography.num_gcps || 0;
            const inlierRatio = homography.inlier_ratio || 0;
            const meanError = homography.mean_error_px || 0;
            const maxError = homography.max_error_px || 0;
            const outliers = homography.outliers || [];

            // Determine quality status
            let qualityClass = 'good';
            let qualityText = 'Good';
            if (inlierRatio < 0.5 || meanError > 10) {{
                qualityClass = 'bad';
                qualityText = 'Poor';
            }} else if (inlierRatio < 0.7 || meanError > 5) {{
                qualityClass = 'warning';
                qualityText = 'Fair';
            }}

            let html = `
                <div class="distribution-grid" style="margin-bottom: 10px;">
                    <div class="metric-box">
                        <div class="metric-value ${{qualityClass === 'bad' ? 'error' : qualityClass === 'warning' ? 'warning' : ''}}">${{(inlierRatio * 100).toFixed(0)}}%</div>
                        <div class="metric-label">Inlier Ratio</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value ${{meanError > 10 ? 'error' : meanError > 5 ? 'warning' : ''}}">${{meanError.toFixed(1)}}px</div>
                        <div class="metric-label">Mean Error</div>
                    </div>
                </div>
            `;

            // Show inlier stats
            html += `
                <div class="inlier-stats">
                    <strong>${{numInliers}}/${{numGcps}}</strong> inliers | Max error: <strong>${{maxError.toFixed(1)}}px</strong>
                </div>
            `;

            // Show outlier summary if any
            if (outliers.length > 0) {{
                html += `
                    <div class="outlier-summary">
                        <strong>${{outliers.length}} outlier${{outliers.length > 1 ? 's' : ''}} detected:</strong><br>
                        ${{outliers.slice(0, 3).map(o => `${{o.description}} (${{o.error_px.toFixed(1)}}px)`).join(', ')}}
                        ${{outliers.length > 3 ? '...' : ''}}
                    </div>
                `;
            }}

            panel.innerHTML = html;
        }}

        // Create a custom draggable icon
        function createGcpIcon(color = '#4CAF50', size = 12) {{
            return L.divIcon({{
                className: 'gcp-drag-marker',
                html: `<div style="
                    width: ${{size}}px;
                    height: ${{size}}px;
                    background: ${{color}};
                    border: 2px solid white;
                    border-radius: 50%;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    cursor: grab;
                "></div>`,
                iconSize: [size, size],
                iconAnchor: [size/2, size/2]
            }});
        }}

        // Convert image pixel coords to Leaflet coords
        // For image_v format: Image V (0 at top) -> Leaflet Y (0 at bottom): leaflet_y = imageHeight - v
        // For legacy format (null): V was stored as leaflet_y directly, no conversion needed
        function imageToLeaflet(u, v) {{
            if (coordinateSystem === 'image_v') {{
                return [imageHeight - v, u];  // [lat, lng] = [leaflet_y, x]
            }} else {{
                // Legacy format: v is already leaflet_y
                return [v, u];
            }}
        }}

        // Convert Leaflet coords to image pixel coords
        // Uses current coordinate system to stay consistent with loaded data
        function leafletToImage(lat, lng) {{
            if (coordinateSystem === 'image_v') {{
                return {{ u: lng, v: imageHeight - lat }};
            }} else {{
                // Legacy format: store leaflet_y directly as v
                return {{ u: lng, v: lat }};
            }}
        }}

        // Update markers on map
        function updateMarkers() {{
            // Remove existing markers
            gcpMarkers.forEach(m => map.removeLayer(m));
            gcpMarkers = [];

            // Add new markers (draggable, color-coded by error status)
            gcps.forEach((gcp, i) => {{
                // Convert image coords to Leaflet coords (invert Y)
                const leafletCoords = imageToLeaflet(gcp.image.u, gcp.image.v);

                // Determine marker color based on reprojection error
                let markerColor = '#4CAF50';  // Default: green (good)
                let markerSize = 14;
                let tooltipExtra = '';

                if (homography && homography.errors && homography.errors[i]) {{
                    const errInfo = homography.errors[i];
                    if (!errInfo.is_inlier) {{
                        markerColor = '#f44336';  // Red for outliers
                        markerSize = 16;
                        tooltipExtra = ` - OUTLIER (${{errInfo.error_px.toFixed(1)}}px)`;
                    }} else if (errInfo.status === 'bad') {{
                        markerColor = '#f44336';  // Red for high error
                        tooltipExtra = ` (${{errInfo.error_px.toFixed(1)}}px)`;
                    }} else if (errInfo.status === 'warning') {{
                        markerColor = '#ff9800';  // Orange for medium error
                        tooltipExtra = ` (${{errInfo.error_px.toFixed(1)}}px)`;
                    }} else {{
                        tooltipExtra = ` (${{errInfo.error_px.toFixed(1)}}px)`;
                    }}
                }}

                const marker = L.marker(leafletCoords, {{
                    icon: createGcpIcon(markerColor, markerSize),
                    draggable: true
                }}).addTo(map);

                // Store index and original color for reference
                marker.gcpIndex = i;
                marker.originalColor = markerColor;
                marker.originalSize = markerSize;

                // CRITICAL: Stop click propagation to prevent map click from firing
                // This allows dragging to work without opening the modal
                marker.on('click', function(e) {{
                    L.DomEvent.stopPropagation(e);
                }});
                marker.on('mousedown', function(e) {{
                    L.DomEvent.stopPropagation(e);
                }});

                // Add label with error info
                marker.bindTooltip(`${{gcp.metadata?.description || 'GCP ' + (i+1)}}${{tooltipExtra}}`, {{
                    permanent: false,
                    direction: 'top',
                    offset: [0, -10]
                }});

                // Handle drag end - update position
                marker.on('dragend', function(e) {{
                    const newPos = e.target.getLatLng();
                    // Convert Leaflet coords back to image coords (invert Y)
                    const imgCoords = leafletToImage(newPos.lat, newPos.lng);
                    const newU = imgCoords.u;
                    const newV = imgCoords.v;

                    // Check bounds
                    if (newU < 0 || newU > imageWidth || newV < 0 || newV > imageHeight) {{
                        // Revert to original position (convert back to Leaflet coords)
                        e.target.setLatLng(imageToLeaflet(gcp.image.u, gcp.image.v));
                        return;
                    }}

                    // Update on server
                    fetch('/api/update_gcp_position', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            index: i,
                            u: newU,
                            v: newV
                        }})
                    }})
                    .then(r => r.json())
                    .then(data => {{
                        gcps = data.gcps;
                        distribution = data.distribution;
                        updateUI();
                    }});
                }});

                // Visual feedback during drag
                marker.on('dragstart', function() {{
                    marker.setIcon(createGcpIcon('#00ffff', 18));
                }});

                marker.on('drag', function() {{
                    // Update tooltip during drag with image coordinates
                    const pos = marker.getLatLng();
                    const imgCoords = leafletToImage(pos.lat, pos.lng);
                    marker.setTooltipContent(`${{gcp.metadata?.description || 'GCP ' + (i+1)}}<br>(${{imgCoords.u.toFixed(1)}}, ${{imgCoords.v.toFixed(1)}})`);
                }});

                gcpMarkers.push(marker);
            }});
        }}

        // Highlight GCP on hover
        function highlightGCP(index) {{
            if (gcpMarkers[index]) {{
                gcpMarkers[index].setIcon(createGcpIcon('#00ffff', 18));
            }}
        }}

        function unhighlightGCP(index) {{
            if (gcpMarkers[index]) {{
                // Restore to original color based on error status
                const marker = gcpMarkers[index];
                const color = marker.originalColor || '#4CAF50';
                const size = marker.originalSize || 14;
                marker.setIcon(createGcpIcon(color, size));
            }}
        }}

        // Handle YAML file upload
        function handleYamlUpload(event) {{
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {{
                const content = e.target.result;

                fetch('/api/load_yaml', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ yaml_content: content }})
                }})
                .then(r => r.json())
                .then(data => {{
                    if (data.error) {{
                        alert('Error loading YAML: ' + data.error);
                        return;
                    }}

                    // Set coordinate system based on loaded data
                    // If null/undefined, it's legacy format (V stored as leaflet_y)
                    coordinateSystem = data.coordinate_system || null;
                    console.log('Loaded coordinate system:', coordinateSystem);

                    gcps = data.gcps;
                    distribution = data.distribution;
                    homography = data.homography;
                    updateUI();

                    let msg = `Loaded ${{data.gcps_loaded}} GCPs`;
                    if (data.warnings && data.warnings.length > 0) {{
                        msg += '\\n\\nWarnings:\\n' + data.warnings.join('\\n');
                    }}

                    // Check if PTZ position differs and live camera is available
                    if (data.has_live_camera && data.loaded_ptz && data.current_ptz) {{
                        const loadedPtz = data.loaded_ptz;
                        const currentPtz = data.current_ptz;

                        // Check if positions differ significantly (threshold: 0.5 deg for pan/tilt, 0.1 for zoom)
                        const panDiff = Math.abs((loadedPtz.pan || 0) - (currentPtz.pan || 0));
                        const tiltDiff = Math.abs((loadedPtz.tilt || 0) - (currentPtz.tilt || 0));
                        const zoomDiff = Math.abs((loadedPtz.zoom || 1) - (currentPtz.zoom || 1));

                        if (panDiff > 0.5 || tiltDiff > 0.5 || zoomDiff > 0.1) {{
                            const moveMsg = `\\n\\nThe YAML was captured at a different PTZ position:\\n` +
                                `  Loaded:  P=${{loadedPtz.pan?.toFixed(1) || '?'}}° T=${{loadedPtz.tilt?.toFixed(1) || '?'}}° Z=${{loadedPtz.zoom?.toFixed(1) || '?'}}x\\n` +
                                `  Current: P=${{currentPtz.pan?.toFixed(1) || '?'}}° T=${{currentPtz.tilt?.toFixed(1) || '?'}}° Z=${{currentPtz.zoom?.toFixed(1) || '?'}}x\\n\\n` +
                                `Move camera to the saved position?`;

                            if (confirm(msg + moveMsg)) {{
                                // Move camera to loaded position
                                moveCameraToPosition(loadedPtz);
                                return;
                            }}
                        }}
                    }}

                    alert(msg);
                }});
            }};
            reader.readAsText(file);

            // Reset input so same file can be loaded again
            event.target.value = '';
        }}

        // Move camera to PTZ position
        function moveCameraToPosition(ptzPosition) {{
            const statusEl = document.getElementById('instructions');
            const originalText = statusEl.textContent;
            const originalBg = statusEl.style.background;

            statusEl.textContent = 'Moving camera to saved position...';
            statusEl.style.background = 'rgba(255, 152, 0, 0.9)';

            fetch('/api/move_camera', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    ptz_position: ptzPosition,
                    wait_time: 3.0
                }})
            }})
            .then(r => r.json())
            .then(data => {{
                statusEl.textContent = originalText;
                statusEl.style.background = originalBg;

                if (data.success) {{
                    alert(data.message + '\\n\\nNote: The displayed frame is from the old position. Reload page for new frame.');
                }} else {{
                    alert('Failed to move camera: ' + data.message);
                }}
            }})
            .catch(err => {{
                statusEl.textContent = originalText;
                statusEl.style.background = originalBg;
                alert('Error moving camera: ' + err);
            }});
        }}

        // ============================================================
        // Batch Mode Functions
        // ============================================================

        function toggleBatchMode() {{
            if (batchMode) {{
                // Turning off batch mode - show finalize modal if there are points
                if (batchPoints.length > 0) {{
                    openBatchModal();
                }} else {{
                    batchMode = false;
                    updateBatchModeUI();
                }}
            }} else {{
                // Turning on batch mode
                batchMode = true;
                batchPoints = [];
                batchGpsCoords = [];
                clearBatchMarkers();
                updateBatchModeUI();
            }}
        }}

        function updateBatchModeUI() {{
            const btn = document.getElementById('batchModeBtn');
            const instructions = document.getElementById('instructions');
            if (batchMode) {{
                btn.textContent = `Batch Mode: ON (${{batchPoints.length}})`;
                btn.classList.add('active');
                instructions.textContent = 'BATCH MODE: Click points to add them. Click button again to finalize.';
                instructions.style.background = 'rgba(255, 152, 0, 0.9)';
                instructions.style.color = 'black';
            }} else {{
                btn.textContent = 'Batch Mode: OFF';
                btn.classList.remove('active');
                instructions.textContent = 'Scroll to zoom, drag to pan. Click to add GCP.';
                instructions.style.background = 'rgba(0,0,0,0.7)';
                instructions.style.color = 'white';
            }}
        }}

        function updateBatchUI() {{
            document.getElementById('batchModeBtn').textContent = `Batch Mode: ON (${{batchPoints.length}})`;
        }}

        function updateBatchMarkers() {{
            // Clear existing batch markers
            clearBatchMarkers();

            // Add numbered markers for batch points
            batchPoints.forEach((pt, i) => {{
                // Convert image coords to Leaflet coords (invert Y)
                const leafletCoords = imageToLeaflet(pt.u, pt.v);
                const marker = L.marker(leafletCoords, {{
                    icon: L.divIcon({{
                        className: 'batch-marker',
                        html: `<div style="
                            width: 24px;
                            height: 24px;
                            background: #ff9800;
                            border: 2px solid white;
                            border-radius: 50%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 11px;
                            font-weight: bold;
                            color: black;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                        ">${{i + 1}}</div>`,
                        iconSize: [24, 24],
                        iconAnchor: [12, 12]
                    }}),
                    draggable: true
                }}).addTo(map);

                // Stop click propagation to prevent adding duplicate points
                marker.on('click', function(e) {{
                    L.DomEvent.stopPropagation(e);
                }});
                marker.on('mousedown', function(e) {{
                    L.DomEvent.stopPropagation(e);
                }});

                // Allow dragging batch markers to adjust position
                marker.on('dragend', function(e) {{
                    const newPos = e.target.getLatLng();
                    // Convert Leaflet coords back to image coords (coordinate-system aware)
                    const newImgCoords = leafletToImage(newPos.lat, newPos.lng);
                    batchPoints[i].u = newImgCoords.u;
                    batchPoints[i].v = newImgCoords.v;
                }});

                batchMarkers.push(marker);
            }});
        }}

        function clearBatchMarkers() {{
            batchMarkers.forEach(m => map.removeLayer(m));
            batchMarkers = [];
        }}

        function openBatchModal() {{
            document.getElementById('batchPointCount').textContent = batchPoints.length;
            document.getElementById('kmlStatus').textContent = '';
            batchGpsCoords = [];

            // Build the points list
            let html = '<table style="width: 100%; border-collapse: collapse;">';
            html += '<tr style="background: #3a3a3a; font-size: 12px;">';
            html += '<th style="padding: 8px; text-align: left;">#</th>';
            html += '<th style="padding: 8px; text-align: left;">Pixel (u, v)</th>';
            html += '<th style="padding: 8px; text-align: left;">Name</th>';
            html += '<th style="padding: 8px; text-align: left;">GPS</th>';
            html += '<th style="padding: 8px; text-align: center;">Del</th>';
            html += '</tr>';

            batchPoints.forEach((pt, i) => {{
                html += `<tr style="border-bottom: 1px solid #444;">`;
                html += `<td style="padding: 8px; font-size: 12px;">${{i + 1}}</td>`;
                html += `<td style="padding: 8px; font-size: 12px;">(${{pt.u.toFixed(1)}}, ${{pt.v.toFixed(1)}})</td>`;
                html += `<td style="padding: 8px;"><input type="text" id="batchName_${{i}}" placeholder="GCP ${{i + 1}}" style="width: 150px; padding: 4px; font-size: 12px; background: #3a3a3a; border: 1px solid #555; color: white; border-radius: 4px;"></td>`;
                html += `<td style="padding: 8px; font-size: 12px;" id="batchGps_${{i}}">-</td>`;
                html += `<td style="padding: 8px; text-align: center;"><button onclick="removeBatchPoint(${{i}})" style="background: #c62828; color: white; border: none; padding: 2px 8px; cursor: pointer; border-radius: 3px; font-size: 11px;">×</button></td>`;
                html += '</tr>';
            }});
            html += '</table>';

            document.getElementById('batchPointsList').innerHTML = html;
            document.getElementById('batchConfirmBtn').disabled = true;
            document.getElementById('batchModal').classList.add('active');
        }}

        function closeBatchModal() {{
            document.getElementById('batchModal').classList.remove('active');
            // Keep batch mode on, user can continue adding points
        }}

        function removeBatchPoint(index) {{
            batchPoints.splice(index, 1);
            if (batchGpsCoords.length > index) {{
                batchGpsCoords.splice(index, 1);
            }}
            updateBatchMarkers();
            openBatchModal(); // Refresh the modal
        }}

        function handleKmlUpload(event) {{
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {{
                const content = e.target.result;
                const coords = parseKmlCoordinates(content);

                if (coords.length === 0) {{
                    document.getElementById('kmlStatus').textContent = '❌ No coordinates found in KML';
                    document.getElementById('kmlStatus').style.color = '#f44336';
                    return;
                }}

                if (coords.length !== batchPoints.length) {{
                    document.getElementById('kmlStatus').textContent = `❌ Mismatch: KML has ${{coords.length}} points, you clicked ${{batchPoints.length}}`;
                    document.getElementById('kmlStatus').style.color = '#f44336';
                    alert(`Count mismatch!\\n\\nKML file has ${{coords.length}} coordinates.\\nYou clicked ${{batchPoints.length}} points.\\n\\nPlease ensure the KML has the same number of points in the correct order.`);
                    return;
                }}

                // Assign GPS coords to batch points
                batchGpsCoords = coords;
                document.getElementById('kmlStatus').textContent = `✓ Loaded ${{coords.length}} coordinates`;
                document.getElementById('kmlStatus').style.color = '#4CAF50';

                // Update GPS display in table
                coords.forEach((coord, i) => {{
                    const gpsCell = document.getElementById(`batchGps_${{i}}`);
                    if (gpsCell) {{
                        gpsCell.textContent = `${{coord.lat.toFixed(6)}}, ${{coord.lon.toFixed(6)}}`;
                        gpsCell.style.color = '#4CAF50';
                    }}
                }});

                // Enable confirm button
                document.getElementById('batchConfirmBtn').disabled = false;
            }};
            reader.readAsText(file);

            // Reset input so same file can be loaded again
            event.target.value = '';
        }}

        function parseKmlCoordinates(kmlContent) {{
            const coords = [];

            // Parse KML - look for <coordinates> tags
            // KML format: longitude,latitude,altitude (whitespace separated)
            const coordPattern = /<coordinates>([^<]+)<\\/coordinates>/gi;
            let match;

            while ((match = coordPattern.exec(kmlContent)) !== null) {{
                const coordText = match[1].trim();
                // Split by whitespace (newlines, spaces)
                const points = coordText.split(/\\s+/);

                points.forEach(point => {{
                    if (!point.trim()) return;
                    const parts = point.split(',');
                    if (parts.length >= 2) {{
                        const lon = parseFloat(parts[0]);
                        const lat = parseFloat(parts[1]);
                        if (!isNaN(lat) && !isNaN(lon)) {{
                            coords.push({{ lat, lon }});
                        }}
                    }}
                }});
            }}

            // Also try to parse <Point> elements with single coordinates
            const pointPattern = /<Point>[^<]*<coordinates>([^<]+)<\\/coordinates>[^<]*<\\/Point>/gi;
            while ((match = pointPattern.exec(kmlContent)) !== null) {{
                const coordText = match[1].trim();
                const parts = coordText.split(',');
                if (parts.length >= 2) {{
                    const lon = parseFloat(parts[0]);
                    const lat = parseFloat(parts[1]);
                    if (!isNaN(lat) && !isNaN(lon)) {{
                        // Check if not already added
                        const exists = coords.some(c => c.lat === lat && c.lon === lon);
                        if (!exists) {{
                            coords.push({{ lat, lon }});
                        }}
                    }}
                }}
            }}

            return coords;
        }}

        async function confirmBatchGCPs() {{
            if (batchGpsCoords.length !== batchPoints.length) {{
                alert('Please load a KML file with matching coordinates first.');
                return;
            }}

            // Add all GCPs one by one
            for (let i = 0; i < batchPoints.length; i++) {{
                const pt = batchPoints[i];
                const gps = batchGpsCoords[i];
                const nameInput = document.getElementById(`batchName_${{i}}`);
                const name = nameInput ? nameInput.value.trim() || `GCP ${{i + 1}}` : `GCP ${{i + 1}}`;

                await fetch('/api/add_gcp', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        u: pt.u,
                        v: pt.v,
                        lat: gps.lat,
                        lon: gps.lon,
                        description: name,
                        accuracy: 'medium'
                    }})
                }})
                .then(r => r.json())
                .then(data => {{
                    gcps = data.gcps;
                    distribution = data.distribution;
                }});
            }}

            // Clean up batch mode
            batchMode = false;
            batchPoints = [];
            batchGpsCoords = [];
            clearBatchMarkers();
            updateBatchModeUI();
            updateUI();

            document.getElementById('batchModal').classList.remove('active');
            alert(`Added ${{batchPoints.length || gcps.length}} GCPs successfully!`);
        }}

        // Handle keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                closeModal();
                closeBatchModal();
            }}
            if (e.key === 'Enter' && document.getElementById('addGcpModal').classList.contains('active')) {{
                confirmAddGCP();
            }}
        }});

        // Add event listener for GPS input to show real-time parsing
        document.getElementById('gpsInput').addEventListener('input', updateParsedDisplay);

        // Initial UI update
        updateUI();
    </script>
</body>
</html>
"""
    return html


class GCPCaptureHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for GCP capture interface."""

    session: GCPCaptureWebSession = None
    output_path: str = None
    temp_dir: str = None
    frame_filename: str = None
    has_live_camera: bool = False  # True if using live camera, False if using static frame

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/':
            # Serve main HTML
            html = generate_capture_html(self.session, f'/{self.frame_filename}')
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())

        elif parsed.path == f'/{self.frame_filename}':
            # Serve the frame image
            frame_path = os.path.join(self.temp_dir, self.frame_filename)
            with open(frame_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)

        elif parsed.path == '/api/save':
            # Generate and serve YAML
            yaml_content = self.session.generate_yaml()

            # Save to file if output path specified
            if self.output_path:
                with open(self.output_path, 'w') as f:
                    f.write(yaml_content)
                print(f"\nSaved configuration to: {self.output_path}")
                print(f"  GCPs: {len(self.session.gcps)}")

            # Serve as download
            filename = os.path.basename(self.output_path) if self.output_path else f"gcps_{self.session.camera_name}.yaml"
            self.send_response(200)
            self.send_header('Content-type', 'application/x-yaml')
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(yaml_content.encode())

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        if parsed.path == '/api/add_gcp':
            data = json.loads(post_data)
            self.session.add_gcp(
                u=data['u'],
                v=data['v'],
                lat=data['lat'],
                lon=data['lon'],
                description=data.get('description', ''),
                accuracy=data.get('accuracy', 'medium')
            )
            self.send_json_response({
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography()
            })

        elif parsed.path == '/api/delete_gcp':
            data = json.loads(post_data)
            self.session.remove_gcp(data['index'])
            self.send_json_response({
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography()
            })

        elif parsed.path == '/api/clear_gcps':
            self.session.gcps.clear()
            self.session.current_homography = None
            self.session.last_reproj_errors = []
            self.session.inlier_mask = None
            self.send_json_response({
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography()
            })

        elif parsed.path == '/api/update_gcp_position':
            data = json.loads(post_data)
            success = self.session.update_gcp_position(
                index=data['index'],
                u=data['u'],
                v=data['v']
            )
            self.send_json_response({
                'success': success,
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography()
            })

        elif parsed.path == '/api/remove_outliers':
            result = self.session.remove_outliers()
            self.send_json_response({
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography(),
                'removed_count': result['removed_count'],
                'removed_indices': result['removed_indices'],
                'removed_descriptions': result.get('removed_descriptions', []),
                'remaining_gcps': result['remaining_gcps']
            })

        elif parsed.path == '/api/predict_error':
            data = json.loads(post_data)
            result = self.session.predict_new_gcp_error(
                u=data['u'],
                v=data['v'],
                lat=data['lat'],
                lon=data['lon']
            )
            self.send_json_response(result)

        elif parsed.path == '/api/load_yaml':
            data = json.loads(post_data)
            # Store current PTZ before loading
            current_ptz = self.session.ptz_status.copy() if self.session.ptz_status else None
            result = self.session.load_from_yaml(data['yaml_content'])
            self.send_json_response({
                'success': True,
                'gcps_loaded': result['gcps_loaded'],
                'warnings': result['warnings'],
                'gcps': self.session.gcps,
                'distribution': self.session.calculate_distribution(),
                'homography': self.session.update_homography(),
                'loaded_ptz': result.get('loaded_ptz'),
                'loaded_camera_name': result.get('loaded_camera_name'),
                'current_ptz': current_ptz,
                'has_live_camera': self.has_live_camera,
                'coordinate_system': result.get('coordinate_system')
            })

        elif parsed.path == '/api/move_camera':
            data = json.loads(post_data)
            ptz_position = data.get('ptz_position', {})
            wait_time = data.get('wait_time', 3.0)
            result = self.session.move_camera_to_ptz(ptz_position, wait_time)
            self.send_json_response({
                'success': result['success'],
                'message': result['message'],
                'ptz_status': self.session.ptz_status
            })

        else:
            self.send_error(404)

    def send_json_response(self, data):
        content = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)


def start_capture_server(
    session: GCPCaptureWebSession,
    output_path: str = None,
    port: int = 8765,
    auto_open: bool = True,
    has_live_camera: bool = False
):
    """Start the web server for GCP capture."""

    # Create temp directory for frame
    temp_dir = tempfile.mkdtemp(prefix='gcp_capture_')
    frame_filename = 'frame.jpg'
    frame_path = os.path.join(temp_dir, frame_filename)
    cv2.imwrite(frame_path, session.frame)

    # Configure handler
    GCPCaptureHandler.session = session
    GCPCaptureHandler.output_path = output_path
    GCPCaptureHandler.temp_dir = temp_dir
    GCPCaptureHandler.frame_filename = frame_filename
    GCPCaptureHandler.has_live_camera = has_live_camera

    # Find available port
    while True:
        try:
            server = socketserver.TCPServer(("", port), GCPCaptureHandler)
            break
        except OSError:
            port += 1

    url = f"http://localhost:{port}"
    print(f"\nGCP Capture server running at: {url}")
    print("Press Ctrl+C to stop\n")

    if auto_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
        # Cleanup temp files
        try:
            os.remove(frame_path)
            os.rmdir(temp_dir)
        except Exception:
            pass


def grab_frame_from_camera(camera_name: str, wait_time: float = 2.0) -> tuple:
    """
    Grab a frame from a camera and get PTZ status.

    Returns:
        Tuple of (frame, ptz_status)
    """
    if not CAMERA_CONFIG_AVAILABLE:
        raise RuntimeError(
            "Camera config not available. Set CAMERA_USERNAME and CAMERA_PASSWORD "
            "environment variables, or use --frame to load an existing image."
        )

    cam_info = get_camera_by_name(camera_name)
    if not cam_info:
        available = [c['name'] for c in CAMERAS]
        raise ValueError(
            f"Camera '{camera_name}' not found. Available: {', '.join(available)}"
        )

    rtsp_url = get_rtsp_url(camera_name)

    print(f"Connecting to camera '{camera_name}'...")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to connect to camera: {rtsp_url}")

    # Get PTZ status
    ptz_status = None
    if INTRINSICS_AVAILABLE:
        try:
            ptz_status = get_ptz_status(cam_info['ip'], USERNAME, PASSWORD)
            print(f"PTZ: pan={ptz_status['pan']:.1f}, tilt={ptz_status['tilt']:.1f}, zoom={ptz_status['zoom']:.1f}x")
        except Exception as e:
            print(f"Warning: Could not get PTZ status: {e}")

    # Grab frame
    print("Grabbing frame...")
    import time
    time.sleep(wait_time)  # Wait for camera to stabilize

    frame = None
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            break

    cap.release()

    if frame is None:
        raise RuntimeError("Failed to grab frame from camera")

    print(f"Frame captured: {frame.shape[1]}x{frame.shape[0]}")
    return frame, ptz_status


def main():
    parser = argparse.ArgumentParser(
        description='Web-based GCP Capture Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'camera',
        nargs='?',
        type=str,
        help='Camera name (e.g., Valte, Setram)'
    )
    parser.add_argument(
        '--frame', '-f',
        type=str,
        help='Path to an existing frame image (skips camera connection)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output YAML file path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (filename auto-generated)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8765,
        help='Server port (default: 8765)'
    )
    parser.add_argument(
        '--no-open',
        action='store_true',
        help='Do not automatically open browser'
    )
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List available cameras and exit'
    )

    args = parser.parse_args()

    if args.list_cameras:
        if CAMERA_CONFIG_AVAILABLE:
            print("Available cameras:")
            for cam in CAMERAS:
                print(f"  - {cam['name']} ({cam['ip']})")
        else:
            print("Camera config not available.")
        sys.exit(0)

    # Get frame
    frame = None
    ptz_status = None
    camera_name = args.camera or "Unknown"
    has_live_camera = False

    if args.frame:
        # Load from file
        frame = cv2.imread(args.frame)
        if frame is None:
            print(f"Error: Could not load image from {args.frame}")
            sys.exit(1)
        print(f"Loaded frame from {args.frame}: {frame.shape[1]}x{frame.shape[0]}")
        camera_name = Path(args.frame).stem

    elif args.camera:
        # Grab from camera
        try:
            frame, ptz_status = grab_frame_from_camera(args.camera)
            camera_name = args.camera
            has_live_camera = True
        except (RuntimeError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        print("\nError: Either camera name or --frame must be specified.")
        sys.exit(1)

    # Determine output path
    output_path = args.output
    if output_path is None and args.output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = str(Path(args.output_dir) / f"gcps_{camera_name}_{timestamp}.yaml")
    elif output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"gcps_{camera_name}_{timestamp}.yaml"

    # Create session and start server
    session = GCPCaptureWebSession(
        frame=frame,
        camera_name=camera_name,
        ptz_status=ptz_status
    )

    start_capture_server(
        session=session,
        output_path=output_path,
        port=args.port,
        auto_open=not args.no_open,
        has_live_camera=has_live_camera
    )


if __name__ == '__main__':
    main()
