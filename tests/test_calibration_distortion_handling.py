"""
Tests for Issue #135: Remove image undistortion from calibration pipeline.

Verifies that the calibration pipeline operates entirely in distorted image space
with original K matrix, using cv2.projectPoints() and cv2.solvePnP() with
distortion coefficients rather than undistorting frames.
"""

import re

import pytest


class TestNoUndistortionInCalibration:
    """Tests verifying no cv2.undistort or getOptimalNewCameraMatrix calls in calibration."""

    def test_no_cv2_undistort_in_unified_gcp_tool(self):
        """Verify cv2.undistort() is not called in unified_gcp_tool.py."""
        with open("tools/unified_gcp_tool.py") as f:
            content = f.read()

        # Check for cv2.undistort calls (not in comments)
        lines = content.split("\n")
        undistort_calls = []
        for i, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "cv2.undistort(" in line:
                undistort_calls.append((i, line.strip()))

        assert len(undistort_calls) == 0, (
            "Found cv2.undistort() calls in unified_gcp_tool.py:\n"
            + "\n".join(f"  Line {ln}: {code}" for ln, code in undistort_calls)
        )

    def test_no_getOptimalNewCameraMatrix_in_unified_gcp_tool(self):
        """Verify getOptimalNewCameraMatrix() is not called in unified_gcp_tool.py."""
        with open("tools/unified_gcp_tool.py") as f:
            content = f.read()

        # Check for getOptimalNewCameraMatrix calls (not in comments)
        lines = content.split("\n")
        optimal_k_calls = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "getOptimalNewCameraMatrix" in line:
                optimal_k_calls.append((i, line.strip()))

        assert len(optimal_k_calls) == 0, (
            "Found getOptimalNewCameraMatrix() calls in unified_gcp_tool.py:\n"
            + "\n".join(f"  Line {ln}: {code}" for ln, code in optimal_k_calls)
        )


class TestDistortionCoefficientsUsage:
    """Tests verifying proper use of distortion coefficients in OpenCV functions."""

    def test_solvePnP_uses_dist_coeffs(self):
        """Verify cv2.solvePnP calls include dist_coeffs parameter."""
        with open("tools/unified_gcp_tool.py") as f:
            content = f.read()

        # Find all solvePnP calls
        solvepnp_pattern = r"cv2\.solvePnP(?:Ransac)?\s*\("
        matches = list(re.finditer(solvepnp_pattern, content))

        assert len(matches) > 0, "No cv2.solvePnP calls found in unified_gcp_tool.py"

        # For each solvePnP call, verify dist_coeffs is passed
        for match in matches:
            start = match.start()
            # Find the closing parenthesis
            paren_count = 0
            end = start
            for i, char in enumerate(content[start:], start):
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count == 0:
                        end = i + 1
                        break

            call_text = content[start:end]
            assert "dist_coeffs" in call_text, (
                f"cv2.solvePnP call at position {start} does not include dist_coeffs:\n{call_text[:200]}"
            )

    def test_projectPoints_uses_dist_coeffs(self):
        """Verify cv2.projectPoints calls include dist_coeffs parameter."""
        with open("tools/unified_gcp_tool.py") as f:
            content = f.read()

        # Find all projectPoints calls
        project_pattern = r"cv2\.projectPoints\s*\("
        matches = list(re.finditer(project_pattern, content))

        assert len(matches) > 0, "No cv2.projectPoints calls found in unified_gcp_tool.py"

        # For each projectPoints call, verify dist_coeffs is passed
        for match in matches:
            start = match.start()
            # Find the closing parenthesis
            paren_count = 0
            end = start
            for i, char in enumerate(content[start:], start):
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count == 0:
                        end = i + 1
                        break

            call_text = content[start:end]
            assert "dist_coeffs" in call_text, (
                f"cv2.projectPoints call at position {start} does not include dist_coeffs:\n{call_text[:200]}"
            )


class TestUndistortedFlagHandling:
    """Tests verifying camera_params['undistorted'] is always False."""

    def test_undistorted_flag_always_false_first_location(self):
        """Verify undistorted flag is set to False at first frame capture location."""
        with open("tools/unified_gcp_tool.py") as f:
            content = f.read()

        # Check that undistorted is set to False (not True)
        assert "camera_params['undistorted'] = False" in content, (
            "camera_params['undistorted'] should be explicitly set to False"
        )

        # Verify no location sets undistorted to True
        assert "camera_params['undistorted'] = True" not in content, (
            "camera_params['undistorted'] should never be set to True"
        )


class TestNoUndistortPointInCalibration:
    """Tests verifying CameraGeometry.undistort_point is not called during calibration."""

    def test_no_undistort_point_calls_in_unified_gcp_tool(self):
        """Verify undistort_point() is not called in unified_gcp_tool.py."""
        with open("tools/unified_gcp_tool.py") as f:
            content = f.read()

        # Check for undistort_point or undistort_points calls (not in comments)
        lines = content.split("\n")
        undistort_calls = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "undistort_point" in line or "undistort_points" in line:
                undistort_calls.append((i, line.strip()))

        assert len(undistort_calls) == 0, (
            "Found undistort_point/undistort_points calls in unified_gcp_tool.py:\n"
            + "\n".join(f"  Line {ln}: {code}" for ln, code in undistort_calls)
        )


class TestDistortionCoefficientFormat:
    """Tests verifying distortion coefficients use 5-parameter format [k1, k2, p1, p2, k3]."""

    def test_dist_coeffs_five_parameters(self):
        """Verify distortion coefficient arrays use 5 parameters."""
        with open("tools/unified_gcp_tool.py") as f:
            content = f.read()

        # Check for np.zeros(5) for dist_coeffs initialization
        # This ensures we're using the full [k1, k2, p1, p2, k3] format
        assert "np.zeros(5)" in content, (
            "Distortion coefficients should be initialized with 5 parameters [k1, k2, p1, p2, k3]"
        )

    def test_k3_coefficient_loaded(self):
        """Verify k3 coefficient is loaded from camera config."""
        with open("tools/unified_gcp_tool.py") as f:
            content = f.read()

        # Check that k3 is being loaded
        assert "k3 = cam_config.get('k3'" in content, (
            "k3 distortion coefficient should be loaded from camera config"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
