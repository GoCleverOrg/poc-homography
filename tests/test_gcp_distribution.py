#!/usr/bin/env python3
"""Unit tests for the standalone GCP spatial-distribution analyzer.

Covers the pure logic (load_image_points, calculate_distribution, rendering)
and the `hom gcp distribution` Typer command.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typer.testing import CliRunner

from poc_homography.cli.main import app
from poc_homography.validation.gcp_distribution import (
    GOOD_COVERAGE_RATIO,
    DistributionMetrics,
    calculate_distribution,
    load_image_points,
    render_ascii_scatter,
    render_text_report,
)


def _write_yaml(content: str) -> Path:
    """Write content to a temp YAML file and return its path."""
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    handle.write(content)
    handle.close()
    return Path(handle.name)


class TestLoadImagePoints(unittest.TestCase):
    """Tests for parsing GCP YAML into image points."""

    def test_simple_gcps_list(self):
        """Simple top-level gcps list is parsed."""
        path = _write_yaml(
            "gcps:\n"
            "  - map_id: m1\n    image_u: 100\n    image_v: 200\n"
            "  - map_id: m2\n    image_u: 300\n    image_v: 400\n"
        )
        try:
            points, width, height = load_image_points(path)
        finally:
            path.unlink()
        self.assertEqual(points, [(100.0, 200.0), (300.0, 400.0)])
        self.assertIsNone(width)
        self.assertIsNone(height)

    def test_complex_feature_match_with_ctx_dims(self):
        """feature_match.ground_control_points with camera_capture_context dims."""
        path = _write_yaml(
            "feature_match:\n"
            "  camera_capture_context:\n"
            "    image_width: 1920\n"
            "    image_height: 1080\n"
            "  ground_control_points:\n"
            "    - map_id: m1\n      image_u: 10\n      image_v: 20\n"
            "    - map_id: m2\n      image_u: 30\n      image_v: 40\n"
        )
        try:
            points, width, height = load_image_points(path)
        finally:
            path.unlink()
        self.assertEqual(points, [(10.0, 20.0), (30.0, 40.0)])
        self.assertEqual(width, 1920)
        self.assertEqual(height, 1080)

    def test_homography_wrapped_feature_match(self):
        """feature_match nested under a homography wrapper is found."""
        path = _write_yaml(
            "homography:\n"
            "  feature_match:\n"
            "    image_width: 800\n"
            "    image_height: 600\n"
            "    ground_control_points:\n"
            "      - map_id: m1\n        image_u: 1\n        image_v: 2\n"
            "      - map_id: m2\n        image_u: 3\n        image_v: 4\n"
        )
        try:
            points, width, height = load_image_points(path)
        finally:
            path.unlink()
        self.assertEqual(points, [(1.0, 2.0), (3.0, 4.0)])
        self.assertEqual(width, 800)
        self.assertEqual(height, 600)

    def test_dict_of_named_sets_uses_first_set(self):
        """A dict of named GCP sets uses the first set."""
        path = _write_yaml(
            "gcps:\n"
            "  set_a:\n"
            "    - map_id: m1\n      image_u: 5\n      image_v: 6\n"
            "    - map_id: m2\n      image_u: 7\n      image_v: 8\n"
            "  set_b:\n"
            "    - map_id: m9\n      image_u: 99\n      image_v: 99\n"
        )
        try:
            points, _w, _h = load_image_points(path)
        finally:
            path.unlink()
        self.assertEqual(points, [(5.0, 6.0), (7.0, 8.0)])

    def test_missing_image_coords_raises(self):
        """GCPs without image_u/image_v raise ValueError."""
        path = _write_yaml("gcps:\n  - map_id: m1\n    map_pixel_x: 1\n    map_pixel_y: 2\n")
        try:
            with self.assertRaises(ValueError) as ctx:
                load_image_points(path)
        finally:
            path.unlink()
        self.assertIn("image coordinates", str(ctx.exception))

    def test_invalid_format_raises(self):
        """A YAML lacking gcps and feature_match raises ValueError."""
        path = _write_yaml("something_else:\n  foo: bar\n")
        try:
            with self.assertRaises(ValueError) as ctx:
                load_image_points(path)
        finally:
            path.unlink()
        self.assertIn("Invalid GCP format", str(ctx.exception))


class TestCalculateDistribution(unittest.TestCase):
    """Tests for the distribution metric computation."""

    def test_well_spread_covers_all_quadrants(self):
        """A well-spread point set covers all 4 quadrants with a decent score."""
        points = [
            (100.0, 100.0),
            (900.0, 100.0),
            (100.0, 500.0),
            (900.0, 500.0),
            (500.0, 300.0),
        ]
        metrics = calculate_distribution(points, 1000, 600)
        self.assertEqual(metrics.quadrant_count, 4)
        self.assertTrue(all(metrics.quadrants_covered))
        self.assertGreater(metrics.score, 0.4)
        self.assertIn(metrics.quality, ("Good", "Fair"))

    def test_clustered_in_one_quadrant(self):
        """A clustered set yields quadrant_count 1, warnings, low coverage."""
        points = [(10.0, 10.0), (20.0, 15.0), (15.0, 20.0), (12.0, 18.0)]
        metrics = calculate_distribution(points, 1000, 600)
        self.assertEqual(metrics.quadrant_count, 1)
        self.assertTrue(metrics.warnings)
        self.assertLess(metrics.coverage_ratio, 0.15)

    def test_insufficient_points(self):
        """Fewer than MIN_GCPS_FOR_ANALYSIS points -> Insufficient, zero scores."""
        points = [(10.0, 10.0), (20.0, 20.0)]
        metrics = calculate_distribution(points, 1000, 600)
        self.assertEqual(metrics.quality, "Insufficient")
        self.assertEqual(metrics.score, 0.0)
        self.assertEqual(metrics.coverage_score, 0.0)
        self.assertEqual(metrics.quadrant_score, 0.0)
        self.assertEqual(metrics.spread_score, 0.0)
        self.assertTrue(metrics.warnings)

    def test_known_square_coverage_and_spread(self):
        """Verify coverage_ratio and spread math on a centered square box."""
        # 1000x1000 image, square from (250,250) to (750,750): side 500.
        points = [
            (250.0, 250.0),
            (750.0, 250.0),
            (250.0, 750.0),
            (750.0, 750.0),
        ]
        metrics = calculate_distribution(points, 1000, 1000)
        # Hull area = 500*500 = 250000; total = 1_000_000 -> 0.25.
        self.assertAlmostEqual(metrics.coverage_ratio, 0.25, places=3)
        # Spread = 500/1000 = 0.5 each.
        self.assertAlmostEqual(metrics.spread_x, 0.5, places=3)
        self.assertAlmostEqual(metrics.spread_y, 0.5, places=3)

    def test_score_formula(self):
        """score == 0.4*cov + 0.3*quad + 0.3*spread for a constructed case."""
        points = [
            (250.0, 250.0),
            (750.0, 250.0),
            (250.0, 750.0),
            (750.0, 750.0),
        ]
        m = calculate_distribution(points, 1000, 1000)
        expected = 0.4 * m.coverage_score + 0.3 * m.quadrant_score + 0.3 * m.spread_score
        self.assertAlmostEqual(m.score, expected, places=9)
        # Sanity: coverage_score = coverage_ratio / GOOD_COVERAGE_RATIO (capped).
        self.assertAlmostEqual(
            m.coverage_score,
            min(m.coverage_ratio / GOOD_COVERAGE_RATIO, 1.0),
            places=9,
        )

    def test_coverage_and_quadrant_warnings_trigger(self):
        """Low coverage (<15%) and few quadrants (<2) each trigger a warning."""
        points = [(5.0, 5.0), (8.0, 6.0), (6.0, 9.0)]
        metrics = calculate_distribution(points, 1000, 1000)
        text = " ".join(metrics.warnings)
        self.assertIn("coverage", text.lower())
        self.assertIn("quadrant", text.lower())

    def test_spread_recommendations_trigger(self):
        """spread_x<0.15 and spread_y<0.15 each produce a recommendation."""
        points = [(5.0, 5.0), (8.0, 6.0), (6.0, 9.0)]
        metrics = calculate_distribution(points, 1000, 1000)
        rec_text = " ".join(metrics.recommendations).lower()
        self.assertIn("horizontal spread", rec_text)
        self.assertIn("vertical spread", rec_text)


class TestRendering(unittest.TestCase):
    """Tests for text rendering helpers."""

    def _metrics(self) -> DistributionMetrics:
        points = [
            (100.0, 100.0),
            (900.0, 100.0),
            (100.0, 500.0),
            (900.0, 500.0),
        ]
        return calculate_distribution(points, 1000, 600)

    def test_text_report_contains_key_fields(self):
        """The text report includes the title, count, and quality."""
        metrics = self._metrics()
        report = render_text_report("my_gcps.yaml", metrics)
        self.assertIn("GCP Distribution Analysis", report)
        self.assertIn(f"GCPs: {metrics.n_gcps}", report)
        self.assertIn(metrics.quality.upper(), report)

    def test_ascii_scatter_robust_to_empty(self):
        """The scatter renderer handles an empty point list."""
        out = render_ascii_scatter([], 1000, 600)
        self.assertTrue(out)
        self.assertIn("+", out)


class TestDistributionCLI(unittest.TestCase):
    """Tests for the `hom gcp distribution` Typer command."""

    def setUp(self):
        self.runner = CliRunner()

    def test_with_yaml_dims_succeeds(self):
        """A YAML with dims runs to exit 0 and prints the report."""
        path = _write_yaml(
            "feature_match:\n"
            "  image_width: 1000\n"
            "  image_height: 600\n"
            "  ground_control_points:\n"
            "    - map_id: m1\n      image_u: 100\n      image_v: 100\n"
            "    - map_id: m2\n      image_u: 900\n      image_v: 100\n"
            "    - map_id: m3\n      image_u: 100\n      image_v: 500\n"
            "    - map_id: m4\n      image_u: 900\n      image_v: 500\n"
        )
        try:
            result = self.runner.invoke(app, ["gcp", "distribution", "-g", str(path)])
        finally:
            path.unlink()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("GCP Distribution", result.output)

    def test_missing_dims_fails(self):
        """A YAML lacking dims and no --width/--height fails mentioning dimensions."""
        path = _write_yaml(
            "gcps:\n"
            "  - map_id: m1\n    image_u: 100\n    image_v: 100\n"
            "  - map_id: m2\n    image_u: 900\n    image_v: 100\n"
            "  - map_id: m3\n    image_u: 100\n    image_v: 500\n"
        )
        try:
            result = self.runner.invoke(app, ["gcp", "distribution", "-g", str(path)])
        finally:
            path.unlink()
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("dimensions", result.output.lower())

    def test_cli_dims_override(self):
        """Passing --width/--height for a dimensionless YAML succeeds."""
        path = _write_yaml(
            "gcps:\n"
            "  - map_id: m1\n    image_u: 100\n    image_v: 100\n"
            "  - map_id: m2\n    image_u: 900\n    image_v: 100\n"
            "  - map_id: m3\n    image_u: 500\n    image_v: 500\n"
        )
        try:
            result = self.runner.invoke(
                app,
                [
                    "gcp",
                    "distribution",
                    "-g",
                    str(path),
                    "--width",
                    "1000",
                    "--height",
                    "600",
                ],
            )
        finally:
            path.unlink()
        self.assertEqual(result.exit_code, 0, result.output)

    def test_html_output_written(self):
        """--html --output writes an HTML file containing '<html'."""
        path = _write_yaml(
            "feature_match:\n"
            "  image_width: 1000\n"
            "  image_height: 600\n"
            "  ground_control_points:\n"
            "    - map_id: m1\n      image_u: 100\n      image_v: 100\n"
            "    - map_id: m2\n      image_u: 900\n      image_v: 100\n"
            "    - map_id: m3\n      image_u: 100\n      image_v: 500\n"
        )
        out_dir = tempfile.mkdtemp()
        out_path = Path(out_dir) / "report.html"
        try:
            result = self.runner.invoke(
                app,
                [
                    "gcp",
                    "distribution",
                    "-g",
                    str(path),
                    "--html",
                    "--output",
                    str(out_path),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(out_path.exists())
            content = out_path.read_text(encoding="utf-8")
            self.assertIn("<html", content)
        finally:
            path.unlink()
            if out_path.exists():
                out_path.unlink()
            os.rmdir(out_dir)


if __name__ == "__main__":
    unittest.main()
