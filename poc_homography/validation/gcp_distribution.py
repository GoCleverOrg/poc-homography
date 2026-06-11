"""Standalone GCP (Ground Control Point) spatial-distribution analysis.

This module is the canonical, freshly-authored implementation of the GCP
image-space distribution algorithm. It analyzes how well a set of GCPs is
distributed across the *image* (image_u/image_v) plane for an existing GCP YAML
file. It requires NO camera connection and NO frame image -- only the GCP
coordinates and the image dimensions.

No prior implementation exists to mirror, so the algorithm, formulas, and
quality thresholds are defined here and documented for future reference.

Scoring formula
----------------
Three sub-scores in [0, 1] are combined into a weighted overall score:

    coverage_score = min(coverage_ratio / GOOD_COVERAGE_RATIO, 1.0)
    quadrant_score = quadrant_count / 4.0
    spread_score   = (min(spread_x, 1.0) + min(spread_y, 1.0)) / 2.0

    score = 0.4 * coverage_score + 0.3 * quadrant_score + 0.3 * spread_score

where:
  * coverage_ratio = (convex-hull area of the points) / (image_width * image_height)
  * quadrant_count = number of the 4 image quadrants (TL, TR, BL, BR) that
    contain at least one point. The quadrant boundary is the image center
    (width / 2, height / 2).
  * spread_x = (max_u - min_u) / image_width
  * spread_y = (max_v - min_v) / image_height

Quality labels (freshly authored thresholds)
---------------------------------------------
    score >= 0.70 -> "Good"
    score >= 0.45 -> "Fair"
    otherwise     -> "Poor"
    fewer than MIN_GCPS_FOR_ANALYSIS points -> "Insufficient"

Quadrant index convention: 0 = TL, 1 = TR, 2 = BL, 3 = BR (quadrants_covered[i]).
The vertical axis follows image convention: V = 0 at the top, increasing downward.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import yaml

if TYPE_CHECKING:
    from pathlib import Path

# --- Thresholds and constants -------------------------------------------------

#: Convex-hull coverage ratio below this is considered too low -> warning.
MIN_COVERAGE_RATIO = 0.15
#: Convex-hull coverage ratio at/above this is considered good (caps coverage_score).
GOOD_COVERAGE_RATIO = 0.35
#: Fewer than this many covered quadrants -> warning.
MIN_QUADRANT_COVERAGE = 2
#: spread_x / spread_y below this -> recommendation to spread points out.
MIN_SPREAD = 0.15
#: Minimum number of points required for convex-hull / meaningful analysis.
MIN_GCPS_FOR_ANALYSIS = 3
#: Recommended GCP count; below this we suggest capturing more.
RECOMMENDED_GCP_COUNT = 10

#: Quadrant index -> short code (matches index convention 0=TL,1=TR,2=BL,3=BR).
_QUADRANT_CODES: tuple[str, str, str, str] = ("TL", "TR", "BL", "BR")
#: Quadrant short code -> human-readable name.
_QUADRANT_NAMES: dict[str, str] = {
    "TL": "top-left",
    "TR": "top-right",
    "BL": "bottom-left",
    "BR": "bottom-right",
}


@dataclass(frozen=True)
class DistributionMetrics:
    """Computed spatial-distribution metrics for a set of GCP image points."""

    n_gcps: int
    image_width: int
    image_height: int
    coverage_ratio: float
    quadrants_covered: tuple[bool, bool, bool, bool]
    quadrant_count: int
    spread_x: float
    spread_y: float
    coverage_score: float
    quadrant_score: float
    spread_score: float
    score: float
    quality: str
    warnings: list[str]
    recommendations: list[str]


# --- YAML loading -------------------------------------------------------------


def _find_feature_match(data: dict[str, Any]) -> dict[str, Any] | None:
    """Locate a ``feature_match`` block at top-level or under ``homography``."""
    fm = data.get("feature_match")
    if isinstance(fm, dict):
        return fm
    homography = data.get("homography")
    if isinstance(homography, dict):
        nested = homography.get("feature_match")
        if isinstance(nested, dict):
            return nested
    return None


def _normalize_gcp_list(gcps: Any) -> list[dict[str, Any]]:
    """Normalize a list-or-dict-of-lists GCP structure to a single list.

    Mirrors the normalization in ``validate_ground_control_points``: if a dict
    of named sets is provided, the first set is used.
    """
    if isinstance(gcps, list):
        return [g for g in gcps if isinstance(g, dict)]
    if isinstance(gcps, dict):
        if not gcps:
            return []
        first_value = next(iter(gcps.values()))
        if isinstance(first_value, list):
            return [g for g in first_value if isinstance(g, dict)]
        return []
    return []


def _extract_dims(feature_match: dict[str, Any]) -> tuple[int | None, int | None]:
    """Extract image dimensions from a feature_match block (with ctx fallback)."""
    width = feature_match.get("image_width")
    height = feature_match.get("image_height")
    if width is None or height is None:
        ctx = feature_match.get("camera_capture_context")
        if isinstance(ctx, dict):
            if width is None:
                width = ctx.get("image_width")
            if height is None:
                height = ctx.get("image_height")
    width_int = int(width) if isinstance(width, (int, float)) else None
    height_int = int(height) if isinstance(height, (int, float)) else None
    return width_int, height_int


def load_image_points(
    yaml_path: Path,
) -> tuple[list[tuple[float, float]], int | None, int | None]:
    """Load image-space GCP points from a GCP YAML file.

    Supports two formats:

    1. Simple: top-level ``gcps`` key whose value is a list of GCP dicts or a
       dict of named sets (first set used).
    2. Complex: a ``feature_match`` block (top-level or under ``homography``)
       containing ``ground_control_points`` of the same shape. Image dimensions
       may be read from ``feature_match.image_width/height`` or from
       ``feature_match.camera_capture_context.image_width/height``.

    Args:
        yaml_path: Path to the GCP YAML file.

    Returns:
        A tuple of (points, width, height) where ``points`` is a list of
        (image_u, image_v) tuples and width/height are the dimensions found in
        metadata (or ``None`` if absent).

    Raises:
        ValueError: If the format is unrecognized, or if no GCP has image
            coordinates.
    """
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "Invalid GCP format. Expected top-level 'gcps' list or a "
            "'feature_match.ground_control_points' structure."
        )

    gcp_list: list[dict[str, Any]]
    width: int | None = None
    height: int | None = None

    if "gcps" in data:
        gcp_list = _normalize_gcp_list(data["gcps"])
    else:
        feature_match = _find_feature_match(data)
        if feature_match is None or "ground_control_points" not in feature_match:
            raise ValueError(
                "Invalid GCP format. Expected top-level 'gcps' list or a "
                "'feature_match.ground_control_points' structure."
            )
        gcp_list = _normalize_gcp_list(feature_match["ground_control_points"])
        width, height = _extract_dims(feature_match)

    points: list[tuple[float, float]] = []
    for gcp in gcp_list:
        image_u = gcp.get("image_u")
        image_v = gcp.get("image_v")
        if isinstance(image_u, (int, float)) and isinstance(image_v, (int, float)):
            points.append((float(image_u), float(image_v)))

    if not points:
        raise ValueError(f"No GCPs with image coordinates (image_u/image_v) found in {yaml_path}")

    return points, width, height


# --- Distribution calculation -------------------------------------------------


def _clamp01(value: float) -> float:
    """Clamp a float to the [0, 1] range."""
    return max(0.0, min(1.0, value))


def calculate_distribution(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> DistributionMetrics:
    """Compute spatial-distribution metrics for image-space GCP points.

    See the module docstring for the full formula and threshold definitions.

    Args:
        points: List of (image_u, image_v) tuples.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        A populated :class:`DistributionMetrics` instance.
    """
    n_gcps = len(points)

    if n_gcps < MIN_GCPS_FOR_ANALYSIS:
        # Compute quadrant coverage for context, but force scores to zero.
        quadrants = _compute_quadrants(points, image_width, image_height)
        return DistributionMetrics(
            n_gcps=n_gcps,
            image_width=image_width,
            image_height=image_height,
            coverage_ratio=0.0,
            quadrants_covered=quadrants,
            quadrant_count=sum(quadrants),
            spread_x=0.0,
            spread_y=0.0,
            coverage_score=0.0,
            quadrant_score=0.0,
            spread_score=0.0,
            score=0.0,
            quality="Insufficient",
            warnings=[
                f"Need at least {MIN_GCPS_FOR_ANALYSIS} GCPs for distribution "
                f"analysis (found {n_gcps})."
            ],
            recommendations=["Capture more GCPs."],
        )

    pts = np.asarray(points, dtype=np.float32)

    # Coverage ratio via convex hull area.
    hull = cv2.convexHull(pts)
    hull_area = float(cv2.contourArea(hull))
    total_area = float(image_width) * float(image_height)
    coverage_ratio = _clamp01(hull_area / total_area) if total_area > 0 else 0.0

    # Quadrant coverage.
    quadrants = _compute_quadrants(points, image_width, image_height)
    quadrant_count = sum(quadrants)

    # Spread.
    us = pts[:, 0]
    vs = pts[:, 1]
    spread_x = _clamp01(float(us.max() - us.min()) / image_width) if image_width > 0 else 0.0
    spread_y = _clamp01(float(vs.max() - vs.min()) / image_height) if image_height > 0 else 0.0

    # Sub-scores.
    coverage_score = min(coverage_ratio / GOOD_COVERAGE_RATIO, 1.0)
    quadrant_score = quadrant_count / 4.0
    spread_score = (min(spread_x, 1.0) + min(spread_y, 1.0)) / 2.0
    score = 0.4 * coverage_score + 0.3 * quadrant_score + 0.3 * spread_score

    if score >= 0.70:
        quality = "Good"
    elif score >= 0.45:
        quality = "Fair"
    else:
        quality = "Poor"

    warnings: list[str] = []
    if coverage_ratio < MIN_COVERAGE_RATIO:
        warnings.append(
            f"Low convex-hull coverage: {round(coverage_ratio * 100)}% of the image "
            f"(below the {round(MIN_COVERAGE_RATIO * 100)}% minimum)."
        )
    if quadrant_count < MIN_QUADRANT_COVERAGE:
        warnings.append(
            f"Only {quadrant_count} of 4 quadrants covered (need at least {MIN_QUADRANT_COVERAGE})."
        )

    recommendations: list[str] = []
    for index, covered in enumerate(quadrants):
        if not covered:
            name = _QUADRANT_NAMES[_QUADRANT_CODES[index]]
            recommendations.append(f"Add GCPs in the {name} quadrant")
    if spread_x < MIN_SPREAD:
        recommendations.append(
            f"Increase horizontal spread (currently {round(spread_x * 100)}% of width)"
        )
    if spread_y < MIN_SPREAD:
        recommendations.append(
            f"Increase vertical spread (currently {round(spread_y * 100)}% of height)"
        )
    if n_gcps < RECOMMENDED_GCP_COUNT:
        recommendations.append(
            f"Consider adding more GCPs (have {n_gcps}, recommend >={RECOMMENDED_GCP_COUNT})"
        )

    return DistributionMetrics(
        n_gcps=n_gcps,
        image_width=image_width,
        image_height=image_height,
        coverage_ratio=coverage_ratio,
        quadrants_covered=quadrants,
        quadrant_count=quadrant_count,
        spread_x=spread_x,
        spread_y=spread_y,
        coverage_score=coverage_score,
        quadrant_score=quadrant_score,
        spread_score=spread_score,
        score=score,
        quality=quality,
        warnings=warnings,
        recommendations=recommendations,
    )


def _compute_quadrants(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> tuple[bool, bool, bool, bool]:
    """Return which quadrants contain at least one point.

    Index convention: 0=TL, 1=TR, 2=BL, 3=BR. The quadrant index for a point is
    ``(1 if u >= center_u else 0) + (2 if v >= center_v else 0)``.
    """
    center_u = image_width / 2.0
    center_v = image_height / 2.0
    covered = [False, False, False, False]
    for u, v in points:
        q = (1 if u >= center_u else 0) + (2 if v >= center_v else 0)
        covered[q] = True
    return (covered[0], covered[1], covered[2], covered[3])


# --- Reporting ----------------------------------------------------------------


def render_ascii_scatter(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
    grid_w: int = 40,
    grid_h: int = 12,
) -> str:
    """Render an ASCII scatter plot of points in image space.

    The grid is drawn with a border box. Occupied cells are marked with ``*``,
    empty cells with a space, and a ``+`` crosshair marks the center axes where
    no point occupies the cell. V = 0 maps to the top row.
    """
    grid_w = max(1, grid_w)
    grid_h = max(1, grid_h)

    grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]

    center_col = grid_w // 2
    center_row = grid_h // 2
    for r in range(grid_h):
        grid[r][center_col] = "+"
    for c in range(grid_w):
        grid[center_row][c] = "+"

    if image_width > 0 and image_height > 0:
        for u, v in points:
            col = int(u / image_width * (grid_w - 1)) if grid_w > 1 else 0
            row = int(v / image_height * (grid_h - 1)) if grid_h > 1 else 0
            col = max(0, min(grid_w - 1, col))
            row = max(0, min(grid_h - 1, row))
            grid[row][col] = "*"

    top = "+" + "-" * grid_w + "+"
    lines = [top]
    for r in range(grid_h):
        lines.append("|" + "".join(grid[r]) + "|")
    lines.append(top)
    return "\n".join(lines)


def render_text_report(yaml_name: str, metrics: DistributionMetrics) -> str:
    """Render a plain-text distribution report."""
    checks = " ".join("[✓]" if covered else "[✗]" for covered in metrics.quadrants_covered)
    coverage_pct = round(metrics.coverage_ratio * 100)
    spread_x_pct = round(metrics.spread_x * 100)
    spread_y_pct = round(metrics.spread_y * 100)

    lines = [
        "GCP Distribution Analysis",
        "=========================",
        f"File: {yaml_name}",
        f"GCPs: {metrics.n_gcps}",
        f"Image: {metrics.image_width}x{metrics.image_height}",
        "",
        "Distribution Metrics:",
        f"  Coverage Ratio: {coverage_pct}% [{metrics.quality.upper()}]",
        f"  Quadrants: {checks}  ({metrics.quadrant_count}/4 covered)",
        f"  Spread X: {metrics.spread_x:.2f} ({spread_x_pct}% of width)",
        f"  Spread Y: {metrics.spread_y:.2f} ({spread_y_pct}% of height)",
        f"  Overall Score: {metrics.score:.2f}",
        "",
        f"Quality: {metrics.quality.upper()}",
        "",
    ]

    # The scatter is rendered from the metrics' dimensions; points themselves
    # are not stored on the metrics, so the caller passes an empty plot when
    # unavailable. Here we render a header line and let render() include it.
    lines.append("Warnings:")
    if metrics.warnings:
        lines.extend(f"  • {w}" for w in metrics.warnings)
    else:
        lines.append("  (none)")

    lines.append("Recommendations:")
    if metrics.recommendations:
        lines.extend(f"  • {r}" for r in metrics.recommendations)
    else:
        lines.append("  (none)")

    return "\n".join(lines)


def render_full_text_report(
    yaml_name: str,
    metrics: DistributionMetrics,
    points: list[tuple[float, float]],
) -> str:
    """Render the text report including the ASCII scatter plot."""
    base = render_text_report(yaml_name, metrics)
    scatter = render_ascii_scatter(points, metrics.image_width, metrics.image_height)
    # Insert the scatter after the "Quality:" block (before "Warnings:").
    marker = "\nWarnings:"
    if marker in base:
        head, tail = base.split(marker, 1)
        return f"{head}\n{scatter}\n{marker.lstrip(chr(10))}{tail}"
    return f"{base}\n{scatter}"


def render_html_report(
    yaml_name: str,
    metrics: DistributionMetrics,
    points: list[tuple[float, float]],
) -> str:
    """Render a minimal, self-contained HTML distribution report.

    The HTML contains a 2x2 quadrant grid (green = covered, gray = empty) and an
    SVG scatter plot of the points in image space (V increasing downward,
    aspect ratio preserved). No external assets are referenced.
    """
    safe_name = _html.escape(yaml_name)

    # SVG scatter: scale to a max dimension while preserving aspect ratio.
    max_dim = 400.0
    width = metrics.image_width if metrics.image_width > 0 else 1
    height = metrics.image_height if metrics.image_height > 0 else 1
    scale = max_dim / max(width, height)
    svg_w = width * scale
    svg_h = height * scale

    circles = "".join(
        f'<circle cx="{u * scale:.1f}" cy="{v * scale:.1f}" r="4" '
        f'fill="#1565c0" fill-opacity="0.7" />'
        for u, v in points
    )
    svg = (
        f'<svg width="{svg_w:.0f}" height="{svg_h:.0f}" '
        f'viewBox="0 0 {svg_w:.1f} {svg_h:.1f}" '
        f'style="border:1px solid #999;background:#fafafa">'
        f'<line x1="{svg_w / 2:.1f}" y1="0" x2="{svg_w / 2:.1f}" y2="{svg_h:.1f}" '
        f'stroke="#ccc" stroke-dasharray="4" />'
        f'<line x1="0" y1="{svg_h / 2:.1f}" x2="{svg_w:.1f}" y2="{svg_h / 2:.1f}" '
        f'stroke="#ccc" stroke-dasharray="4" />'
        f"{circles}</svg>"
    )

    # 2x2 quadrant grid. Layout rows: [TL, TR], [BL, BR].
    cell_order = [(0, "TL"), (1, "TR"), (2, "BL"), (3, "BR")]
    cells = []
    for index, code in cell_order:
        covered = metrics.quadrants_covered[index]
        color = "#4caf50" if covered else "#bdbdbd"
        label = _QUADRANT_NAMES[code]
        mark = "✓" if covered else "✗"
        cells.append(
            f'<div style="background:{color};color:#fff;padding:18px;'
            f'text-align:center;font-weight:bold">{mark} {label}</div>'
        )
    quadrant_grid = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;'
        f'max-width:320px">{"".join(cells)}</div>'
    )

    warnings_html = (
        "".join(f"<li>{_html.escape(w)}</li>" for w in metrics.warnings) or "<li>(none)</li>"
    )
    recs_html = (
        "".join(f"<li>{_html.escape(r)}</li>" for r in metrics.recommendations) or "<li>(none)</li>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>GCP Distribution Analysis - {safe_name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; color: #222; }}
  h1 {{ font-size: 1.4rem; }}
  table.metrics td {{ padding: 2px 12px 2px 0; }}
  .section {{ margin-top: 24px; }}
</style>
</head>
<body>
<h1>GCP Distribution Analysis</h1>
<p><strong>File:</strong> {safe_name}<br />
<strong>GCPs:</strong> {metrics.n_gcps}<br />
<strong>Image:</strong> {metrics.image_width}x{metrics.image_height}<br />
<strong>Quality:</strong> {_html.escape(metrics.quality)}</p>

<div class="section">
<h2>Metrics</h2>
<table class="metrics">
<tr><td>Coverage Ratio</td><td>{metrics.coverage_ratio * 100:.0f}%</td></tr>
<tr><td>Quadrants Covered</td><td>{metrics.quadrant_count} / 4</td></tr>
<tr><td>Spread X</td><td>{metrics.spread_x:.2f}</td></tr>
<tr><td>Spread Y</td><td>{metrics.spread_y:.2f}</td></tr>
<tr><td>Overall Score</td><td>{metrics.score:.2f}</td></tr>
</table>
</div>

<div class="section">
<h2>Quadrant Coverage</h2>
{quadrant_grid}
</div>

<div class="section">
<h2>Scatter (image space, V down)</h2>
{svg}
</div>

<div class="section">
<h2>Warnings</h2>
<ul>{warnings_html}</ul>
</div>

<div class="section">
<h2>Recommendations</h2>
<ul>{recs_html}</ul>
</div>
</body>
</html>
"""
