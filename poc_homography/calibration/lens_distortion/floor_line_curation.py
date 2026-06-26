"""Floor-line curation via dominant orientation families.

A genuine painted-floor scene is dominated by a few orientation families: the
parking-bay dividers and boundaries run in one or two directions (and their
perspective-projected near-parallels). Scattered clutter and the 3-D edges of
*cars* parked on the apron rarely align with those families. Because a single
radial distortion straightens all true floor lines simultaneously, feeding the
solver off-family / car edges only biases it.

This module curates a line set down to its dominant orientation families using
a length-weighted chord-angle histogram (mirroring the bucketing in
:mod:`~poc_homography.calibration.lens_distortion.visibility`). It iterates --
recompute families from the surviving set, re-filter -- until the kept set is
stable or a small iteration cap is reached. Pure and offline: no detection, no
hardware, no heavy dependencies.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poc_homography.calibration.lens_distortion.models import CameraLine

logger = logging.getLogger(__name__)

# Histogram bucket width in degrees over the 0..180 chord-angle range. Mirrors
# the 30-degree orientation bucketing used by visibility, but finer so adjacent
# families stay distinct.
_BUCKET_DEG = 10.0
_NUM_BUCKETS = round(180.0 / _BUCKET_DEG)


def _chord_angle_deg(line: CameraLine) -> float:
    """Chord angle of a line in [0, 180) degrees (orientation, not direction)."""
    sx, sy = line.start_pixel
    ex, ey = line.end_pixel
    return math.degrees(math.atan2(ey - sy, ex - sx)) % 180.0


def _circular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two orientations on a 180 circle."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _dominant_families(
    lines: list[CameraLine],
    *,
    min_family_fraction: float,
) -> list[float]:
    """Length-weighted dominant chord-angle families (bucket centres, degrees).

    Builds a length-weighted histogram over ``_BUCKET_DEG`` buckets, then keeps
    every bucket holding at least ``min_family_fraction`` of the total weight.
    The representative angle of a kept family is its weight-weighted mean angle
    (so the family centre tracks the real lines, not the bucket edge).
    """
    weights = [0.0] * _NUM_BUCKETS
    angle_acc = [0.0] * _NUM_BUCKETS
    total = 0.0
    for line in lines:
        w = line.length_pixels
        if w <= 0.0:
            continue
        angle = _chord_angle_deg(line)
        idx = min(int(angle // _BUCKET_DEG), _NUM_BUCKETS - 1)
        weights[idx] += w
        angle_acc[idx] += w * angle
        total += w

    if total <= 0.0:
        return []

    families: list[float] = []
    for idx in range(_NUM_BUCKETS):
        if weights[idx] / total >= min_family_fraction:
            families.append(angle_acc[idx] / weights[idx])
    return families


def curate_floor_lines(
    lines: list[CameraLine],
    image_w: float,
    image_h: float,
    *,
    angle_tol_deg: float = 12.0,
    min_family_fraction: float = 0.15,
    min_lines: int = 6,
    max_iterations: int = 5,
) -> list[CameraLine]:
    """Curate lines down to their dominant orientation families.

    Computes dominant chord-angle families from a length-weighted histogram and
    keeps only lines whose chord angle is within ``angle_tol_deg`` of a dominant
    family. The process iterates -- recompute families from the kept set, then
    re-filter -- until the kept set stops changing (convergence) or
    ``max_iterations`` is reached. This strips scattered clutter and off-family
    car edges while preserving the true floor lines.

    The curation is conservative: if filtering would drop below ``min_lines``,
    the previous (larger) set is returned unchanged so a thin-but-real scene is
    never starved. ``image_w`` / ``image_h`` are accepted for a stable signature
    and possible future spatial weighting; the orientation logic does not need
    them.

    Args:
        lines: Candidate camera lines (any orientation, any quality).
        image_w: Image width in pixels (currently unused; see above).
        image_h: Image height in pixels (currently unused; see above).
        angle_tol_deg: Max chord-angle deviation from a dominant family.
        min_family_fraction: Min length-weighted fraction for a dominant family.
        min_lines: Floor on the kept-set size; never curate below this.
        max_iterations: Convergence cap on the recompute/re-filter loop.

    Returns:
        The curated list of camera lines (a subset of ``lines``).
    """
    if len(lines) <= min_lines:
        return list(lines)

    current = list(lines)
    for _ in range(max_iterations):
        families = _dominant_families(current, min_family_fraction=min_family_fraction)
        if not families:
            # No dominant orientation emerged; leave the set untouched.
            return current

        kept = [
            line
            for line in current
            if any(
                _circular_diff_deg(_chord_angle_deg(line), fam) <= angle_tol_deg for fam in families
            )
        ]

        if len(kept) < min_lines:
            # Filtering too aggressive for this scene -- keep the prior set.
            return current
        if len(kept) == len(current):
            # Converged: no line changed family membership this pass.
            return kept
        current = kept

    return current
