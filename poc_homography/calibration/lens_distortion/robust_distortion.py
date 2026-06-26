"""Robust (IRLS-style) distortion solve that rejects inconsistent lines.

A single radial distortion straightens every true floor line simultaneously. A
3-D car edge -- or any line whose curvature is inconsistent with that one
distortion -- cannot be straightened by the same coefficients, so it shows up as
a large per-line residual. This module wraps :class:`DistortionSolver` in a
reweighting/rejection loop:

    solve -> read rmse_per_line -> if the worst line exceeds a robust threshold
    (max(reject_min_px, median + scale * MAD)) and enough lines remain, drop the
    worst offender(s) and re-solve -> repeat.

The loop stops when no line is rejected, the kept set hits ``min_lines``, or the
recovered ``k1`` stabilises (``|dk1| < k1_stable_tol``). Floor lines (mutually
consistent with one distortion) survive; car edges get rejected. Pure / offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from poc_homography.calibration.lens_distortion.distortion_solver import DistortionSolver

if TYPE_CHECKING:
    from poc_homography.calibration.lens_distortion.distortion_solver import (
        SolverConfig,
        SolverResult,
    )
    from poc_homography.calibration.lens_distortion.models import CameraLine
    from poc_homography.domain.vo.lens_distortion import LensDistortion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobustSolveResult:
    """Outcome of a robust distortion solve.

    Attributes:
        result: The final :class:`SolverResult` over the surviving lines.
        kept_lines: The lines that survived rejection (the inliers).
        rejected_lines: The lines rejected as distortion-inconsistent.
        iterations: Number of solve passes performed.
    """

    result: SolverResult
    kept_lines: tuple[CameraLine, ...]
    rejected_lines: tuple[CameraLine, ...]
    iterations: int


def _reject_threshold(rmse_per_line: list[float], *, min_px: float, mad_scale: float) -> float:
    """Robust per-line rejection threshold: max(min_px, median + scale*MAD)."""
    arr = np.asarray(rmse_per_line, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return max(min_px, median + mad_scale * mad)


def solve_robust(
    lines: list[CameraLine],
    intrinsic_matrix: np.ndarray,
    config: SolverConfig,
    *,
    initial_guess: LensDistortion | None = None,
    min_lines: int = 6,
    reject_min_px: float = 2.0,
    reject_mad_scale: float = 3.0,
    k1_stable_tol: float = 1e-4,
    max_passes: int = 12,
) -> RobustSolveResult:
    """Solve for distortion, iteratively rejecting inconsistent lines.

    Args:
        lines: Candidate camera lines (already curated / curvature-filtered).
        intrinsic_matrix: 3x3 camera intrinsic matrix K.
        config: Solver configuration (must carry the tight physical bounds).
        initial_guess: Initial distortion (zeros when None).
        min_lines: Stop rejecting once this many lines remain.
        reject_min_px: Floor for the per-line rejection threshold.
        reject_mad_scale: MAD multiplier in the rejection threshold.
        k1_stable_tol: Stop when ``|k1_new - k1_prev|`` drops below this.
        max_passes: Hard cap on solve passes.

    Returns:
        A :class:`RobustSolveResult` with the final fit and kept/rejected lines.

    Raises:
        ValueError: If fewer than ``min_lines`` usable lines are provided, or the
            underlying solver raises (propagated).
    """
    if len(lines) < min_lines:
        raise ValueError(f"robust solve needs >= {min_lines} lines, got {len(lines)}")

    solver = DistortionSolver(config)
    kept = list(lines)
    rejected: list[CameraLine] = []
    prev_k1: float | None = None
    result = solver.solve(kept, intrinsic_matrix, initial_guess)

    passes = 1
    while passes < max_passes:
        k1 = float(result.distortion.k1)
        rmse = result.rmse_per_line
        # rmse_per_line is aligned with the solver's *post-curvature-filter*
        # lines; when their counts agree we can map residuals back to lines.
        if len(rmse) != len(kept):
            logger.debug(
                "robust solve: rmse/line count mismatch (%d vs %d); stopping rejection",
                len(rmse),
                len(kept),
            )
            break

        threshold = _reject_threshold(rmse, min_px=reject_min_px, mad_scale=reject_mad_scale)
        worst_idx = int(np.argmax(rmse))
        worst = rmse[worst_idx]

        coeff_stable = prev_k1 is not None and abs(k1 - prev_k1) < k1_stable_tol
        if worst <= threshold or len(kept) <= min_lines or coeff_stable:
            break

        rejected.append(kept.pop(worst_idx))
        prev_k1 = k1
        result = solver.solve(kept, intrinsic_matrix, initial_guess)
        passes += 1

    logger.info(
        "robust solve: kept %d line(s), rejected %d, k1=%.4f over %d pass(es)",
        len(kept),
        len(rejected),
        float(result.distortion.k1),
        passes,
    )
    return RobustSolveResult(
        result=result,
        kept_lines=tuple(kept),
        rejected_lines=tuple(rejected),
        iterations=passes,
    )
