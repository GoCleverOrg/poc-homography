#!/usr/bin/env python3
"""#550 inter-pose homography bbox-warp re-identification across zoom — PoC harness.

Validates the depth-free inter-pose homography

    H(a->b) = K(zoom_b) . R_b . R_a^T . K(zoom_a)^-1

as a bbox-warp re-identification primitive across PTZ zoom cycles, using only
intrinsics and relative rotation (no depth, no world frame, no extrinsics, no
second camera). This is the Layer-1 core validation PoC behind PRD
`docs/prd-spatial-world-model.md` (FR-L1-2).

Three independent parts:

  (1) SYNTHETIC  -- construct pose pairs analytically, warp a bbox via H, and
      compare against an INDEPENDENT forward projection of the same world rays
      at the target pose. Proves the implementation is arithmetically correct
      (catches transpose/convention/ordering bugs) before touching real pixels.
      No hardware, no data: this is the CI-adjacent path.

  (2) FIELD      -- load the real #530 Hikvision zoom-cycle stack (the
      `zoom_{up,down}_<z>x.jpg` frames saved by `poc/ptz_530/harness.py` into
      `/tmp/ptz-530`). The sweep holds pan/tilt fixed, so the inter-pose
      homography reduces to H = K(z_b).K(z_a)^-1 (a scaling about the principal
      point) -- the exact "re-identify a static region across a zoom cycle" use
      case. For each adjacent zoom pair we re-find static features by normalized
      cross-correlation, measure the re-projection error of the depth-free warp,
      and decompose out the focal-scale error to isolate the geometric residual.
      Manual hardware gate: SKIPs cleanly (exit 0) when the footage is absent.

  (3) NODAL      -- the field sweep is pure zoom (delta-rotation == 0), so the
      ROTATIONAL nodal-offset parallax is identically zero in that footage. We
      therefore quantify it ANALYTICALLY: the pixel parallax a rotation-axis
      offset induces at a given zoom / target distance / re-centering angle.
      This is a model prediction, clearly labelled, never a fabricated field
      measurement.

Run (canonical, from the poc-homography repo root, using its existing venv):

    uv run python poc_550_interpose_homography.py
    uv run python poc_550_interpose_homography.py --footage-dir /tmp/ptz-530

Camera conventions (intrinsics K(zoom), rotation R(pan,tilt,roll)) are reused
verbatim from #531 (`poc_homography/camera_geometry.py`) when that package is
importable, and otherwise from an in-file vendored port of the same constants so
the script also runs standalone. When both are present the two are asserted
numerically identical, guarding against drift.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Camera conventions (reused from #531 / poc_homography.camera_geometry).
# --------------------------------------------------------------------------- #
# Hikvision DS-2DF8425IX-AELW datasheet linear-focal model (the same constants
# baked into poc_homography.camera_geometry.get_intrinsics).
BASE_FOCAL_MM = 5.9
SENSOR_WIDTH_MM = 6.78


def _vendored_intrinsics(
    zoom: float, w: int = 1920, h: int = 1080, sensor_mm: float = SENSOR_WIDTH_MM
) -> np.ndarray:
    """K(zoom) per the #531 linear-focal model: f_px = base_mm * zoom * (W/sensor_mm)."""
    f_px = BASE_FOCAL_MM * zoom * (w / sensor_mm)
    return np.array([[f_px, 0.0, w / 2.0], [0.0, f_px, h / 2.0], [0.0, 0.0, 1.0]])


def _vendored_rotation(pan_deg: float, tilt_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """R(pan,tilt,roll) = R_tilt . R_roll . R_base . R_pan, with R_base the #531
    world->camera base [[1,0,0],[0,0,-1],[0,1,0]] (poc _get_rotation_matrix_static)."""
    pr, tr, rr = math.radians(pan_deg), math.radians(tilt_deg), math.radians(roll_deg)
    r_base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    rz_pan = np.array(
        [[math.cos(pr), -math.sin(pr), 0.0], [math.sin(pr), math.cos(pr), 0.0], [0.0, 0.0, 1.0]]
    )
    rz_roll = np.array(
        [[math.cos(rr), -math.sin(rr), 0.0], [math.sin(rr), math.cos(rr), 0.0], [0.0, 0.0, 1.0]]
    )
    rx_tilt = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(tr), -math.sin(tr)], [0.0, math.sin(tr), math.cos(tr)]]
    )
    return rx_tilt @ rz_roll @ r_base @ rz_pan


try:
    from poc_homography.camera_geometry import CameraGeometry

    _HAVE_REAL = True
    CONV_SOURCE = "poc_homography.camera_geometry (#531)"
except Exception:  # pragma: no cover - standalone (e.g. mira/poc copy) fallback
    CameraGeometry = None  # type: ignore[assignment]
    _HAVE_REAL = False
    CONV_SOURCE = "vendored verbatim port of #531 conventions"


def intrinsics(
    zoom: float, w: int = 1920, h: int = 1080, sensor_mm: float = SENSOR_WIDTH_MM
) -> np.ndarray:
    if _HAVE_REAL:
        return np.asarray(CameraGeometry.get_intrinsics(zoom, w, h, sensor_mm), dtype=float)
    return _vendored_intrinsics(zoom, w, h, sensor_mm)


def rotation(pan_deg: float, tilt_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    if _HAVE_REAL:
        return np.asarray(
            CameraGeometry._get_rotation_matrix_static(pan_deg, tilt_deg, roll_deg), dtype=float
        )
    return _vendored_rotation(pan_deg, tilt_deg, roll_deg)


def assert_conventions_consistent() -> None:
    """If both the real module and the vendored port are available, they must agree."""
    if not _HAVE_REAL:
        return
    for zoom in (1.0, 2.5, 8.0, 25.0):
        assert np.allclose(
            np.asarray(CameraGeometry.get_intrinsics(zoom)), _vendored_intrinsics(zoom)
        ), f"intrinsics drift at zoom={zoom}"
    for pan, tilt in ((0.0, 0.0), (30.0, 10.0), (-45.0, 25.0)):
        assert np.allclose(
            np.asarray(CameraGeometry._get_rotation_matrix_static(pan, tilt, 0.0)),
            _vendored_rotation(pan, tilt),
        ), f"rotation drift at {pan},{tilt}"


# --------------------------------------------------------------------------- #
# Layer-1 geometric primitives (the future `mira-ptz-geometry` surface).
# --------------------------------------------------------------------------- #
def interpose_homography(pose_a: tuple, pose_b: tuple, w: int = 1920, h: int = 1080) -> np.ndarray:
    """H(a->b) = K(zoom_b) . R_b . R_a^T . K(zoom_a)^-1. Poses are (pan, tilt, zoom) deg/x."""
    pan_a, tilt_a, zoom_a = pose_a
    pan_b, tilt_b, zoom_b = pose_b
    k_a, k_b = intrinsics(zoom_a, w, h), intrinsics(zoom_b, w, h)
    r_a, r_b = rotation(pan_a, tilt_a), rotation(pan_b, tilt_b)
    return k_b @ r_b @ r_a.T @ np.linalg.inv(k_a)


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N,2) pixel coords, returning (N,2)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    out = (h @ homo.T).T
    return out[:, :2] / out[:, 2:3]


def aabb(pts: np.ndarray) -> tuple:
    """Axis-aligned bounding box (x0,y0,x1,y1) of an (N,2) point array, as floats."""
    pts = np.asarray(pts, dtype=float)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def warp_bbox(h: np.ndarray, bbox: tuple) -> tuple:
    """Warp the four corners of (x0,y0,x1,y1) and return the AABB of the result."""
    x0, y0, x1, y1 = bbox
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return aabb(apply_h(h, corners))


def bbox_center(bbox: tuple) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])


def project_world_ray(
    zoom: float, pan: float, tilt: float, ray: np.ndarray, w: int = 1920, h: int = 1080
) -> tuple[np.ndarray, float]:
    """Forward-project a world ray direction to a pixel at pose (pan,tilt,zoom).

    Returns (pixel (2,), camera-frame depth). depth > 0 == in front of camera.
    Depth-free: only the ray *direction* matters, so a unit world direction is
    enough (no camera position / scene depth)."""
    cam = rotation(pan, tilt) @ np.asarray(ray, dtype=float)
    pix = intrinsics(zoom, w, h) @ cam
    return pix[:2] / pix[2], float(cam[2])


def backproject_pixel(
    zoom: float, pan: float, tilt: float, px: np.ndarray, w: int = 1920, h: int = 1080
) -> np.ndarray:
    """Inverse of project_world_ray: pixel -> unit world ray direction at the pose."""
    cam = np.linalg.inv(intrinsics(zoom, w, h)) @ np.array([px[0], px[1], 1.0])
    ray = rotation(pan, tilt).T @ cam
    return ray / np.linalg.norm(ray)


# --------------------------------------------------------------------------- #
# (1) Synthetic validation.
# --------------------------------------------------------------------------- #
# >=5 pose pairs spanning the full 1x..8x zoom range, with pan deltas in
# {0, +-10, +-30} and tilt deltas in {0, +-10}, per the issue's requirements.
SYNTHETIC_PAIRS = [
    ((0.0, 10.0, 1.0), (0.0, 10.0, 2.0)),  # pure zoom 1x -> 2x
    ((0.0, 10.0, 1.0), (10.0, 10.0, 4.0)),  # +10 pan, zoom 4x
    ((0.0, 10.0, 1.0), (30.0, 20.0, 8.0)),  # +30 pan, +10 tilt, zoom 8x
    ((20.0, 30.0, 2.0), (-10.0, 20.0, 4.0)),  # -30 pan, -10 tilt, zoom 2x -> 4x
    ((0.0, 0.0, 1.0), (0.0, 10.0, 8.0)),  # +10 tilt, zoom 8x
    ((-15.0, 25.0, 1.0), (15.0, 15.0, 6.0)),  # +30 pan, -10 tilt, zoom 6x
    ((10.0, 10.0, 1.0), (0.0, 20.0, 4.0)),  # -10 pan, +10 tilt, zoom 4x
]


def run_synthetic(w: int = 1920, h: int = 1080) -> dict:
    """Warp a synthetic bbox + probe points via H and compare to an independent
    forward projection of the same world rays at pose b. Sub-pixel residual proves
    the homography composition (and the K/R conventions) are arithmetically exact."""
    print(
        "\n# (1) SYNTHETIC  cols: idx|pose_a(pan,tilt,zoom)|pose_b|zoom_ratio|"
        "n_pts|max_pt_err_px|bbox_center_err_px",
        flush=True,
    )
    # probe pixels expressed as fractions of the frame, kept off-center so the
    # warp is non-trivial (a centred point is a fixed point of a pure-zoom H).
    fracs = [(0.5, 0.5), (0.62, 0.55), (0.45, 0.6), (0.55, 0.42), (0.4, 0.46)]
    base = np.array([[fx * w, fy * h] for fx, fy in fracs])
    worst_pt = 0.0
    worst_bbox = 0.0
    rows = []
    for idx, (pose_a, pose_b) in enumerate(SYNTHETIC_PAIRS):
        h_ab = interpose_homography(pose_a, pose_b, w, h)
        pt_errs = []
        src_pts = []
        for px in base:
            ray = backproject_pixel(pose_a[2], pose_a[0], pose_a[1], px, w, h)
            truth, depth_b = project_world_ray(pose_b[2], pose_b[0], pose_b[1], ray, w, h)
            _, depth_a = project_world_ray(pose_a[2], pose_a[0], pose_a[1], ray, w, h)
            if depth_a <= 0 or depth_b <= 0:
                continue  # ray behind camera at one pose: not a valid probe
            warped = apply_h(h_ab, px)[0]
            pt_errs.append(float(np.linalg.norm(warped - truth)))
            src_pts.append(px)
        if not pt_errs:
            continue
        # Exercise the warp_bbox primitive: source bbox = AABB of the in-frame
        # probes at pose a, warped via warp_bbox (the future mira-ptz-geometry
        # surface), compared to an INDEPENDENT forward projection of the same
        # four corner rays at pose b.
        src_bbox = aabb(np.array(src_pts))
        warped_bbox = warp_bbox(h_ab, src_bbox)
        x0, y0, x1, y1 = src_bbox
        truth_corners = []
        for corner in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            cray = backproject_pixel(pose_a[2], pose_a[0], pose_a[1], np.array(corner), w, h)
            ct, _ = project_world_ray(pose_b[2], pose_b[0], pose_b[1], cray, w, h)
            truth_corners.append(ct)
        truth_bbox = aabb(np.array(truth_corners))
        bbox_err = float(np.linalg.norm(bbox_center(warped_bbox) - bbox_center(truth_bbox)))
        max_pt = max(pt_errs)
        worst_pt = max(worst_pt, max_pt)
        worst_bbox = max(worst_bbox, bbox_err)
        ratio = pose_b[2] / pose_a[2]
        rows.append((idx, pose_a, pose_b, ratio, len(pt_errs), max_pt, bbox_err))
        print(
            f"{idx}|{pose_a}|{pose_b}|{ratio:.2f}x|{len(pt_errs)}|{max_pt:.3e}|{bbox_err:.3e}",
            flush=True,
        )
    passed = worst_bbox < 0.5 and worst_pt < 0.5
    print(
        f"# (1) summary: pairs={len(rows)} worst_point_err={worst_pt:.3e}px "
        f"worst_bbox_center_err={worst_bbox:.3e}px  DoD(<0.5px): "
        f"{'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return {"rows": rows, "worst_pt": worst_pt, "worst_bbox": worst_bbox, "passed": passed}


# --------------------------------------------------------------------------- #
# (2) Field validation against the #530 zoom-cycle footage.
# --------------------------------------------------------------------------- #
def _load_zoom_stack(footage_dir: Path, leg: str):
    """Return [(zoom_x, gray_image)] sorted ascending for one leg, or []."""
    import cv2

    out = []
    for p in footage_dir.glob(f"zoom_{leg}_*x.jpg"):
        try:
            z = float(p.name.split("_")[2][:-5])  # zoom_up_6.25x.jpg -> 6.25
        except (IndexError, ValueError):
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out.append((z, img))
    return sorted(out, key=lambda t: t[0])


def _match_pair(img_a, img_b, ratio_cmd: float, principal: np.ndarray, ncc_min: float = 0.5):
    """Re-find static features of frame a inside the (more zoomed) frame b.

    Pure-zoom inter-pose H predicts p_b = c + ratio*(p_a - c). We seed feature
    detection in the central 1/ratio region of a (the part that survives the
    zoom-in into b's FOV), upscale each patch by ratio_cmd to b's scale, and NCC
    search near the predicted location. Returns matched (p_a, p_b, ncc)."""
    import cv2

    h, w = img_a.shape
    cx, cy = principal
    # central ROI of a that remains within b's narrower FOV (with margin).
    half_w, half_h = w / (2.0 * ratio_cmd) * 0.85, h / (2.0 * ratio_cmd) * 0.85
    x0, y0 = int(cx - half_w), int(cy - half_h)
    x1, y1 = int(cx + half_w), int(cy + half_h)
    mask = np.zeros_like(img_a)
    mask[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = 255
    corners = cv2.goodFeaturesToTrack(
        img_a, maxCorners=80, qualityLevel=0.06, minDistance=40, mask=mask
    )
    if corners is None:
        return []
    patch = 28  # half-size of the source template window in a
    search = 90  # half-size of the search window in b around the prediction
    matches = []
    for c in corners.reshape(-1, 2):
        pa = np.array([float(c[0]), float(c[1])])
        sx0, sy0 = int(pa[0] - patch), int(pa[1] - patch)
        if sx0 < 0 or sy0 < 0 or sx0 + 2 * patch >= w or sy0 + 2 * patch >= h:
            continue
        tmpl = img_a[sy0 : sy0 + 2 * patch, sx0 : sx0 + 2 * patch]
        # upscale the template to b's apparent scale
        ts = max(8, round(2 * patch * ratio_cmd))
        tmpl_rs = cv2.resize(tmpl, (ts, ts), interpolation=cv2.INTER_CUBIC)
        pred = principal + ratio_cmd * (pa - principal)
        rx0, ry0 = int(pred[0] - ts / 2 - search), int(pred[1] - ts / 2 - search)
        rx1, ry1 = int(pred[0] - ts / 2 + search), int(pred[1] - ts / 2 + search)
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(w - ts, rx1), min(h - ts, ry1)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        region = img_b[ry0 : ry1 + ts, rx0 : rx1 + ts]
        if region.shape[0] < ts or region.shape[1] < ts:
            continue
        res = cv2.matchTemplate(region, tmpl_rs, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < ncc_min:
            continue
        pb = np.array([rx0 + max_loc[0] + ts / 2.0, ry0 + max_loc[1] + ts / 2.0])
        matches.append((pa, pb, float(max_val)))
    return matches


def _fit_scale_about(principal: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> float:
    """Least-squares scale s minimising ||(pb-c) - s(pa-c)|| about fixed centre c."""
    da, db = pa - principal, pb - principal
    denom = float(np.sum(da * da))
    return float(np.sum(da * db) / denom) if denom > 0 else float("nan")


def _fit_similarity(pa: np.ndarray, pb: np.ndarray) -> tuple[float, np.ndarray]:
    """Least-squares scale s + translation t for pb ~= s*pa + t (no rotation).

    Letting the centre float absorbs principal-point / nodal drift, so the
    residual is the irreducible non-similarity floor (lens distortion + match
    noise). Returns (s, t (2,))."""
    n = len(pa)
    # unknowns x = [s, tx, ty]; rows: [pa.x, 1, 0] and [pa.y, 0, 1]
    a = np.zeros((2 * n, 3))
    a[0::2, 0], a[0::2, 1] = pa[:, 0], 1.0
    a[1::2, 0], a[1::2, 2] = pa[:, 1], 1.0
    b = pb.reshape(-1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(sol[0]), sol[1:3]


def run_field(footage_dir: Path, min_matches: int = 6, ncc_min: float = 0.5) -> dict:
    """Measure the depth-free zoom warp on the real #530 Hikvision zoom stack."""
    try:
        import cv2  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"# (2) FIELD: SKIP -- OpenCV unavailable ({exc})", flush=True)
        return {"status": "skip", "reason": "no-opencv"}

    if not footage_dir.is_dir():
        print(
            f"# (2) FIELD: SKIP -- footage dir {footage_dir} absent "
            "(manual hardware gate; run poc/ptz_530/harness.py zoom)",
            flush=True,
        )
        return {"status": "skip", "reason": "no-footage"}

    legs = {leg: _load_zoom_stack(footage_dir, leg) for leg in ("up", "down")}
    if not any(legs.values()):
        print(f"# (2) FIELD: SKIP -- no zoom_*.jpg frames under {footage_dir}", flush=True)
        return {"status": "skip", "reason": "no-frames"}

    # frame resolution + principal point from the first available frame.
    sample = next(stack[0][1] for stack in legs.values() if stack)
    fh, fw = sample.shape
    principal = np.array([fw / 2.0, fh / 2.0])
    print(f"\n# (2) FIELD  footage={footage_dir} res={fw}x{fh} principal={principal.tolist()}")
    print(
        "# pure-zoom sweep (delta-pan=delta-tilt=0) => H(a->b)=K(z_b).K(z_a)^-1; "
        "rotational nodal parallax is identically 0 in this footage (see part 3)."
    )
    print("# reproj_err = depth-free warp w/ COMMANDED zoom (total model error);")
    print("# scalefit_res = residual after best-fit scale about centre (removes focal-model")
    print("#   error; leaves principal/nodal drift + distortion + match noise);")
    print("# simfit_res = residual after best-fit scale+translation (distortion + noise floor);")
    print("#   (scalefit_res - simfit_res) ~ principal-point / nodal-shift drift.")
    print(
        "# cols: leg|z_from|z_to|ratio_cmd|n|med_ncc|reproj_err_px|scalefit_res_px|"
        "simfit_res_px|meas_scale|focal_dev_pct|status",
        flush=True,
    )

    per_level: dict = {}
    for leg in ("up", "down"):
        stack = legs[leg]
        for i in range(1, len(stack)):
            z_a, img_a = stack[i - 1]
            z_b, img_b = stack[i]
            ratio_cmd = z_b / z_a
            m = _match_pair(img_a, img_b, ratio_cmd, principal, ncc_min)
            if len(m) < min_matches:
                print(
                    f"{leg}|{z_a:g}|{z_b:g}|{ratio_cmd:.3f}|{len(m)}|-|-|-|-|-|scene-limited",
                    flush=True,
                )
                per_level.setdefault(z_b, []).append(
                    {"leg": leg, "status": "scene-limited", "n": len(m), "z_from": z_a}
                )
                continue
            pa = np.array([x[0] for x in m])
            pb = np.array([x[1] for x in m])
            ncc = np.array([x[2] for x in m])
            pred_cmd = principal + ratio_cmd * (pa - principal)
            reproj = np.linalg.norm(pb - pred_cmd, axis=1)
            s_meas = _fit_scale_about(principal, pa, pb)
            pred_scale = principal + s_meas * (pa - principal)
            scalefit = np.linalg.norm(pb - pred_scale, axis=1)
            s_sim, t_sim = _fit_similarity(pa, pb)
            pred_sim = s_sim * pa + t_sim
            simfit = np.linalg.norm(pb - pred_sim, axis=1)
            focal_dev = (s_meas / ratio_cmd - 1.0) * 100.0
            reproj_med = float(np.median(reproj))
            scalefit_med, simfit_med = float(np.median(scalefit)), float(np.median(simfit))
            print(
                f"{leg}|{z_a:g}|{z_b:g}|{ratio_cmd:.3f}|{len(m)}|{np.median(ncc):.3f}|"
                f"{reproj_med:.2f}|{scalefit_med:.2f}|{simfit_med:.2f}|{s_meas:.3f}|"
                f"{focal_dev:+.1f}|ok",
                flush=True,
            )
            per_level.setdefault(z_b, []).append(
                {
                    "leg": leg,
                    "status": "ok",
                    "n": len(m),
                    "z_from": z_a,
                    "med_ncc": float(np.median(ncc)),
                    "reproj_px": reproj_med,
                    "scalefit_px": scalefit_med,
                    "simfit_px": simfit_med,
                    "meas_scale": s_meas,
                    "focal_dev_pct": focal_dev,
                }
            )

    levels = sorted({z for stack in legs.values() for z, _ in stack})
    ok_levels = [z for z in per_level if any(r["status"] == "ok" for r in per_level[z])]
    print(
        f"# (2) summary: zoom levels in footage = {[f'{z:g}x' for z in levels]}; "
        f"measurable adjacent-pair targets = {sorted(f'{z:g}x' for z in ok_levels)}",
        flush=True,
    )
    return {
        "status": "ok",
        "levels": levels,
        "per_level": per_level,
        "principal": principal.tolist(),
        "res": (fw, fh),
    }


# --------------------------------------------------------------------------- #
# (3) Analytic rotational nodal-offset parallax.
# --------------------------------------------------------------------------- #
def nodal_parallax_px(
    offset_m: float, distance_m: float, recenter_deg: float, zoom: float, w: int = 2560
) -> float:
    """Pixel parallax induced by a rotation-axis offset (nodal offset).

    A rotation by `recenter_deg` about an axis `offset_m` from the optical centre
    translates the centre by |t| = 2*offset*sin(recenter/2). For a feature at
    range `distance_m`, the worst-case (perpendicular) parallax angle is |t|/D
    rad, i.e. f_px(zoom) * |t| / D pixels."""
    t = 2.0 * offset_m * math.sin(math.radians(recenter_deg) / 2.0)
    f_px = BASE_FOCAL_MM * zoom * (w / SENSOR_WIDTH_MM)
    return f_px * t / distance_m


def run_nodal(w: int = 2560, max_zoom: float = 25.0) -> dict:
    """Tabulate the analytic rotational nodal-offset parallax across plausible
    rig offsets, terminal-scale distances, and re-centering angles."""
    print(
        f"\n# (3) NODAL (analytic rotational parallax; field sweep is pure zoom "
        f"=> empirical rotational residual == 0). frame width={w}px"
    )
    f_px_8 = BASE_FOCAL_MM * 8 * (w / SENSOR_WIDTH_MM)
    f_px_max = BASE_FOCAL_MM * max_zoom * (w / SENSOR_WIDTH_MM)
    print(f"# f_px(8x)={f_px_8:.0f}  f_px({max_zoom:g}x)={f_px_max:.0f}")
    print(
        f"# cols: offset_m|distance_m|recenter_deg|parallax_px@8x|parallax_px@{max_zoom:g}x",
        flush=True,
    )
    offsets = [0.01, 0.03, 0.05, 0.10]
    distances = [20.0, 50.0, 100.0, 200.0]
    recenters = [2.0, 10.0, 30.0]
    rows = []
    for off in offsets:
        for dist in distances:
            for rec in recenters:
                p8 = nodal_parallax_px(off, dist, rec, 8.0, w)
                pmax = nodal_parallax_px(off, dist, rec, max_zoom, w)
                rows.append((off, dist, rec, p8, pmax))
                print(f"{off:.2f}|{dist:.0f}|{rec:.0f}|{p8:.2f}|{pmax:.2f}", flush=True)
    # representative "typical terminal" case for the TL;DR
    typ = nodal_parallax_px(0.05, 50.0, 10.0, max_zoom, w)
    print(
        f"# (3) representative (offset=5cm, D=50m, recenter=10deg) @ {max_zoom:g}x "
        f"= {typ:.1f}px  (PRD assoc_residual_px tolerance = 24px)",
        flush=True,
    )
    return {"rows": rows, "max_zoom": max_zoom, "w": w, "representative_max_zoom_px": typ}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--footage-dir",
        default="/tmp/ptz-530",
        help="dir with #530 zoom_{up,down}_<z>x.jpg frames (default /tmp/ptz-530)",
    )
    ap.add_argument(
        "--ncc-min",
        type=float,
        default=0.5,
        help="min normalized cross-correlation to accept a feature match",
    )
    ap.add_argument(
        "--min-matches",
        type=int,
        default=6,
        help="min matched features for a measurable adjacent-pair",
    )
    ap.add_argument(
        "--max-zoom",
        type=float,
        default=25.0,
        help="max zoom for the analytic nodal-offset table (footage max)",
    )
    args = ap.parse_args()

    print(f"# #550 inter-pose homography PoC  conventions: {CONV_SOURCE}")
    assert_conventions_consistent()

    synth = run_synthetic()
    field = run_field(Path(args.footage_dir), args.min_matches, args.ncc_min)
    nodal = run_nodal(max_zoom=args.max_zoom)

    print(
        "\n# DONE  synthetic={} field={} nodal_rows={}".format(
            "PASS" if synth["passed"] else "FAIL", field["status"], len(nodal["rows"])
        )
    )
    # the script completing is the DoD; synthetic correctness is a hard gate.
    return 0 if synth["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
