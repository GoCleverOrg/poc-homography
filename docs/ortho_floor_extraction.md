# Ortho floor extraction PoC — process, tools, results

End-to-end exploration of how to take the icozee site orthophoto and produce a "floor-only" raster (cars and other above-ground occluders removed), then how to extract ground lines / markings from it. All experimental code lives in a sibling worktree of the `mira` repo:

```
/Users/vasco/workspace/goclever/mira/.worktrees/poc-ortho-lines-534/poc/ortho_lines_534/
```

The branch is `poc/ortho-lines-534` (not pushed). Every commit referenced below lives there.

## 1. Goal

Two coupled goals:

1. **Floor extraction.** From the icozee orthophoto produce a clean raster of just the ground (asphalt, paint stripes, lane markings, rails) with all above-ground occluders (parked cars, boom/crane structures, the small building on-site) removed or in-painted.
2. **Line/marker detection on the floor.** From the floor raster produce structured outputs that downstream consumers can use: a per-pixel "is line" mask, a vectorised set of line segments, and ideally a region partition into `parking_row` / `road` / `aisle` / `other` classes.

Both are inputs to the auto-calibration / homography refinement task that this repo owns.

## 2. Tooling overview

| layer | tools | role |
|---|---|---|
| tile cutting + recomposition | OpenCV, pyinvoke | slice the 3327×2731 ortho into 500×500 PNG tiles; later stitch processed tiles back into a full-res JPEG + 1024² preview |
| upscaling | Replicate · `prunaai/p-image-upscale` (target=8 MPx, factor=2, enhance_realism) | super-resolve each 500×500 tile to ~2896×2896 so per-car features (paint stripes, car silhouettes) are tens of pixels wide |
| monocular depth | Replicate · `chenxwh/depth-anything-v2` (Large) | per-tile inverse-depth map — used as the *first* car-mask source |
| metric depth (T4) | Replicate · `david20321/depth-anything-v3-metric-large` | tested as an alternative when DA-v2 collapsed the dynamic range; same blindness to overhead cars (see § 6) |
| auto-segmentation | Replicate · `pablodawson/segment-anything-automatic` (SAM v1 auto-mask generator) | the actual winning car-mask source |
| classical CV mask shaping | OpenCV morphology (white-tophat, blackhat, dilate, convex hull), scikit-image (skeletonize, frangi/meijering/sato/hessian ridge filters, structure tensor, SLIC) | per-car bridging + polygon hull + margin; line-detection sweep; positive parking-row detection |
| line detection | OpenCV (Canny, Hough, HoughP, ximgproc FastLineDetector, ximgproc EdgeDrawing), scikit-image | 19-technique sweep + 10 fusion combinations of tophat × Hough |
| in-painting | OpenCV Telea (`cv2.INPAINT_TELEA`) | fill the masked car regions with synthetic floor texture |
| secrets | Bitwarden Secrets Manager (`bws`), key `REPLICATE_API_DEFAULT_TOKEN` in the GoClever SM project | the only API token consumed; injected into the child Python process via `bws run --project-id <id> -- .venv/bin/inv <task>` (no `sh -c`/`bash -c` wrappers — those leak the env via `declare -x`; lesson learnt the hard way) |
| reproducibility | pyinvoke (`inv <task>`); single `extract_floor.py` orchestrator; per-stage scripts (`tile.py`, `upscale.py`, `depth.py`, `sam_automask.py`, `sam_to_mask.py`, `compose.py`, `region_v2.py`, `line_detect.py`, `line_detect_combos.py`) | every stage is one `inv` task with idempotent skip-on-exists; safe to interrupt and resume |

The wiki page `~/workspace/PedwebOrg/wiki/research/ml-hosting/replicate.md` is the canonical entry for the Replicate setup.

## 3. The pipeline

Numbered stages, each `inv <task>`. Stages 3a–3c are interchangeable mask-source variants; stage 5 (line detection) consumes the chosen floor output.

```
                       (chenxwh/depth-anything-v2)        ┌─────────────────────────────────────┐
                                                          │  3a depth → bridged-blob → blacked  │
                                                          │     `inv floor-all-bridged-blacked` │
                              ┌── per-tile depth PNG ─────┤  3a' same with percentile threshold │
                              │                           │     `inv floor-all-bridged-blacked-pthr`
                              │                           └─────────────────────────────────────┘
1. tile         2. upscale    │
   `inv tile`      `inv upscale-all` ── 42 upscaled tiles
   500×500         (Replicate)         ┌── per-tile coloured                ┌─── 3b SAM-only blacked
   42 tiles                            │   mask PNG                         │     `inv floor-all-sam`
                                       │  (`inv sam-all`)                   │
                                       └─ pablodawson/                      └─── 3c SAM ∪ depth blacked
                                          segment-anything-automatic              `inv floor-all-combined`

4. compose                                                                  5. region / line analysis
   `inv compose-floor[-sam|-combined]`                                          (region_v2.py, line_detect*.py)
   → full-res JPEG + 1024² preview                                             produces stripe-detection +
                                                                                positive parking-row map
```

## 4. Inputs and per-stage outputs (all absolute paths)

### 4.1 Tiling
- Source: `~/workspace/goclever/poc-homography/data/maps/icozee-cropped.tif` (3327×2731, 0.15 m/px in EPSG:31370).
- Output: 42 lossless PNG tiles at `…/poc/ortho_lines_534/map-tiles-500x500/` plus `tiles_manifest.json` (the recomposition contract: per-tile row/col + pixel-space `x0,y0,w,h`).

### 4.2 Upscaling (Replicate)
- One call per tile via `prunaai/p-image-upscale`, target 8 MPx, factor 2, enhance_realism, PNG output, q=100.
- Output: `…/poc/ortho_lines_534/map-tiles-500x500-upscaled/` — 42 PNGs, ~2896×2896 each (edge tiles smaller proportionally), ~12 MB each.
- Runtime: ~552 s total for the 40 remaining tiles after the rate-limit retry pattern was added; the first run hit the 6 RPM / burst 1 throttle for accounts with < $5 credit. Cost: ~$0.30 (42 × $0.007).
- Composed full-site preview: `…/poc/ortho_lines_534/out/composed/icozee_overview_1024_q90.jpg` (511 KB, 1024²) and `icozee_overview_1024.png` (lossless, 2.3 MB).

### 4.3 Depth (Replicate)
- One call per tile via `chenxwh/depth-anything-v2` Large.
- Output: 42 single-channel uint8 PNGs at `…/poc/ortho_lines_534/out/depth/` (`<stem>_depth.png`).
- Runtime: 642 s under the same throttle.

### 4.4 Per-tile mask — depth-based variants

Source code: `extract_floor.py::build_floor_mask` (polygon variant) and `build_bridged_car_mask` (bridged variant).

Algorithm (both variants):
1. **Local-peak residual** = `MORPH_TOPHAT(depth, ellipse_kernel=251)`. Cancels the global depth gradient so the residual is "how much higher than local neighbours". 251 px ≈ 1.5× a car silhouette at the upscaled GSD.
2. **Per-car blobs** = `residual ≥ thr`, where `thr` is either `20` (absolute, default) or `max(20, percentile(residual, 90))` (adaptive — needed for tiles where the boom consumes the dynamic range and squashes the car signal).
3. **Cleanup** = morph open/close at 15 px to drop speckle and close roof-glare holes.
4. **Min-area filter** = drop CCs `< 400 px²`.
5. **Bridge** = dilate at `group_bridge_px=80` so cars in the same row touch; CCs of the dilated mask = car groups.
6. **(Polygon variant only)** Convex hull of each group, dilated by `hull_margin_px=40` for slack. The bridged variant skips this and keeps the dilated blob outline.
7. **Apply** = either blacked (set non-floor to 0) or Telea-inpainted (synthetic asphalt texture).

Per-tile outputs:

| variant | folder | size |
|---|---|---|
| polygon + Telea inpaint | `…/map-tiles-500x500-upscaled-floor/` | 42 PNGs, ~10–15 MB each |
| bridged + blacked, absolute threshold | `…/map-tiles-500x500-upscaled-floor-bridged-blacked/` | 42 PNGs |
| bridged + blacked, percentile threshold | `…/map-tiles-500x500-upscaled-floor-bridged-blacked-pthr/` | 42 PNGs (bug fix: `max(abs, percentile)` so flat tiles aren't all-masked) |

### 4.5 Per-tile mask — SAM variant (the winner)

Source code: `sam_automask.py::call_sam` (one Replicate call per tile) + `sam_to_mask.py::build_car_mask` (size filter).

Algorithm:
1. **SAM** = `pablodawson/segment-anything-automatic` returns a coloured PNG where each non-black colour is one segmented region (typically 300–400 regions per tile).
2. **Size filter** = keep regions with `500 ≤ area ≤ 10000 px²` (car-sized) AND a separate "big" pool for `area > 10000 px²` (boom, building) — both are non-floor.
3. **Aspect filter** initially used (`≥ 1.3`) but dropped to `1.0` after it rejected half the cars (overhead cars look near-square when packed).
4. **Upscale + dilate** = NEAREST-resize the binary mask back to the upscaled tile resolution; 11 px ellipse dilation swallows the resize-rounding ring.
5. **Apply** = blacked (no inpaint for this variant; combined variant uses depth's polygon shapes).

Per-tile outputs:
- Raw SAM coloured PNGs: `…/poc/ortho_lines_534/out/sam/` (42 files, ~30–150 KB each).
- SAM-only blacked: `…/poc/ortho_lines_534/map-tiles-500x500-upscaled-floor-sam/`.
- SAM ∪ depth-bridged blacked: `…/poc/ortho_lines_534/map-tiles-500x500-upscaled-floor-combined/`.

Cost: 42 × $0.004 ≈ $0.17. Runtime: 733 s under throttle (~17 s/tile).

### 4.6 Composed full-site outputs

`compose.py::compose_overview` stitches per-tile floor PNGs at the canonical 5.792× scale into a single 19270×15818 BGR canvas, then INTER_AREA-downscales to 1024² in one step. JPEG only (no WebP, no lossless PNG for the full-res output — too large).

| view | path | bytes |
|---|---|---:|
| original ortho preview | `…/out/composed/icozee_overview_1024_q90.jpg` | 523 KB |
| floor (polygon + inpaint) full-res | `…/out/composed/icozee_floor_full.jpg` | 40 MB |
| floor (polygon + inpaint) 1024² | `…/out/composed/icozee_floor_1024.jpg` | 394 KB |
| floor (SAM-only) full-res | `…/out/composed/icozee_floor_sam_full.jpg` | 34 MB |
| floor (SAM-only) 1024² | `…/out/composed/icozee_floor_sam_1024.jpg` | 433 KB |
| floor (SAM ∪ depth) full-res | `…/out/composed/icozee_floor_combined_full.jpg` | 26 MB |
| floor (SAM ∪ depth) 1024² | `…/out/composed/icozee_floor_combined_1024.jpg` | 336 KB |

## 5. Line / region detection on the floor

### 5.1 Sweep — 19 classical CV techniques

Script: `line_detect.py`. Documented in the worktree at `docs/line-detection-techniques.md`. Runs each of:

- Edge magnitude: Canny, Sobel, Scharr, Laplacian-of-Gaussian, Difference-of-Gaussians, bilateral+Canny.
- Morphology: white-tophat, blackhat, directional-tophat (8 orientations), Otsu+skeletonize.
- Ridge filters: Frangi, Meijering, Sato, Hessian, Gabor-max.
- Segment detectors: Hough standard, Hough probabilistic, Fast Line Detector, EDLines.

Outputs (per variant) at `…/poc/ortho_lines_534/out/lines/<variant>/<NN>_<technique>.png`, with green-on-input overlays for the four segment detectors. Each result is AND-masked with the eroded floor ROI to remove polygon-boundary artefacts.

### 5.2 Fusion — tophat (07) × Hough (16) combinations

Script: `line_detect_combos.py`. Ten fusion variants A–J:

| id | combo | purpose |
|---|---|---|
| A | tophat-binary → Hough | Hough's edge input IS the tophat |
| B | tophat-grey → Canny → Hough | tophat as preprocess |
| C | tophat-skeleton → Hough | 1-px input |
| D | Canny → Hough → filter by tophat overlap | post-filter Hough lines that follow stripes |
| **E** | tophat ∩ Hough | **chosen as best by user** — only emits pixels where both fire |
| F | tophat-skel ∪ Hough | union, thinned |
| G | Hough on dilated tophat | thicken stripes so Hough votes |
| **H** | validated full-extent lines drawn everywhere | extrapolation visualisation |
| **I** | E (floor) ∪ validated Hough lines (car-mask region only) | best-of-both: real evidence + predicted under cars |
| J | two-colour diagnostic: E white, extrapolation red | visual inspector |

Outputs: `…/poc/ortho_lines_534/out/lines/polygon_inpainted_combos/<id>_<name>.png` (+ overlays).

The user picked **combo I** as the canonical line-detection output.

### 5.3 Region reconstruction (v2 — positive parking detection)

Script: `region_v2.py`. After v1 (Roboflow wall→room recipe) collapsed because painted stripes sit INSIDE parking rows rather than enclosing them, v2 detects parking rows POSITIVELY:

```
parking_score(x, y) = local_stripe_density · local_orientation_coherence
```

- `local_stripe_density` = Gaussian-smoothed tophat binary, σ ≈ 2 m, normalised to its 95th percentile.
- `local_orientation_coherence` = `(λ₁−λ₂)/(λ₁+λ₂)` of the structure-tensor eigenvalues (1 = perfectly oriented, 0 = isotropic).

Threshold (0.25) → close (0.5 m) → open (1.0 m) → CC → drop < 50 m² = parking rows. The complement of (parking ∪ NOT-floor) is then per-pixel colour-classified into `road` / `vegetation` / `water` / `other` (calibrated to this site's strong blue bias on asphalt; water gate ≥ 2.0× brighter than (R+G)/2 because asphalt itself sits at ≈ 1.18×).

Outputs (5000×5000 central crop tested): `…/poc/ortho_lines_534/out/regions_v2/03_overlay.jpg` (final view) + `03_overlay_1024.jpg`.

Final tuned counts on the central crop: 11 parking_row CCs (37.6% coverage), 2 road CCs, 0 vegetation, 0 water. Honest write-up of limits: `…/poc/ortho_lines_534/docs/region-reconstruction-v2.md`.

## 6. Findings — what worked, what didn't, why

### 6.1 Depth-anything-v2 alone is the wrong mask source for crowded car rows

The mode-killer is **per-tile dynamic-range compression**. When a tile contains a tall feature (e.g. the on-site boom/crane spanning multiple cells), the boom consumes ~95% of the depth map's [0, 255] range and squashes the car signal into the bottom 2–5 grey levels. Concretely on tile `r001_c004`:

| | r1c4 (boom tile, fails) | r2c3 (central, works) |
|---|---:|---:|
| depth p95 | 24 / 255 | 202 / 255 |
| tophat residual p95 | 9 | 76 |
| tophat residual p99 | 223 (= boom) | 110 |

Switching to **depth-anything-v3-metric-large** spreads the dynamic range better in aggregate but does NOT recover the per-car signal (cars are simply invisible in monocular depth from a near-orthographic overhead view — the deeper cause). H1 (dynamic-range compression) is real but downstream of H2 (model blindness).

### 6.2 SAM auto-mask is the right mask source

`pablodawson/segment-anything-automatic` segmented every car as its own instance on the boom-failing tile — 366 regions total; after size filter 298 cars + 3 big structures (boom, building). Visually clean and per-tile reproducible. Cost is negligible ($0.17 for the whole site).

Known open issue: SAM **over-masks** flat-asphalt tiles (r1c2 at 99.3% non-floor) because it segments every minor asphalt texture into small "regions" that pass the size filter. Two cheap mitigations queued:
1. Tighten size range to 2000–6000 px² (cars are uniform).
2. Reject masks whose mean BGR matches local asphalt within Δ20 (cars are darker / coloured).

### 6.3 Recursive masking (T3) — re-queued

Initial reasoning was wrong: I thought "if depth doesn't see cars, recursing won't add information". The user clarified that "cars" = any visibly-above-ground occluder, which depth-anything DOES see; recursing after masking the boom should re-normalise the dynamic range and surface the squashed car signal. T3 was not run before SAM was chosen as the canonical path, but the design is documented and can be turned on if SAM ever degrades.

### 6.4 Line detection: combo E (tophat ∩ Hough) is canonical; combo I extends through cars

Of the 19 sweep techniques + 10 fusions, the user identified **E** (tophat ∩ Hough — accepts a line pixel only when both the morphological "thin bright feature" detector AND the geometric Hough line agree). Combo **I** extends E by drawing validated full-extent Hough lines through the car-mask region so we get predicted line continuations under occluders.

### 6.5 Region reconstruction: positive interior detection beats boundary closure

The classical-CV ceiling for region partition is hit here. v1 (wall→room closure) failed because the boundary primitive doesn't enclose regions. v2 (positive parking-row detection by stripe density × orientation coherence) gave 11 sensible parking CCs on the central crop, but cannot distinguish road from aisle (photometrically identical asphalt). The structurally right next step — flagged but not yet implemented — is a small instance-seg model trained on curb/lane-band primitives (the Roboflow Mendes recipe transferred properly).

## 7. Process lessons (recorded for next iteration)

1. **NEVER use `bash -c '… $SECRET …'` to forward Replicate tokens.** Bash will dump `declare -x` on error and leak the entire env. The safe shape is `bws run --project-id <id> -- .venv/bin/inv <task>` and let the Python entrypoint promote `REPLICATE_API_DEFAULT_TOKEN → REPLICATE_API_TOKEN` in-process (see `upscale.py:11–14`).
2. **Replicate's `replicate.run()` defaults to a 60-s httpx read timeout.** Large model inputs (DA-v3-metric on a 2896 image) regularly exceed that. Pass an explicit `httpx.Timeout(600, connect=30)` via `replicate.Client(...)`.
3. **Throttle awareness.** Under-$5 accounts get 6 RPM / burst-1. Use the `_RATE_LIMIT_RESETS_RE`-driven retry with backoff (pattern in `upscale.py::_call_with_retry`).
4. **Image-tool sandbox quirk.** The harness Bash tool sometimes silently drops stdout when a known secret literal appears in expanded form (`${VAR:0:3}`) — looks like the command produced no output but actually ran. Redirect to a file or run via the Python entrypoint to confirm.
5. **Per-tile percentile thresholds need an absolute floor.** A tile with flat depth (90 % of pixels at residual = 0) makes `percentile(_, 90) = 0` and masks the entire image. Fix: `thr = max(absolute_floor, percentile)`.
6. **For images: WebP forbidden per user decision.** Lossless PNG for archival, JPEG q90 for downstream consumption. WebP is great but isn't on the table for this PoC.

## 8. Reproduce from scratch

```sh
cd /Users/vasco/workspace/goclever/mira/.worktrees/poc-ortho-lines-534/poc/ortho_lines_534

# one-time
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. cut
.venv/bin/inv tile --max-width 500 --max-height 500 --out-dir map-tiles-500x500

# 2. upscale (Replicate)
bws run --project-id 19f59ce7-1fec-4885-b86b-b3b701408989 -- \
  .venv/bin/inv upscale-all --workers 1

# 3. depth (Replicate)
bws run --project-id 19f59ce7-1fec-4885-b86b-b3b701408989 -- \
  .venv/bin/inv depth-all --workers 1

# 3b. SAM (Replicate, the winner)
bws run --project-id 19f59ce7-1fec-4885-b86b-b3b701408989 -- \
  .venv/bin/inv sam-all --workers 1

# 4. masks (pure CV)
.venv/bin/inv floor-all                            # polygon + inpaint
.venv/bin/inv floor-all-bridged-blacked            # bridged blob, abs thr
.venv/bin/inv floor-all-bridged-blacked-pthr      # bridged blob, percentile thr (with abs-floor fix)
.venv/bin/inv floor-all-sam                        # SAM-only blacked
.venv/bin/inv floor-all-combined                   # SAM ∪ depth blacked

# 5. stitch
.venv/bin/inv compose-floor
.venv/bin/inv compose-floor-sam
.venv/bin/inv compose-floor-combined

# 6. region/line analysis on the composed floor
.venv/bin/python region_v2.py
.venv/bin/python line_detect.py
.venv/bin/python line_detect_combos.py
```

Total Replicate cost: 42 upscale + 42 depth + 42 SAM ≈ $0.40 for the whole site, one-time.

## 9. Open follow-ups (ordered by impact)

1. **Tighten SAM size filter** to 2000–6000 px² and add an asphalt-colour reject. Should fix the over-masking on flat-asphalt tiles without losing real cars.
2. **Roll T3 (recursive masking)** out across all tiles as an additional mask source; union with SAM. Cheap (DA-v2 calls only) and addresses the case where SAM misses an occluder shape that depth catches via local relief.
3. **Train a small instance-seg model for curb / lane-band** primitives on 50–100 hand-labelled tiles. This is the structurally right answer for road-vs-parking-vs-aisle and unblocks the wall→room recipe.
4. **Per-tile uniform colour normalisation before stitching** — there are visible tile-boundary seams in the composed JPEGs that bias the downstream colour classifier; a histogram-matching pass across adjacent tiles would help.
5. **Wire the floor + line outputs back into the homography refinement pipeline** as the actual map-side fiducial set.
