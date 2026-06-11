# Test & survey fixtures (without DVC)

DVC has been removed from this repo (issue #325). Map GeoTIFF assets are now
served from S3/MinIO (see [map_asset_storage.md](./map_asset_storage.md)); the
remaining DVC-tracked artifacts were **test and survey fixtures**, which are now
plain local files that are not version-controlled.

## CI needs no fixtures

`poe ci` is green with no fixtures present. Tests that depend on local fixtures
guard themselves with `pytest.mark.skipif(...)` and **skip** when the files are
absent — exactly as they did under DVC, since CI never ran `dvc pull`. CI runs
`uv sync --group dev` then `uv run poe ci`; no fixture-acquisition step exists or
is needed.

## Fixture locations

| Path | Used by |
| --- | --- |
| `tests/homography/test_data/` | `tests/homography/test_map_points*.py`, `test_homography_precision.py` |
| `survey/<date>/` | survey offline reprocessing (`hom cleanplate ...`, survey runners) |
| `data/captured_frames/` | clean-plate / capture tooling |

## Obtaining fixtures for local dev

These files previously lived in a shared Google Drive folder reached via DVC.
With DVC removed, obtain them out-of-band and drop them at the paths above:

- Ask a teammate for the current fixture bundle, or copy them from an existing
  checkout that still has them on disk.
- Place each file at the path the test/tool expects (see the table). The
  fixture-dependent tests detect presence and run; otherwise they skip.

In a **git worktree**, `scripts/worktree-setup.sh` (invoked by `./run.sh`) copies
`tests/homography/test_data/` from the main checkout automatically, so a worktree
inherits whatever fixtures the main checkout already has.

## Map GeoTIFFs

Map `.tif` files are **not** fixtures in this sense — they are served from
S3/MinIO at runtime via `Map.asset_key`. To (re)populate the bucket, place the
`.tif` next to its YAML sidecar in `data/maps/` and run `poe upload-maps`. See
[map_asset_storage.md](./map_asset_storage.md).
