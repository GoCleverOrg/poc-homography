# Map asset storage

Map GeoTIFFs (`data/maps/*.tif`) are stored in an S3/MinIO bucket and uploaded
by a repeatable pipeline. The upload client mirrors the survey frame store
(`MinioFrameStore`); it shares the same MinIO endpoint and credentials and only
differs in the target bucket.

## Bucket

| Property | Value | Source |
| --- | --- | --- |
| Bucket name | `map-assets` (default) | `MINIO_MAP_BUCKET` |
| Region | `us-east-1` (default; MinIO ignores it) | `MINIO_REGION` |
| Endpoint | — | `MINIO_ENDPOINT` |

The bucket is created on first use (`ensure_bucket` is idempotent).

## Environment variables

All MinIO config comes from the environment (see `.env.example`):

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `MINIO_ENDPOINT` | yes | — | MinIO S3 endpoint, e.g. `http://s3.10-121-15-59.sslip.io:9000` |
| `MINIO_ACCESS_KEY` | yes | — | MinIO access key |
| `MINIO_SECRET_KEY` | yes | — | MinIO secret key |
| `MINIO_REGION` | no | `us-east-1` | S3 region label |
| `MINIO_MAP_BUCKET` | no | `map-assets` | Bucket for map GeoTIFFs |

## Object key structure

Each map is described by a YAML sidecar in `data/maps/` carrying a `tenant_id`
and a `photo.path` (the GeoTIFF filename). The object key is tenant-scoped:

```
{tenant_id}/{photo.path}
```

For the committed maps:

| Sidecar | tenant_id | photo.path | Object key |
| --- | --- | --- | --- |
| `icozee.yaml` | `icozee` | `icozee-cropped.tif` | `icozee/icozee-cropped.tif` |
| `valte.yaml` | `valte` | `Cartografia_valencia.tif` | `valte/Cartografia_valencia.tif` |

## Upload pipeline

The `.tif` files are DVC-tracked and not normally on disk. Materialize them
first, then upload:

```bash
dvc pull           # or: poe dvc-pull — fetch the .tif files
hom maps upload     # or: poe upload-maps
```

`hom maps upload` globs `data/maps/*.yaml`, derives the key for each, and
uploads the matching `.tif`. A tif that is absent on disk is reported as
`SKIP (missing on disk — run 'dvc pull')` and the run continues; it is not an
error. A summary line reports the uploaded/missing counts.

Override the directory with `--maps-dir <path>`.

### Idempotency

Re-running `hom maps upload` is safe: each upload overwrites the same key, and
`ensure_bucket` is a no-op when the bucket already exists. The object metadata
records a `sha256` of the uploaded bytes.

## Adding a new map

1. Add `data/maps/<name>.yaml` with at least `tenant_id` and `photo.path`:

   ```yaml
   id: <map_id>
   tenant_id: <tenant>
   photo:
     path: <name>.tif
   ```

2. Place `data/maps/<name>.tif` and DVC-track it:

   ```bash
   dvc add data/maps/<name>.tif
   dvc push          # or: poe dvc-push
   ```

3. Upload it:

   ```bash
   hom maps upload    # or: poe upload-maps
   ```

   The new asset lands at `{tenant_id}/<name>.tif` in the `map-assets` bucket.
