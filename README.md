# POC Homography

Homography computation system for PTZ cameras - transforms image coordinates to map/world coordinates.

## Project Structure

```
poc-homography/
├── poc_homography/          # Python library
│   ├── homography/          # Homography providers (strategy pattern)
│   ├── map_points/          # Map point registry
│   ├── calibration/         # Calibration data structures
│   └── ...
├── tools/cli/               # CLI commands (hom)
├── webapp/                  # Django web application
└── tests/                   # Test suite
```

## Getting Started

### Install

```bash
uv sync
```

### Library

```python
from poc_homography import (
    CameraGeometry,
    CameraParameters,
    IntrinsicExtrinsicHomography,
    MapPointHomography,
)

# Create camera parameters
params = CameraParameters.create(
    pan_deg=45.0,
    tilt_deg=30.0,
    zoom_factor=2.0,
    camera_height_m=5.0,
    image_width=2560,
    image_height=1440,
)

# Compute homography
result = CameraGeometry.compute(params)

# Project points
world_x, world_y = CameraGeometry.project_image_to_world(result, u=1280, v=720)
```

### CLI

```bash
# Show available commands
hom --help

# Camera operations
hom camera intrinsics --zoom 5.0
hom camera validate

# Calibration
hom calibrate projection
hom calibrate comprehensive

# GCP verification
hom gcp verify

# Interactive calibration UI
hom interactive
```

### Web Application

```bash
cd webapp
uv run python manage.py runserver
```

Open http://localhost:8000/:
- `/capture/` - GCP capture tool (click map to add ground control points)
- `/debug/` - Debug map visualization

## Development

### Run Tests

```bash
uv run poe test          # Run all tests
uv run poe test-cov      # With coverage
```

#### Live camera tests

The Hikvision live integration suite
(`tests/infrastructure/test_hikvision_isapi_live.py`) talks to a real PTZ
camera and is **skipped by default**. It is gated solely on `ICO_CAMERA_IP`
(independent of `DATABASE_URL`), so it runs whenever a camera is reachable:

| Env var               | Required | Purpose                                                        |
| --------------------- | -------- | -------------------------------------------------------------- |
| `ICO_CAMERA_IP`       | yes      | Camera host/IP. Its presence enables the read tests.           |
| `ICO_CAMERA_USERNAME` | no       | Login (defaults to `admin`).                                   |
| `ICO_CAMERA_PASSWORD` | no       | Login password (defaults to empty).                            |
| `ICO_CAMERA_WRITE`    | no       | Set to any value to enable the move/restore test (moves PTZ).  |

```bash
# Read-only suite against cam-04
ICO_CAMERA_IP=10.107.50.5 ICO_CAMERA_USERNAME=admin ICO_CAMERA_PASSWORD=... \
  uv run pytest tests/infrastructure/test_hikvision_isapi_live.py

# Include the physical move/restore test
ICO_CAMERA_IP=10.107.50.5 ICO_CAMERA_WRITE=1 ICO_CAMERA_USERNAME=admin ICO_CAMERA_PASSWORD=... \
  uv run pytest tests/infrastructure/test_hikvision_isapi_live.py
```

No `DATABASE_URL` is needed; the Postgres integration tests remain gated
separately on `DATABASE_URL`.

### Code Quality

```bash
uv run poe lint          # Ruff linter
uv run poe typecheck     # Pyright
uv run poe validate      # All checks (no tests)
uv run poe ci            # Full CI pipeline
```

## License

MIT
