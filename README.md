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

### Code Quality

```bash
uv run poe lint          # Ruff linter
uv run poe typecheck     # Pyright
uv run poe validate      # All checks (no tests)
uv run poe ci            # Full CI pipeline
```

## License

MIT
