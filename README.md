# POC Homography

Homography computation system for PTZ cameras - transforms image coordinates to map/world coordinates.

## Project Structure

```
poc-homography/
├── poc_homography/              # Python library
│   ├── application/             # Composition Root (DI container)
│   ├── domain/                  # Domain layer (entities, VOs, protocols)
│   │   ├── entities/            # Domain entities
│   │   ├── vo/                  # Value objects
│   │   ├── enums/               # Enumerations
│   │   └── repositories/        # Repository protocols
│   ├── infrastructure/          # Infrastructure layer
│   │   └── repositories/        # YAML-backed repository implementations
│   ├── services/                # Domain services with strategy pattern
│   │   ├── orientation/         # Orientation strategies
│   │   └── homography/          # Homography strategies
│   ├── cli/                     # CLI commands (hom)
│   └── ...
├── webapp/                      # Django web application
├── tests/                       # Test suite
└── data/                        # YAML data files
    ├── cameras/                 # Camera configurations
    ├── calibrations/            # Camera calibrations
    ├── maps/                    # Map definitions
    └── ground_control_points/   # GCP data
```

## Architecture

This project follows Domain-Driven Design (DDD) with a layered architecture:

### Domain Layer (`domain/`)

Contains the core business logic, independent of infrastructure concerns.

**Note on GCP coordinates**: Ground Control Points store pixel coordinates on the map image. Since maps are GeoTIFFs, these can be converted to lat/lng via the embedded georeferencing. We use pixel coordinates as the primary representation because homography calculations operate in pixel space.

| Component | Location | Example |
|-----------|----------|---------|
| Entity | `entities/<name>.py` | `CameraConfig`, `Map` |
| Value Object | `vo/<name>.py` | `Orientation`, `PTZState` |
| Repository Protocol | `repositories/repo.py` | `Repo[T]` |

### Infrastructure Layer (`infrastructure/`)

Implements domain interfaces with concrete technologies.

| Component | Pattern | Example |
|-----------|---------|---------|
| Repository class | `Repo<Tech><Entity>` | `RepoYamlCameraConfig` |
| Repository file | `repo_<tech>_<entity>.py` | `repo_yaml_camera_config.py` |
| Base class | `Repo<Tech>` | `RepoYaml` |
| Base file | `repo_<tech>.py` | `repo_yaml.py` |
| Mixin class | `Mixin<Component><Purpose>` | `MixinRepoMapFilter` |
| Mixin file | `mixin_<component>_<purpose>.py` | `mixin_repo_map_filter.py` |

### Application Layer (`application/`)

Composition Root for dependency injection.

```python
from poc_homography.application import ApplicationContext

ctx = ApplicationContext.default()
cameras = ctx.repo_camera_config.get_all()
calibration = ctx.repo_camera_calibration.get(camera.id)
```

### Services Layer (`services/`)

Domain services with pluggable strategies.

| Component | Pattern | Example |
|-----------|---------|---------|
| Service class | `Service<Domain>` | `ServiceOrientation` |
| Service file | `service_<domain>.py` | `service_orientation.py` |
| Strategy protocol | `Strategy<Domain>` | `StrategyOrientation` |
| Strategy protocol file | `<domain>/strategy.py` | `orientation/strategy.py` |
| Strategy impl | `Strategy<Domain><Name>` | `StrategyOrientationAdditive` |
| Strategy impl file | `strategy_<name>.py` | `strategy_additive.py` |

## Getting Started

### Install

```bash
uv sync
```

### Library

```python
from poc_homography.application import ApplicationContext
from poc_homography.services import ServiceHomography

# Get camera config and calibration
ctx = ApplicationContext.default()
config = ctx.repo_camera_config.get("my-camera")
calibration = ctx.repo_camera_calibration.get("my-camera")

# Compute homography
service = ServiceHomography()
result = service.compute(config, calibration, ptz_state, map_entity)
```

### CLI

```bash
# Show available commands
hom --help

# Camera operations
hom camera intrinsics --camera Valte
hom camera validate --camera Valte

# Calibration
hom calibrate comprehensive --camera Valte

# Test data generator
hom test data-generator --list-cameras
```

### Web Application

```bash
cd webapp
uv run python manage.py runserver
```

Open http://localhost:8000/:
- `/capture/` - GCP capture tool
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
