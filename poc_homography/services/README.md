# Services Architecture

Domain services contain business logic that doesn't naturally belong to an entity or value object. They are stateless and receive all dependencies as arguments.

## Directory Structure

```
services/
├── __init__.py                     # Top-level exports (strategies only)
├── README.md                       # This file
├── service_<domain>.py             # Service class file
└── <domain>/                       # Strategies folder
    ├── __init__.py                 # Exports strategies + service
    ├── strategy.py                 # Strategy protocol
    └── <name>_strategy.py          # Strategy implementations
```

## Naming Conventions

| Component | Pattern | Example |
|-----------|---------|---------|
| Service class | `Service<Domain>` | `ServiceOrientation` |
| Service file | `service_<domain>.py` | `service_orientation.py` |
| Strategy protocol | `<Domain>Strategy` | `OrientationStrategy` |
| Strategy file | `strategy.py` | `orientation/strategy.py` |
| Strategy implementation | `<Name>Strategy` | `AdditiveOrientationStrategy` |
| Strategy impl file | `<name>_strategy.py` | `additive_strategy.py` |

## Example: Orientation Service

```
services/
├── service_orientation.py          # ServiceOrientation class
└── orientation/
    ├── __init__.py                 # Exports all orientation-related classes
    ├── strategy.py                 # OrientationStrategy protocol
    ├── additive_strategy.py        # AdditiveOrientationStrategy
    └── rotation_matrix_strategy.py # RotationMatrixStrategy
```

## Import Patterns

```python
# Import service and strategies from domain package
from poc_homography.services.orientation import (
    ServiceOrientation,
    AdditiveOrientationStrategy,
    RotationMatrixStrategy,
)

# Or import strategies from top-level (service not exported here)
from poc_homography.services import (
    AdditiveOrientationStrategy,
    OrientationStrategy,
)

# Or import specific implementation directly
from poc_homography.services.orientation.additive_strategy import AdditiveOrientationStrategy
```

## Design Principles

1. **Stateless**: Services receive all dependencies as constructor or method arguments
2. **Strategy Pattern**: Complex algorithms use pluggable strategies
3. **One class per file**: Java-like organization for clarity
4. **Protocol-based**: Strategy interfaces use `typing.Protocol` for structural typing
5. **Domain folder exports service**: The `<domain>/__init__.py` exports the service class
6. **Top-level exports strategies only**: The `services/__init__.py` exports strategies, not services

## Adding a New Service

1. Create `service_<domain>.py` with `Service<Domain>` class
2. Create `<domain>/` folder for strategies
3. Create `<domain>/strategy.py` with `<Domain>Strategy` protocol
4. Create `<domain>/<name>_strategy.py` for each implementation
5. Update `<domain>/__init__.py` to export everything
6. Update `services/__init__.py` to export strategies
