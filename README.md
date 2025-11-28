# POC Homography - Unified Homography Interface for PTZ Cameras

A flexible homography computation system for PTZ (Pan-Tilt-Zoom) cameras that enables runtime selection between multiple approaches for transforming image coordinates to real-world GPS coordinates.

## Overview

This project provides a unified interface for computing homography transformations between camera image coordinates and world coordinates (GPS/local map). The system supports multiple homography computation approaches with seamless switching at runtime, allowing you to choose the best method for your specific use case or camera configuration.

**Key Benefit**: Switch between different homography computation strategies without changing your code - just update the configuration file.

## Features

- **Multiple Homography Approaches**
  - **Intrinsic/Extrinsic**: Uses camera calibration parameters (focal length, sensor size) and PTZ pose (pan, tilt, zoom) to compute precise homography transformations
  - **Feature Matching**: Uses computer vision techniques (SIFT, ORB, LoFTR) to match image features with known ground control points
  - **Learned**: Machine learning-based homography estimation using neural networks (placeholder for future ML models)

- **Unified Interface**: All approaches implement the same `HomographyProvider` interface for consistent usage
- **Runtime Configuration**: Switch approaches via YAML configuration without code changes
- **Fallback Chain**: Automatic fallback to alternative approaches if primary method fails
- **GPS Coordinate Projection**: Transform pixel coordinates to latitude/longitude (WGS84)
- **PTZ Camera Integration**: Built-in support for Hikvision PTZ cameras with RTSP streaming
- **High Confidence Scoring**: Each projection includes confidence metrics for quality assessment

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV-compatible camera or RTSP stream (for live testing)

### Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `numpy>=1.24` - Numerical computing and matrix operations
- `opencv-python>=4.7.0` - Computer vision and image processing
- `requests>=2.28` - HTTP client for PTZ camera APIs
- `PyYAML>=6.0` - Configuration file parsing

## Quick Start

### Basic Usage with Default Configuration

```python
from homography_factory import HomographyFactory
from homography_config import get_default_config
import numpy as np

# Load default configuration (uses intrinsic/extrinsic approach)
config = get_default_config()

# Create provider for your camera resolution
provider = HomographyFactory.from_config(
    config,
    width=2560,
    height=1440
)

# Capture a frame (example - replace with your camera capture)
frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

# Compute homography with camera pose information
reference = {
    'pan_degrees': 45.0,
    'tilt_degrees': -30.0,
    'zoom_factor': 2.0,
    'camera_height_m': 5.0,
    'camera_latitude': 37.7749,
    'camera_longitude': -122.4194
}

result = provider.compute_homography(frame, reference)
print(f"Homography confidence: {result.confidence}")

# Project image point to GPS coordinates
image_point = (1280, 720)  # Center of image
world_point = provider.project_point(image_point)
print(f"Lat: {world_point.latitude}, Lon: {world_point.longitude}")
print(f"Point confidence: {world_point.confidence}")
```

### Custom Configuration from YAML

```python
from homography_factory import HomographyFactory
from homography_config import HomographyConfig

# Load configuration from YAML file
config = HomographyConfig.from_yaml('homography_config.yaml')

# Create provider based on config
provider = HomographyFactory.from_config(
    config,
    width=2560,
    height=1440,
    try_fallbacks=True  # Enable automatic fallback to alternative approaches
)

# Use provider as shown above
if provider.is_valid():
    world_point = provider.project_point((1280, 720))
```

## Architecture Overview

The system uses a factory pattern with abstract interfaces for maximum flexibility:

```
┌─────────────────────────────────────────────────────────┐
│                   HomographyProvider                    │
│                  (Abstract Interface)                   │
│  - compute_homography(frame, reference)                 │
│  - project_point(image_point) -> WorldPoint             │
│  - project_points(image_points) -> List[WorldPoint]     │
│  - get_confidence() -> float                            │
│  - is_valid() -> bool                                   │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────┴────────┐ ┌───────┴────────┐ ┌──────┴───────┐
│   Intrinsic    │ │    Feature     │ │   Learned    │
│   Extrinsic    │ │     Match      │ │  Homography  │
│  Homography    │ │   Homography   │ │              │
└────────────────┘ └────────────────┘ └──────────────┘

┌─────────────────────────────────────────────────────────┐
│              HomographyFactory                          │
│  - create(approach, width, height, **kwargs)            │
│  - from_config(config, width, height)                   │
│  - register(approach, provider_class)                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              HomographyConfig                           │
│  - approach: HomographyApproach                         │
│  - fallback_approaches: List[HomographyApproach]        │
│  - approach_specific_config: Dict                       │
│  - from_yaml(path) / to_yaml(path)                      │
└─────────────────────────────────────────────────────────┘
```

### Main Classes

- **`HomographyProvider`**: Abstract base class defining the interface all approaches must implement
- **`IntrinsicExtrinsicHomography`**: Camera parameter-based approach using calibration and PTZ pose
- **`FeatureMatchHomography`**: Feature detection and matching approach (SIFT/ORB/LoFTR)
- **`LearnedHomography`**: Neural network-based approach (placeholder for ML models)
- **`HomographyFactory`**: Factory for creating providers with approach registration
- **`HomographyConfig`**: Configuration management with YAML serialization

## Usage Examples

### Creating a Provider Using the Factory

```python
from homography_interface import HomographyApproach
from homography_factory import HomographyFactory

# Direct creation with intrinsic/extrinsic approach
provider = HomographyFactory.create(
    HomographyApproach.INTRINSIC_EXTRINSIC,
    width=2560,
    height=1440,
    pixels_per_meter=100.0
)

# Create with feature matching approach
provider = HomographyFactory.create(
    HomographyApproach.FEATURE_MATCH,
    width=2560,
    height=1440,
    detector='sift',
    min_matches=10,
    ransac_threshold=3.0
)
```

### Computing Homography

```python
import numpy as np

# Example frame (replace with actual camera frame)
frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

# For intrinsic/extrinsic approach - provide camera pose
reference = {
    'pan_degrees': 30.0,
    'tilt_degrees': -25.0,
    'zoom_factor': 1.5,
    'camera_height_m': 6.0,
    'camera_latitude': 37.7749,
    'camera_longitude': -122.4194
}

# Compute homography
result = provider.compute_homography(frame, reference)

print(f"Homography matrix shape: {result.homography_matrix.shape}")
print(f"Confidence score: {result.confidence}")
print(f"Approach used: {result.metadata.get('approach')}")
```

### Projecting Points

```python
# Single point projection
image_point = (1280, 720)  # Pixel coordinates (u, v)
world_point = provider.project_point(image_point)

print(f"Image point {image_point} maps to:")
print(f"  Latitude: {world_point.latitude}")
print(f"  Longitude: {world_point.longitude}")
print(f"  Confidence: {world_point.confidence}")

# Batch point projection (more efficient)
image_points = [
    (640, 360),
    (1280, 720),
    (1920, 1080)
]
world_points = provider.project_points(image_points)

for img_pt, world_pt in zip(image_points, world_points):
    print(f"{img_pt} -> ({world_pt.latitude}, {world_pt.longitude})")
```

### Using with RTSP Streams

```python
import cv2
from homography_factory import HomographyFactory
from homography_config import HomographyConfig

# Load configuration
config = HomographyConfig.from_yaml('homography_config.yaml')

# Open RTSP stream
stream_url = "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"
cap = cv2.VideoCapture(stream_url)

# Get stream resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create provider
provider = HomographyFactory.from_config(config, width, height)

# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update homography with current PTZ pose
    # (retrieve from camera API)
    reference = {
        'pan_degrees': get_current_pan(),
        'tilt_degrees': get_current_tilt(),
        'zoom_factor': get_current_zoom(),
        'camera_height_m': 5.0,
        'camera_latitude': 37.7749,
        'camera_longitude': -122.4194
    }

    result = provider.compute_homography(frame, reference)

    # Project points of interest
    if provider.is_valid():
        point = provider.project_point((width//2, height//2))
        print(f"Center maps to: {point.latitude}, {point.longitude}")

cap.release()
```

## Configuration

### YAML Configuration File

The `homography_config.yaml` file controls which approach is used:

```yaml
homography:
  # Primary approach to use
  approach: intrinsic_extrinsic

  # Fallback approaches if primary fails
  fallback_approaches:
    - feature_match

  # Intrinsic/Extrinsic approach configuration
  intrinsic_extrinsic:
    sensor_width_mm: 7.18        # Physical sensor width
    base_focal_length_mm: 5.9    # Focal length at 1x zoom
    pixels_per_meter: 100.0      # Map visualization scale

  # Feature matching approach configuration
  feature_match:
    detector: sift               # Feature detector: sift | orb | loftr
    min_matches: 4               # Minimum matches for valid homography
    ransac_threshold: 5.0        # RANSAC inlier threshold (pixels)

  # Learned approach configuration
  learned:
    model_path: null             # Path to trained model
    confidence_threshold: 0.5    # Minimum confidence score
```

### Available Approach Options

1. **`intrinsic_extrinsic`** - Best for PTZ cameras with known parameters
   - Requires: Camera calibration parameters, PTZ pose (pan/tilt/zoom)
   - Advantages: Fast, accurate, no training data needed
   - Use when: Camera parameters are known and PTZ position is available

2. **`feature_match`** - Best for unknown camera parameters
   - Requires: Reference image or ground control points
   - Advantages: Works without calibration, robust to camera changes
   - Use when: Camera parameters unknown or reference imagery available

3. **`learned`** - Best for challenging conditions
   - Requires: Trained neural network model
   - Advantages: Can handle complex scenarios, learns from data
   - Use when: Traditional methods fail or training data is available

### Approach-Specific Parameters

**Intrinsic/Extrinsic:**
- `sensor_width_mm`: Physical sensor width (default: 7.18mm for 1/2.8" sensor)
- `base_focal_length_mm`: Focal length at 1x zoom (default: 5.9mm)
- `pixels_per_meter`: Visualization scale (default: 100.0)

**Feature Match:**
- `detector`: Feature type - `'sift'`, `'orb'`, or `'loftr'` (default: 'sift')
- `min_matches`: Minimum feature matches required (default: 4)
- `ransac_threshold`: Inlier threshold in pixels (default: 5.0)

**Learned:**
- `model_path`: Path to model weights file (default: None)
- `confidence_threshold`: Minimum confidence score (default: 0.5)

## API Reference

### HomographyProvider Interface

All homography providers implement these core methods:

#### `compute_homography(frame, reference) -> HomographyResult`
Computes the homography transformation from the input frame.

**Parameters:**
- `frame`: Image as numpy array (height, width, channels)
- `reference`: Dictionary with approach-specific data
  - For intrinsic/extrinsic: `pan_degrees`, `tilt_degrees`, `zoom_factor`, `camera_height_m`, `camera_latitude`, `camera_longitude`
  - For feature match: `ground_points`, `image_points`

**Returns:** `HomographyResult` with `homography_matrix`, `confidence`, and `metadata`

#### `project_point(image_point) -> WorldPoint`
Projects a single image coordinate to GPS coordinates.

**Parameters:**
- `image_point`: Tuple `(u, v)` in pixel coordinates

**Returns:** `WorldPoint` with `latitude`, `longitude`, and `confidence`

#### `project_points(image_points) -> List[WorldPoint]`
Projects multiple image coordinates (batch operation).

**Parameters:**
- `image_points`: List of `(u, v)` tuples

**Returns:** List of `WorldPoint` objects

#### `get_confidence() -> float`
Returns the confidence score of the current homography (0.0 to 1.0).

#### `is_valid() -> bool`
Checks if the homography is valid and ready for projection.

### Data Classes

#### `WorldPoint`
- `latitude`: Latitude in decimal degrees [-90, 90]
- `longitude`: Longitude in decimal degrees [-180, 180]
- `confidence`: Confidence score [0.0, 1.0]

#### `HomographyResult`
- `homography_matrix`: 3x3 numpy array
- `confidence`: Overall confidence score [0.0, 1.0]
- `metadata`: Dictionary with approach-specific information

## Testing

### Run All Tests

```bash
# Test configuration system
python test_config_system.py

# Test intrinsic/extrinsic implementation
python test_intrinsic_extrinsic.py

# Test configuration loading only
python test_config_only.py
```

### Test File Locations

- `/tests/test_homography_consistency.py` - Homography computation consistency tests
- `/tests/test_hikvision_ptz.py` - Hikvision PTZ integration tests
- `/test_config_system.py` - Configuration and factory system tests
- `/test_intrinsic_extrinsic.py` - Intrinsic/extrinsic approach tests
- `/test_config_only.py` - Configuration loading and validation tests

### Example Test Usage

```bash
# Test with example configuration
python test_config_system.py

# Expected output:
# ✓ Default configuration test passed
# ✓ YAML round-trip test passed
# ✓ Example YAML loading test passed
# ✓ Factory creation test passed
```

## Project Structure

```
poc-homography/
├── homography_interface.py          # Abstract interface definitions
├── homography_factory.py            # Factory for creating providers
├── homography_config.py             # Configuration management
├── intrinsic_extrinsic_homography.py # Camera parameter approach
├── feature_match_homography.py      # Feature matching approach
├── learned_homography.py            # ML-based approach
├── homography_config.yaml           # Example configuration
├── requirements.txt                 # Python dependencies
├── tests/                           # Test suite
│   ├── test_homography_consistency.py
│   └── test_hikvision_ptz.py
└── docs/                           # Additional documentation
    ├── CONFIGURATION_GUIDE.md
    ├── MIGRATION_GUIDE.md
    └── IMPLEMENTATION_SUMMARY.md
```

## Contributing

This project tracks issues and development using GitHub Issues. When contributing:

1. Check existing issues before starting new work
2. Follow the existing code style (PEP 8)
3. Add tests for new features
4. Update documentation as needed
5. Reference issue numbers in commit messages (e.g., "Fix #14: Add unified interface")

### Code Style Notes

- Use type hints for all public methods
- Document classes and methods with docstrings (Google style)
- Keep classes focused and single-purpose
- Prefer composition over inheritance
- Use descriptive variable names
- Add comments for complex algorithms

## Additional Documentation

- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - Detailed configuration reference
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migrating from old code to unified interface
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference for common tasks

## License

[To be determined - Add license information here]

## Acknowledgments

This unified homography interface was implemented as part of Issue #14 to provide a flexible, extensible system for homography computation across different camera types and scenarios.
