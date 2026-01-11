# Tools Directory - DEPRECATED

This directory has been deprecated. All CLI tools have been consolidated into the unified `hom` command using Typer.

## Installation

```bash
uv pip install -e .
```

## Migration Guide

| Old Tool | New Command |
|----------|-------------|
| `python tools/calibrate_projection.py` | `hom calibrate projection` |
| `python tools/comprehensive_calibration.py` | `hom calibrate comprehensive` |
| `python tools/get_camera_intrinsics.py` | `hom camera intrinsics` |
| `python tools/validate_camera_model.py` | `hom camera validate` |
| `python tools/verify_gcp_gps.py` | `hom gcp verify` |
| `python tools/test_data_generator.py` | `hom test data-generator` |
| `python tools/test_sam3_prompts.py` | `hom test sam3` |
| `python tools/interactive_calibration.py` | `hom interactive` |

## Getting Help

Each command supports `--help` for detailed usage information:

```bash
# Main help
hom --help

# Subcommand help
hom calibrate --help
hom calibrate projection --help
hom camera intrinsics --help
```

## Notes

- All tools now use the Map Points format (pixel coordinates in JSON). See `MAP_POINTS_FORMAT.md` for details on the coordinate format.
- The `extract_kml_points.py` tool has been removed as KML-based coordinates are no longer supported. Use the Map Points system instead.
- For web-based tools (capture GCPs, GCP verification map), see the Django webapp at `webapp/`.
