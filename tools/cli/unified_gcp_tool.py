#!/usr/bin/env python3
"""CLI for unified GCP tool.

DEPRECATION WARNING:
This tool is deprecated and incompatible with the MapPoint migration (issue #156).
It relies heavily on GPS coordinates (latitude/longitude) and KML files, which have
been removed from the codebase in favor of map-relative pixel coordinates (MapPoint).

For map-based homography, use MapPointHomography with map pixel coordinates instead.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.unified_gcp_tool import main

if __name__ == "__main__":
    main()
