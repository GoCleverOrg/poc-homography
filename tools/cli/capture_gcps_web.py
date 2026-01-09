#!/usr/bin/env python3
"""CLI for web-based GCP capture tool."""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tools.capture_gcps_web import main

if __name__ == "__main__":
    main()
