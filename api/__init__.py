"""FastAPI application package.

Adds ``webapp/`` to *sys.path* so that Django-oriented modules (which use
bare ``from homography_web ...`` imports) can be loaded by the FastAPI app
without requiring the Django manage.py path setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WEBAPP_DIR = str(Path(__file__).resolve().parents[1] / "webapp")
if _WEBAPP_DIR not in sys.path:
    sys.path.insert(0, _WEBAPP_DIR)
