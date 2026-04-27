"""Export FastAPI OpenAPI schema to JSON file for client generation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import app

schema = app.openapi()
out = Path(__file__).resolve().parents[1] / "spa" / "openapi.json"
out.write_text(json.dumps(schema, indent=2))
print(f"Wrote OpenAPI schema to {out}")
