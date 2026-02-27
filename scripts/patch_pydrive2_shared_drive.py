#!/usr/bin/env python3
"""Patch pydrive2 to work with Google Shared Drive folders.

pydrive2 (<=1.21.3) uses ``corpora=drive`` with ``driveId`` when the
remote folder is inside a Google Shared Drive.  This requires the
authenticated user to be a **member** of the Shared Drive.  Users who
only have folder-level access (e.g. ``fileOrganizer``) get:

    HttpError 403 "teamDriveMembershipRequired"

Replacing ``corpora=drive`` + ``driveId`` with ``corpora=allDrives``
(without ``driveId``) makes DVC push/pull work for any user who has at
least editor-level access on the specific folder.

This script is idempotent and safe to re-run after ``uv sync``.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


def _find_spec_py() -> Path:
    spec = importlib.util.find_spec("pydrive2.fs.spec")
    if spec is None or spec.origin is None:
        print("ERROR: pydrive2 is not installed", file=sys.stderr)
        sys.exit(1)
    return Path(spec.origin)


_ORIGINAL = (
    '                params["driveId"] = drive_id\n'
    '                params["corpora"] = "drive"'
)
_PATCHED = '                params["corpora"] = "allDrives"'


def main() -> None:
    path = _find_spec_py()
    src = path.read_text()

    if _PATCHED in src and _ORIGINAL not in src:
        print(f"Already patched: {path}")
        return

    if _ORIGINAL not in src:
        # Try regex for slight formatting differences
        pattern = re.compile(
            r'params\["driveId"\]\s*=\s*drive_id\s*\n\s*params\["corpora"\]\s*=\s*"drive"'
        )
        if not pattern.search(src):
            print(f"WARNING: cannot locate target code in {path}", file=sys.stderr)
            print("pydrive2 may have been updated — check manually.", file=sys.stderr)
            sys.exit(1)
        src = pattern.sub('params["corpora"] = "allDrives"', src)
    else:
        src = src.replace(_ORIGINAL, _PATCHED)

    path.write_text(src)
    print(f"Patched: {path}")


if __name__ == "__main__":
    main()
