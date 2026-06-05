#!/usr/bin/env python3
"""Standalone ISAPI endpoint discovery CLI.

Replaces the retired ``hikvision_ptz_discovery`` script: it builds a
:class:`HikvisionISAPIClient` from command-line arguments and prints the HTTP
status of each well-known ISAPI endpoint, reusing the adapter's
:meth:`HikvisionISAPIClient.discover_endpoints`.

Usage::

    python -m ptz_discovery_and_control.hikvision.discover IP USERNAME PASSWORD
"""

from __future__ import annotations

import sys

from poc_homography.infrastructure.clients.hikvision.isapi_client import (
    HikvisionISAPIClient,
)


def main(argv: list[str] | None = None) -> int:
    """Probe a camera's ISAPI endpoints and print one ``path -> status`` per line.

    Args:
        argv: Optional argument vector (``[ip, username, password]``); defaults
            to ``sys.argv[1:]``.

    Returns:
        Process exit code (``0`` on success, ``2`` on bad arguments).
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print("Usage: python -m ptz_discovery_and_control.hikvision.discover IP USER PASS")
        return 2

    ip, username, password = args
    client = HikvisionISAPIClient(ip, username, password)
    results = client.discover_endpoints()
    for path, status_code in results.items():
        print(f"{status_code:>4}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
