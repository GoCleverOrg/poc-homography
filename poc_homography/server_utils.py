"""
Server utility functions for POC Homography web tools.

This module provides shared utilities for all web server tools in the project,
ensuring consistent behavior for port binding, error handling, and server setup.

Example Usage:
    from poc_homography.server_utils import find_available_port

    port = find_available_port(start_port=8080, max_attempts=10)
    print(f"Server starting on port {port}")
"""

import socket


def find_available_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """
    Find an available port starting from start_port.

    Attempts to bind to consecutive ports starting from start_port until
    an available port is found or max_attempts is reached.

    Args:
        start_port: Port number to start searching from (default: 8080)
        max_attempts: Maximum number of ports to try (default: 10)

    Returns:
        First available port number

    Raises:
        RuntimeError: If no available port found within max_attempts

    Example:
        >>> port = find_available_port(8080, 10)
        >>> print(f"Found available port: {port}")
        Found available port: 8080
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            # Port is in use, try next
            continue

    raise RuntimeError(
        f"Could not find an available port in range "
        f"{start_port}-{start_port + max_attempts - 1}"
    )
