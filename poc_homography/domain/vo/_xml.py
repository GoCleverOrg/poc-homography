"""Namespace-agnostic XML element helpers for value-object parsers.

Hikvision ISAPI documents declare a default XML namespace, so every tag is
namespace-qualified when parsed by ``ElementTree``. Domain value objects must
not import the infrastructure layer (where the namespace literal lives), so
these helpers match tags by their local name and ignore any namespace. This
keeps ``from_element`` parsers in the domain free of both infrastructure
imports and hardcoded namespace strings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element


def _local_name(tag: str) -> str:
    """Return the local part of a possibly namespace-qualified tag."""
    return tag.rsplit("}", 1)[-1]


def find_child(elem: Element, *path: str) -> Element | None:
    """Find a descendant element by a chain of local tag names.

    Args:
        elem: Root element to search from.
        path: One or more local tag names describing the descent path.

    Returns:
        The first matching descendant, or ``None`` if any step is missing.
    """
    current: Element | None = elem
    for name in path:
        if current is None:
            return None
        current = next(
            (child for child in current if _local_name(child.tag) == name),
            None,
        )
    return current


def find_text(elem: Element, *path: str) -> str | None:
    """Return the stripped text of a descendant element, or ``None``.

    Args:
        elem: Root element to search from.
        path: One or more local tag names describing the descent path.

    Returns:
        The element text stripped of surrounding whitespace, or ``None`` if the
        element is absent or has no text.
    """
    found = find_child(elem, *path)
    if found is None or found.text is None:
        return None
    text = found.text.strip()
    return text or None


def find_all(elem: Element, *path: str) -> list[Element]:
    """Return all direct children of a descendant matching a final local name.

    The path locates a container element; the last name selects its matching
    children. For example ``find_all(root, "CameraList", "Camera")`` returns
    every ``<Camera>`` under ``<CameraList>``. With a single name the container
    is ``elem`` itself, so matching direct children of ``elem`` are returned.

    Args:
        elem: Root element to search from.
        path: Container path followed by the repeated child's local name.

    Returns:
        A list of matching elements (possibly empty).
    """
    *container_path, last = path
    container = find_child(elem, *container_path) if container_path else elem
    if container is None:
        return []
    return [child for child in container if _local_name(child.tag) == last]
