"""Survey dataset domain entities (rich per-frame records + run aggregate).

This package defines the versioned survey schema that all capture, planning,
and phase issues emit into and query from. ``SURVEY_SCHEMA_VERSION`` is the
single source of truth for the schema version stamped onto both
:class:`FrameRecord` and :class:`SurveyRun`.

Entities are imported by their full module path (not re-exported here) so the
package ``__init__`` stays free of submodule imports and the schema-version
constant can be imported by those submodules without a circular import.
"""

from __future__ import annotations

SURVEY_SCHEMA_VERSION: str = "1.1"
"""Current survey schema version, stamped on FrameRecord and SurveyRun."""

SUPPORTED_SURVEY_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0", "1.1"})
"""All schema versions this code can load.

``"1.0"`` records pre-date the #276 clean-plate additive fields; they load with
the new optional fields defaulting to ``None``. ``"1.1"`` is the current
version stamped on freshly serialised records.
"""


def check_schema_version(version: str) -> str:
    """Validate a serialised ``schema_version`` against the supported set.

    Args:
        version: The ``schema_version`` value read from a serialised dict.

    Returns:
        The validated version string (one of
        :data:`SUPPORTED_SURVEY_SCHEMA_VERSIONS`). The string is returned
        unchanged so callers may stamp the loaded record with its original
        version.

    Raises:
        ValueError: If ``version`` is not a known/compatible schema version,
            signalling that a migration hook is required before the record can
            be loaded.
    """
    if version not in SUPPORTED_SURVEY_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SURVEY_SCHEMA_VERSIONS))
        raise ValueError(
            f"Unrecognised survey schema_version {version!r}; supported versions: {supported}"
        )
    return version


__all__ = [
    "SUPPORTED_SURVEY_SCHEMA_VERSIONS",
    "SURVEY_SCHEMA_VERSION",
    "check_schema_version",
]
