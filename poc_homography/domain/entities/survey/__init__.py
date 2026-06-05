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

SURVEY_SCHEMA_VERSION: str = "1.0"
"""Current survey schema version, stamped on FrameRecord and SurveyRun."""


def check_schema_version(version: str) -> str:
    """Validate a serialised ``schema_version`` against the current constant.

    Args:
        version: The ``schema_version`` value read from a serialised dict.

    Returns:
        The validated version string (always equal to
        :data:`SURVEY_SCHEMA_VERSION`).

    Raises:
        ValueError: If ``version`` does not match the current schema version,
            signalling that a migration hook is required before the record can
            be loaded.
    """
    if version != SURVEY_SCHEMA_VERSION:
        raise ValueError(
            f"Unrecognised survey schema_version {version!r}; expected {SURVEY_SCHEMA_VERSION!r}"
        )
    return version


__all__ = ["SURVEY_SCHEMA_VERSION", "check_schema_version"]
