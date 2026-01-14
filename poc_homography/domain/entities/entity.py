"""Base entity protocol for dict-serializable domain entities.

This protocol defines the common interface that all persistable entities
must implement to work with generic repository implementations.
"""

from typing import Any, Protocol

from typing_extensions import Self


class Entity(Protocol):
    """Protocol for domain entities that can be serialized to/from dicts.

    Entities implementing this protocol can be used with generic repository
    base classes that provide automatic serialization.

    Required interface:
        - id: Unique identifier property
        - to_dict(): Convert entity to dictionary
        - from_dict(): Class method to construct entity from dictionary
    """

    @property
    def id(self) -> str:
        """Unique identifier for this entity."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to a dictionary for serialization."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct entity from a dictionary."""
        ...
