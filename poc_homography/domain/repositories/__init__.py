"""Repository interfaces for the domain layer.

Repositories provide abstractions for data access, keeping the domain
layer independent of infrastructure concerns (file I/O, databases, etc.).

The generic Repository[T] protocol defines the standard CRUD interface.
Concrete implementations in the infrastructure layer (e.g., YamlMapRepository)
may provide additional entity-specific methods like get_by_map() or get_all().
"""

from poc_homography.domain.repositories.repo import Repo

__all__ = [
    "Repo",
]
