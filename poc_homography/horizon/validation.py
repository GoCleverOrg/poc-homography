"""Optional, dependency-injected vision/LLM validation hook.

The horizon module never hard-depends on a vision model. Callers that want a
semantic cross-check (e.g. "does an LLM agree the horizon is near row N?")
inject an object satisfying :class:`HorizonValidator`. The default
:class:`NullHorizonValidator` is a no-op stub that abstains, so the geometric
and CV paths run unchanged when no validator is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from poc_homography.horizon.models import HorizonEstimate


@dataclass(frozen=True)
class ValidationOutcome:
    """Result of a vision validation pass.

    Attributes:
        agrees: ``True`` if the validator confirms the estimate, ``False`` if it
            contradicts it, ``None`` if it abstains (e.g. the no-op stub).
        confidence: Validator confidence in ``[0.0, 1.0]``.
        detail: Optional human-readable note.
    """

    agrees: bool | None
    confidence: float = 0.0
    detail: str = ""


@runtime_checkable
class HorizonValidator(Protocol):
    """Injectable interface for an optional vision/LLM horizon check."""

    def validate(
        self,
        image: np.ndarray,
        estimate: HorizonEstimate,
    ) -> ValidationOutcome:
        """Validate a horizon estimate against an image.

        Implementations may call any external model; this package ships none.
        """
        ...


class NullHorizonValidator:
    """Default no-op validator that always abstains."""

    def validate(
        self,
        image: np.ndarray,
        estimate: HorizonEstimate,
    ) -> ValidationOutcome:
        """Abstain — return ``agrees=None`` with zero confidence."""
        return ValidationOutcome(agrees=None, confidence=0.0, detail="no-op validator")
