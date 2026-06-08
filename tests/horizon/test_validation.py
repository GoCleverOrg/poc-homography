"""Tests for the injectable vision-validation hook."""

from __future__ import annotations

import numpy as np

from poc_homography.horizon import (
    FramePlacement,
    HorizonEstimate,
    HorizonValidator,
    NullHorizonValidator,
    ValidationOutcome,
)


def _estimate() -> HorizonEstimate:
    return HorizonEstimate(placement=FramePlacement.IN_FRAME, image_height=1440, row=700.0)


class TestNullValidator:
    def test_abstains(self):
        outcome = NullHorizonValidator().validate(np.zeros((4, 4), dtype=np.uint8), _estimate())
        assert outcome.agrees is None
        assert outcome.confidence == 0.0

    def test_satisfies_protocol(self):
        assert isinstance(NullHorizonValidator(), HorizonValidator)


class TestInjectedValidator:
    def test_custom_validator_is_dependency_injected(self):
        class _StubValidator:
            def validate(self, image, estimate):
                return ValidationOutcome(agrees=True, confidence=0.8, detail="stub agrees")

        validator: HorizonValidator = _StubValidator()
        outcome = validator.validate(np.zeros((4, 4), dtype=np.uint8), _estimate())
        assert outcome.agrees is True
        assert outcome.confidence == 0.8
        assert isinstance(validator, HorizonValidator)
