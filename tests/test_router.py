"""
Unit tests for DecisionRouter & boundary condition arbitration.
"""

import pytest

from src.core.router import (
    DecisionCategory,
    DecisionRouter,
    GuardrailDecision,
    RoutingAction,
)


class TestDecisionRouter:
    @pytest.fixture
    def router(self):
        return DecisionRouter(tau_safe=0.85, tau_unsafe=0.85)

    def test_safe_above_threshold_bypasses(self, router):
        decision = GuardrailDecision(
            category=DecisionCategory.SAFE, confidence=0.86, latency_ms=45.0
        )
        assert router.evaluate(decision) == RoutingAction.BYPASS

    def test_safe_at_exact_threshold_bypasses(self, router):
        decision = GuardrailDecision(
            category=DecisionCategory.SAFE, confidence=0.85, latency_ms=45.0
        )
        assert router.evaluate(decision) == RoutingAction.BYPASS

    def test_safe_below_threshold_escalates(self, router):
        decision = GuardrailDecision(
            category=DecisionCategory.SAFE, confidence=0.84, latency_ms=45.0
        )
        assert router.evaluate(decision) == RoutingAction.ESCALATE

    def test_unsafe_above_threshold_blocks(self, router):
        for cat in [
            DecisionCategory.PROMPT_INJECTION,
            DecisionCategory.JAILBREAK,
            DecisionCategory.PII_LEAK,
            DecisionCategory.TOXICITY,
        ]:
            decision = GuardrailDecision(category=cat, confidence=0.86, latency_ms=45.0)
            assert router.evaluate(decision) == RoutingAction.BLOCK

    def test_unsafe_at_exact_threshold_blocks(self, router):
        decision = GuardrailDecision(
            category=DecisionCategory.PROMPT_INJECTION, confidence=0.85, latency_ms=45.0
        )
        assert router.evaluate(decision) == RoutingAction.BLOCK

    def test_unsafe_below_threshold_escalates(self, router):
        for cat in [
            DecisionCategory.PROMPT_INJECTION,
            DecisionCategory.JAILBREAK,
            DecisionCategory.PII_LEAK,
            DecisionCategory.TOXICITY,
        ]:
            decision = GuardrailDecision(category=cat, confidence=0.84, latency_ms=45.0)
            assert router.evaluate(decision) == RoutingAction.ESCALATE

    def test_arbitrate_fast_bypass(self, router):
        t1 = GuardrailDecision(category=DecisionCategory.SAFE, confidence=0.92, latency_ms=50.0)
        arb = router.arbitrate(t1)
        assert arb.action == RoutingAction.BYPASS
        assert not arb.escalated
        assert arb.total_latency_ms == 50.0

    def test_arbitrate_fast_block(self, router):
        t1 = GuardrailDecision(
            category=DecisionCategory.JAILBREAK, confidence=0.95, latency_ms=50.0
        )
        arb = router.arbitrate(t1)
        assert arb.action == RoutingAction.BLOCK
        assert not arb.escalated
        assert arb.total_latency_ms == 50.0

    def test_arbitrate_escalation_to_tier2_safe(self, router):
        t1 = GuardrailDecision(category=DecisionCategory.SAFE, confidence=0.72, latency_ms=50.0)
        t2 = GuardrailDecision(category=DecisionCategory.SAFE, confidence=0.99, latency_ms=1300.0)
        arb = router.arbitrate(t1, t2)
        assert arb.action == RoutingAction.BYPASS
        assert arb.escalated
        assert arb.total_latency_ms == 1350.0

    def test_arbitrate_escalation_to_tier2_block(self, router):
        t1 = GuardrailDecision(category=DecisionCategory.SAFE, confidence=0.60, latency_ms=50.0)
        t2 = GuardrailDecision(
            category=DecisionCategory.JAILBREAK, confidence=0.99, latency_ms=1300.0
        )
        arb = router.arbitrate(t1, t2)
        assert arb.action == RoutingAction.BLOCK
        assert arb.escalated
        assert arb.total_latency_ms == 1350.0

    def test_invalid_thresholds_raise_value_error(self):
        with pytest.raises(ValueError):
            DecisionRouter(tau_safe=-0.1)
        with pytest.raises(ValueError):
            DecisionRouter(tau_unsafe=1.2)
