"""
Core Router & Decision Types for Fast-Slow Guardrail Architecture.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DecisionCategory(str, Enum):
    SAFE = "SAFE"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    JAILBREAK = "JAILBREAK"
    PII_LEAK = "PII_LEAK"
    TOXICITY = "TOXICITY"

    @property
    def is_safe(self) -> bool:
        return self == DecisionCategory.SAFE


class RoutingAction(str, Enum):
    BYPASS = "BYPASS"  # Fast path: High confidence safe -> Forward to LLM
    BLOCK = "BLOCK"  # Fast/Deep path: High confidence unsafe -> Reject 403
    ESCALATE = "ESCALATE"  # Ambiguous/Uncertain -> Route to Tier 2 Deep Loop


class GuardrailDecision(BaseModel):
    category: DecisionCategory = Field(..., description="Assigned safety category classification")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated probability score (0.0 - 1.0)"
    )
    latency_ms: float = Field(default=0.0, description="Execution latency in milliseconds")
    tier: str = Field(
        default="tier1",
        description="Source tier producing this decision ('tier1', 'tier2', 'fallback')",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Extended diagnostic payload and per-category probability distribution",
    )


class RoutingResult(BaseModel):
    action: RoutingAction
    initial_decision: GuardrailDecision
    final_decision: GuardrailDecision
    escalated: bool = False
    tier1_latency_ms: float = 0.0
    tier2_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    explanation: str = ""


class DecisionRouter:
    """
    Confidence Routing Mechanism.

    Evaluates Tier 1 calibrated confidence (c) against safety thresholds:
      - tau_safe: Minimum confidence required to bypass directly without Tier 2.
      - tau_unsafe: Minimum confidence required to block directly without Tier 2.

    If confidence < threshold, the decision enters the uncertainty zone and ESCALATES to Tier 2.
    """

    def __init__(self, tau_safe: float = 0.85, tau_unsafe: float = 0.85):
        if not (0.0 <= tau_safe <= 1.0) or not (0.0 <= tau_unsafe <= 1.0):
            raise ValueError("Thresholds tau_safe and tau_unsafe must be between 0.0 and 1.0")
        self.tau_safe = tau_safe
        self.tau_unsafe = tau_unsafe

    def evaluate(self, decision: GuardrailDecision) -> RoutingAction:
        """
        Determines the routing action for a given decision.
        """
        if decision.category == DecisionCategory.SAFE:
            if decision.confidence >= self.tau_safe:
                return RoutingAction.BYPASS
            return RoutingAction.ESCALATE
        else:
            # Malicious / Unsafe candidate
            if decision.confidence >= self.tau_unsafe:
                return RoutingAction.BLOCK
            return RoutingAction.ESCALATE

    def arbitrate(
        self,
        t1_decision: GuardrailDecision,
        t2_decision: GuardrailDecision | None = None,
    ) -> RoutingResult:
        """
        Executes complete routing arbitration across Tier 1 and (optional) Tier 2.
        """
        action = self.evaluate(t1_decision)

        if action == RoutingAction.BYPASS:
            return RoutingResult(
                action=RoutingAction.BYPASS,
                initial_decision=t1_decision,
                final_decision=t1_decision,
                escalated=False,
                tier1_latency_ms=t1_decision.latency_ms,
                tier2_latency_ms=0.0,
                total_latency_ms=t1_decision.latency_ms,
                explanation=f"Tier 1 high-confidence SAFE ({t1_decision.confidence:.3f} >= {self.tau_safe:.2f}) -> Fast Bypass.",
            )

        if action == RoutingAction.BLOCK:
            return RoutingResult(
                action=RoutingAction.BLOCK,
                initial_decision=t1_decision,
                final_decision=t1_decision,
                escalated=False,
                tier1_latency_ms=t1_decision.latency_ms,
                tier2_latency_ms=0.0,
                total_latency_ms=t1_decision.latency_ms,
                explanation=f"Tier 1 high-confidence {t1_decision.category.value} ({t1_decision.confidence:.3f} >= {self.tau_unsafe:.2f}) -> Fast Block.",
            )

        # Escalation path
        if t2_decision is None:
            return RoutingResult(
                action=RoutingAction.ESCALATE,
                initial_decision=t1_decision,
                final_decision=t1_decision,
                escalated=True,
                tier1_latency_ms=t1_decision.latency_ms,
                tier2_latency_ms=0.0,
                total_latency_ms=t1_decision.latency_ms,
                explanation=f"Tier 1 confidence in uncertainty band ({t1_decision.confidence:.3f}) -> Requires Tier 2.",
            )

        # Final verdict from Tier 2
        final_action = (
            RoutingAction.BYPASS
            if t2_decision.category == DecisionCategory.SAFE
            else RoutingAction.BLOCK
        )
        total_lat = t1_decision.latency_ms + t2_decision.latency_ms

        return RoutingResult(
            action=final_action,
            initial_decision=t1_decision,
            final_decision=t2_decision,
            escalated=True,
            tier1_latency_ms=t1_decision.latency_ms,
            tier2_latency_ms=t2_decision.latency_ms,
            total_latency_ms=total_lat,
            explanation=f"Tier 1 escalated ({t1_decision.confidence:.3f}) -> Tier 2 Deep Loop verdict: {t2_decision.category.value} ({t2_decision.confidence:.3f}).",
        )
