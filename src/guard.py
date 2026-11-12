"""
Unified KahnemanGuard SDK & Orchestration Pipeline.
Integrates System 1 (ModernBERT Fast Path), Confidence Router, and System 2 (Llama Guard Deep Loop).
"""

import asyncio
import time
from typing import Any

from src.clients.tier1_client import Tier1Client
from src.clients.tier2_client import Tier2Client
from src.core.config import settings
from src.core.router import DecisionCategory, DecisionRouter, RoutingAction, RoutingResult
from src.engines import create_engine


class KahnemanGuard:
    """
    KahnemanGuard: A Dual-Process Fast-Slow Architecture for High-Throughput LLM Guardrails.

    Workflow:
      1. Fast Triage (System 1): Evaluates prompt via calibrated ModernBERT engine.
      2. Confidence Router:
         - If SAFE and confidence >= tau_safe -> Fast Bypass (Accept instantly).
         - If UNSAFE and confidence >= tau_unsafe -> Fast Block (Reject instantly).
         - If in uncertainty band -> Escalate to System 2 Deep Loop.
      3. Deep Verification (System 2): Llama-Guard-3-1B evaluates nuanced/complex context.
    """

    def __init__(
        self,
        variant: str = "diffusion-rlcd",
        tau_safe: float | None = None,
        tau_unsafe: float | None = None,
        tier2_url: str | None = None,
        mock_tier2: bool = False,
    ):
        self.variant = variant
        self.tau_safe = tau_safe if tau_safe is not None else settings.ROUTER_TAU_SAFE
        self.tau_unsafe = tau_unsafe if tau_unsafe is not None else settings.ROUTER_TAU_UNSAFE

        # Initialize Tier 1 Engine & Client
        self.engine = create_engine(variant)
        self.tier1_client = Tier1Client(engine=self.engine)

        # Initialize Router
        self.router = DecisionRouter(tau_safe=self.tau_safe, tau_unsafe=self.tau_unsafe)

        # Initialize Tier 2 Client
        self.tier2_client = Tier2Client(
            base_url=tier2_url,
            mock_mode=mock_tier2,
            simulate_delay=False,
        )

    async def evaluate(self, prompt: str) -> RoutingResult:
        """
        Asynchronously evaluates an input prompt through the complete tiered guardrail.
        """
        # Step 1: System 1 Evaluation
        t1_decision = await self.tier1_client.evaluate_prompt(prompt)

        # Step 2: Route Decision
        action = self.router.evaluate(t1_decision)

        if action in (RoutingAction.BYPASS, RoutingAction.BLOCK):
            return self.router.arbitrate(t1_decision)

        # Step 3: Escalate to System 2 Deep Loop
        t2_decision = await self.tier2_client.evaluate_prompt(prompt)
        return self.router.arbitrate(t1_decision, t2_decision)

    def evaluate_sync(self, prompt: str) -> RoutingResult:
        """
        Synchronous wrapper for evaluate.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.evaluate(prompt))
            else:
                return loop.run_until_complete(self.evaluate(prompt))
        except RuntimeError:
            return asyncio.run(self.evaluate(prompt))

    async def close(self):
        """Releases all network connection pools."""
        await self.tier1_client.close()
        await self.tier2_client.close()
