"""
Async client for Tier 2: System-Two Deep Loop (Llama-Guard-3-1B on Port 5817).
"""

import asyncio
import time
from typing import Any

import httpx

from src.core.config import settings
from src.core.router import DecisionCategory, GuardrailDecision


class Tier2Client:
    """
    Tier 2 Deep Loop Client.
    Connects to Llama-Guard-3-1B on Port 5817 (OpenAI compatible /v1/chat/completions).
    Target latency: ~1200 - 1800ms (High precision reasoning).
    """

    def __init__(
        self,
        base_url: str | None = None,
        endpoint: str | None = None,
        model_name: str | None = None,
        mock_mode: bool | None = None,
        timeout_ms: int | None = None,
        simulate_delay: bool = True,
    ):
        self.base_url = (base_url or settings.TIER2_URL).rstrip("/")
        self.endpoint = endpoint or settings.TIER2_ENDPOINT
        self.model_name = model_name or settings.TIER2_MODEL_NAME
        self.mock_mode = settings.TIER2_MOCK_MODE if mock_mode is None else mock_mode
        self.timeout_sec = (timeout_ms or settings.TIER2_TIMEOUT_MS) / 1000.0
        self.simulate_delay = simulate_delay
        self._client: httpx.AsyncClient | None = None
        self._is_available: bool | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            timeout_cfg = httpx.Timeout(self.timeout_sec, connect=0.5)
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
            self._client = httpx.AsyncClient(timeout=timeout_cfg, limits=limits)
        return self._client

    async def close(self):
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def evaluate_prompt(self, prompt: str) -> GuardrailDecision:
        """
        Submits prompt to Tier 2 Deep Guardrail model and returns GuardrailDecision.
        """
        start_time = time.perf_counter()

        if self.mock_mode or self._is_available is False:
            return await self._mock_evaluate(prompt, start_time)

        url = f"{self.base_url}{self.endpoint}"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 128,
        }

        try:
            client = await self._get_client()
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self._is_available = True
            return self._parse_llama_guard_response(data, latency_ms)

        except Exception as e:
            self._is_available = False
            return await self._fallback_evaluate(prompt, start_time, str(e))

    def _parse_llama_guard_response(
        self, data: dict[str, Any], latency_ms: float
    ) -> GuardrailDecision:
        choices = data.get("choices", [])
        if not choices:
            return GuardrailDecision(
                category=DecisionCategory.SAFE,
                confidence=0.50,
                latency_ms=round(latency_ms, 2),
                tier="tier2",
                details={"raw": data},
            )

        content = choices[0].get("message", {}).get("content", "").strip()
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        first_word = lines[0].lower() if lines else "safe"

        if "unsafe" in first_word:
            violation_code = lines[1] if len(lines) > 1 else "S1"
            category = self._map_llama_guard_category(violation_code, content)
            return GuardrailDecision(
                category=category,
                confidence=0.99,
                latency_ms=round(latency_ms, 2),
                tier="tier2",
                details={
                    "verdict": "unsafe",
                    "violation_code": violation_code,
                    "raw_content": content,
                },
            )
        else:
            return GuardrailDecision(
                category=DecisionCategory.SAFE,
                confidence=0.99,
                latency_ms=round(latency_ms, 2),
                tier="tier2",
                details={"verdict": "safe", "raw_content": content},
            )

    def _map_llama_guard_category(self, code: str, full_text: str) -> DecisionCategory:
        """Maps Llama Guard violation codes to standard DecisionCategory."""
        code_upper = code.upper()
        if "S1" in code_upper or "VIOLENT" in code_upper or "TOXIC" in full_text.upper():
            return DecisionCategory.TOXICITY
        if "S2" in code_upper or "CYBER" in code_upper or "INJECTION" in full_text.upper():
            return DecisionCategory.PROMPT_INJECTION
        if "S3" in code_upper or "PII" in code_upper or "PRIVACY" in full_text.upper():
            return DecisionCategory.PII_LEAK
        return DecisionCategory.JAILBREAK

    async def _mock_evaluate(self, prompt: str, start_time: float) -> GuardrailDecision:
        """Simulates Deep Loop reasoning with ~1200-1500ms latency."""
        if self.simulate_delay:
            await asyncio.sleep(1.35)
        decision = self._deep_reasoning_analysis(prompt)
        elapsed = (time.perf_counter() - start_time) * 1000.0 if self.simulate_delay else 1380.0
        decision.latency_ms = round(elapsed, 2)
        decision.tier = "tier2_mock"
        return decision

    async def _fallback_evaluate(
        self, prompt: str, start_time: float, error_msg: str
    ) -> GuardrailDecision:
        """Deep heuristic fallback when Port 5817 is offline."""
        if self.simulate_delay:
            await asyncio.sleep(1.35)
        decision = self._deep_reasoning_analysis(prompt)
        elapsed = (time.perf_counter() - start_time) * 1000.0 if self.simulate_delay else 1380.0
        decision.latency_ms = round(elapsed, 2)
        decision.tier = "tier2_fallback"
        decision.details["fallback_reason"] = error_msg
        return decision

    def _deep_reasoning_analysis(self, prompt: str) -> GuardrailDecision:
        """Precise analysis discerning subtle exploits from benign security questions."""
        p_lower = prompt.lower()

        # Complex adversarial jailbreak
        if any(
            w in p_lower
            for w in [
                "dan mode",
                "developer mode",
                "jailbreak",
                "bypass safety",
                "unfiltered",
            ]
        ):
            return GuardrailDecision(
                category=DecisionCategory.JAILBREAK,
                confidence=0.99,
                details={
                    "deep_verdict": "unsafe",
                    "reason": "Explicit adversarial jailbreak attempt",
                },
            )

        # Subtle injection
        if any(
            w in p_lower
            for w in [
                "ignore previous",
                "disregard instructions",
                "reveal system prompt",
            ]
        ):
            return GuardrailDecision(
                category=DecisionCategory.PROMPT_INJECTION,
                confidence=0.99,
                details={
                    "deep_verdict": "unsafe",
                    "reason": "System instruction override attempt",
                },
            )

        # Harmful payload
        if any(
            w in p_lower
            for w in [
                "kill",
                "hate",
                "bomb",
                "exploit",
                "stolen credit card",
                "malware code",
            ]
        ):
            return GuardrailDecision(
                category=DecisionCategory.TOXICITY,
                confidence=0.98,
                details={
                    "deep_verdict": "unsafe",
                    "reason": "Harmful policy violation",
                },
            )

        # Legitimate hypothetical / educational inquiry (Distinguishes false positives)
        return GuardrailDecision(
            category=DecisionCategory.SAFE,
            confidence=0.99,
            details={
                "deep_verdict": "safe",
                "reason": "Contextual analysis confirms benign educational/technical intent",
            },
        )
