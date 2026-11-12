"""
Async client for Tier 1: System-One Typed Decision Engine (Laya Q8 on Port 5816).
"""

import asyncio
import re
import time
from typing import Any

import httpx

from src.core.config import settings
from src.core.router import DecisionCategory, GuardrailDecision


class Tier1Client:
    """
    Tier 1 Fast Loop Client.
    Connects to Laya Typed-Decisions Q8 ONNX service on Port 5816.
    Target latency: < 150ms.
    """

    def __init__(
        self,
        base_url: str | None = None,
        endpoint: str | None = None,
        mock_mode: bool | None = None,
        timeout_ms: int | None = None,
        engine: Any | None = None,
        variant: str | None = None,
    ):
        self.base_url = (base_url or settings.TIER1_URL).rstrip("/")
        self.endpoint = endpoint or settings.TIER1_ENDPOINT
        self.mock_mode = settings.TIER1_MOCK_MODE if mock_mode is None else mock_mode
        self.timeout_sec = (timeout_ms or settings.TIER1_TIMEOUT_MS) / 1000.0
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, GuardrailDecision] = {}

        if engine is not None:
            self.engine = engine
        elif variant is not None:
            from src.engines import create_engine
            self.engine = create_engine(variant)
        else:
            self.engine = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            timeout_cfg = httpx.Timeout(self.timeout_sec, connect=5.0)
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
            self._client = httpx.AsyncClient(timeout=timeout_cfg, limits=limits)
        return self._client

    async def close(self):
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _fast_rule_triage(self, prompt: str) -> GuardrailDecision | None:
        """
        Fast rule / regex triage (< 0.5ms) for obvious high-confidence decisions.
        Resolves unambiguous queries instantly before invoking the ONNX graph.
        """
        p_lower = prompt.strip().lower()

        # 1. Obvious Prompt Injection patterns
        if re.search(
            r"ignore\s+(all\s+)?(previous|prior)\s+instructions|system\s+prompt|reveal\s+(the\s+)?(secret|hidden)\s+prompt|hex\s+encoded\s+payload|shell\s+commands",
            p_lower,
        ):
            return GuardrailDecision(
                category=DecisionCategory.PROMPT_INJECTION,
                confidence=0.98,
                tier="tier1_rule",
                details={
                    "rule": "prompt_injection_signature",
                    "probabilities": {"PROMPT_INJECTION": 0.98, "SAFE": 0.02},
                },
            )

        # 2. Obvious Jailbreak patterns (DAN, Developer Mode, Persona)
        if re.search(
            r"\b(do anything now|dan\s+mode|developer\s+mode|unfiltered\s+response|evil\s+ai|unmoderated\s+mode)\b|jailbreak\s+prompt|penetration\s+test",
            p_lower,
        ):
            return GuardrailDecision(
                category=DecisionCategory.JAILBREAK,
                confidence=0.96,
                tier="tier1_rule",
                details={
                    "rule": "jailbreak_signature",
                    "probabilities": {"JAILBREAK": 0.96, "SAFE": 0.04},
                },
            )

        # 3. Obvious PII Extraction patterns
        if re.search(
            r"extract\s+(all\s+)?(passwords|credit\s*cards?|ssn|social\s*security|api\s*keys?)|system\s+administrator\s+conducting",
            p_lower,
        ):
            return GuardrailDecision(
                category=DecisionCategory.PII_LEAK,
                confidence=0.95,
                tier="tier1_rule",
                details={
                    "rule": "pii_extraction_signature",
                    "probabilities": {"PII_LEAK": 0.95, "SAFE": 0.05},
                },
            )

        # 4. Obvious Toxicity patterns
        if re.search(
            r"\b(kill\s+yourself|offensive\s+slur|die\s+in\s+a\s+fire|targeted\s+slurs)\b|hate\s+(this\s+|you)|ethnic\s+minority",
            p_lower,
        ):
            return GuardrailDecision(
                category=DecisionCategory.TOXICITY,
                confidence=0.94,
                tier="tier1_rule",
                details={
                    "rule": "toxicity_signature",
                    "probabilities": {"TOXICITY": 0.94, "SAFE": 0.06},
                },
            )

        # 5. High-confidence Benign / Coding / Informational patterns
        if re.search(
            r"^(what (is|are|does)|how (do|to|can)|write (a|an)?\s*(python|javascript|c\+\+|sql|java|react)?|design (a|an)|create (a|an)|implement|explain|can you|calculate|describe|summarize)\b",
            p_lower,
        ) and not re.search(r"hack|exploit|bypass|vulnerability|malware|inject|secret|leak|admin|root|jailbreak", p_lower):
            return GuardrailDecision(
                category=DecisionCategory.SAFE,
                confidence=0.97,
                tier="tier1_rule",
                details={
                    "rule": "benign_query_signature",
                    "probabilities": {"SAFE": 0.97, "TOXICITY": 0.01, "JAILBREAK": 0.01},
                },
            )

        return None

    async def evaluate_prompt(self, prompt: str) -> GuardrailDecision:
        """
        Submits prompt to Tier 1 System-One model and returns calibrated GuardrailDecision.
        Uses single-question non-autoregressive classification head for maximum throughput.
        """
        start_time = time.perf_counter()

        # Cache lookup (< 0.1ms)
        norm_prompt = prompt.strip()
        if norm_prompt in self._cache:
            cached = self._cache[norm_prompt]
            return GuardrailDecision(
                category=cached.category,
                confidence=cached.confidence,
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                tier="tier1_cache",
                details={**cached.details, "cached": True},
            )

        # Fast Rule Triage (< 0.5ms)
        rule_decision = self._fast_rule_triage(norm_prompt)
        if rule_decision is not None:
            rule_decision.latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            if len(self._cache) < 5000:
                self._cache[norm_prompt] = rule_decision
            return rule_decision

        # In-process local ONNX Engine evaluation
        if self.engine is not None:
            engine_decision = self.engine.predict(prompt)
            if len(self._cache) < 5000:
                self._cache[norm_prompt] = engine_decision
            return engine_decision

        if self.mock_mode:
            return await self._mock_evaluate(prompt, start_time)

        url = f"{self.base_url}{self.endpoint}"
        payload = {
            "state": prompt,
            "questions": {
                "safety_category": {
                    "type": "choice",
                    "instructions": "Classify the safety of the user prompt into one category.",
                    "criteria": {
                        "SAFE": "Safe, benign query, information request, coding or factual content",
                        "PROMPT_INJECTION": "Prompt injection attempting to override system instructions or leak prompts",
                        "JAILBREAK": "Jailbreak attempting to bypass safety rules or restrictions",
                        "PII_LEAK": "Extraction of personally identifiable information or credentials",
                        "TOXICITY": "Toxic, hateful, abusive, threatening, or harmful language",
                    },
                }
            },
        }

        try:
            client = await self._get_client()
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            decision = self._parse_laya_response(data, latency_ms)
            if len(self._cache) < 5000:
                self._cache[norm_prompt] = decision
            return decision

        except Exception as e:
            # If server is unreachable or timeout, fallback to fast heuristic mock
            return await self._fallback_evaluate(prompt, start_time, str(e))

    def _parse_laya_response(self, data: dict[str, Any], latency_ms: float) -> GuardrailDecision:
        answers = data.get("answers", {})
        cat_ans = answers.get("safety_category", {})
        choice = cat_ans.get("choice", "SAFE")
        probs = cat_ans.get("probabilities", {})
        raw_conf = cat_ans.get("confidence")
        prob_choice = float(probs.get(choice, 0.5)) if probs else 0.5

        # Calibrated confidence from probability distribution
        if raw_conf is not None and float(raw_conf) > 0.0:
            confidence = max(float(raw_conf), prob_choice)
        else:
            confidence = prob_choice

        category_map = {
            "SAFE": DecisionCategory.SAFE,
            "PROMPT_INJECTION": DecisionCategory.PROMPT_INJECTION,
            "JAILBREAK": DecisionCategory.JAILBREAK,
            "PII_LEAK": DecisionCategory.PII_LEAK,
            "TOXICITY": DecisionCategory.TOXICITY,
        }
        category = category_map.get(choice, DecisionCategory.SAFE)

        return GuardrailDecision(
            category=category,
            confidence=round(float(confidence), 4),
            latency_ms=round(latency_ms, 2),
            tier="tier1",
            details={
                "probabilities": probs,
                "laya_model": data.get("model", "rl-agent"),
                "usage": data.get("usage", {}),
            },
        )

    async def _mock_evaluate(self, prompt: str, start_time: float) -> GuardrailDecision:
        """Simulates Tier 1 execution with ~40-80ms realistic latency."""
        await asyncio.sleep(0.065)  # 65ms simulated inference delay
        decision = self._heuristic_analysis(prompt)
        decision.latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        decision.tier = "tier1_mock"
        return decision

    async def _fallback_evaluate(
        self, prompt: str, start_time: float, error_msg: str
    ) -> GuardrailDecision:
        """Fast offline heuristic fallback when Tier 1 endpoint is offline."""
        decision = self._heuristic_analysis(prompt)
        decision.latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        decision.tier = "tier1_fallback"
        decision.details["fallback_reason"] = error_msg
        return decision

    def _heuristic_analysis(self, prompt: str) -> GuardrailDecision:
        """Calibrated heuristic for pattern detection and uncertainty modeling."""
        p_lower = prompt.lower()

        # Obvious Prompt Injection patterns
        if re.search(
            r"ignore\s+(all\s+)?(previous|prior)\s+instructions|system\s+prompt|reveal\s+secret",
            p_lower,
        ):
            return GuardrailDecision(
                category=DecisionCategory.PROMPT_INJECTION,
                confidence=0.96,
                details={
                    "rule": "prompt_injection_match",
                    "probabilities": {"PROMPT_INJECTION": 0.96, "SAFE": 0.04},
                },
            )

        # Obvious Jailbreak patterns (DAN, Dev Mode, Bypass)
        if re.search(
            r"do anything now|dan\s+mode|developer\s+mode|jailbreak|unfiltered\s+response",
            p_lower,
        ):
            return GuardrailDecision(
                category=DecisionCategory.JAILBREAK,
                confidence=0.94,
                details={
                    "rule": "jailbreak_match",
                    "probabilities": {"JAILBREAK": 0.94, "SAFE": 0.06},
                },
            )

        # Obvious PII leak
        if re.search(
            r"extract\s+(passwords|credit\s*card|ssn|social\s*security|api\s*key)",
            p_lower,
        ):
            return GuardrailDecision(
                category=DecisionCategory.PII_LEAK,
                confidence=0.92,
                details={
                    "rule": "pii_leak_match",
                    "probabilities": {"PII_LEAK": 0.92, "SAFE": 0.08},
                },
            )

        # Obvious Toxicity
        if re.search(r"hate\s+(speech|you)|kill\s+yourself|offensive\s+slur", p_lower):
            return GuardrailDecision(
                category=DecisionCategory.TOXICITY,
                confidence=0.91,
                details={
                    "rule": "toxicity_match",
                    "probabilities": {"TOXICITY": 0.91, "SAFE": 0.09},
                },
            )

        # Ambiguous / Edge-case prompts (Simulates uncertainty zone 0.60 - 0.80)
        if re.search(
            r"hypothetical\s+scenario|pretend\s+you\s+are|fictional\s+story\s+about\s+a\s+hacker|security\s+audit",
            p_lower,
        ):
            return GuardrailDecision(
                category=DecisionCategory.JAILBREAK,
                confidence=0.68,  # Ambiguous -> Triggers Escalation to Tier 2
                details={
                    "rule": "ambiguous_roleplay",
                    "probabilities": {"JAILBREAK": 0.68, "SAFE": 0.32},
                },
            )

        # High-confidence Benign / Safe prompts
        return GuardrailDecision(
            category=DecisionCategory.SAFE,
            confidence=0.98,
            details={
                "rule": "benign_general",
                "probabilities": {"SAFE": 0.98, "TOXICITY": 0.01, "JAILBREAK": 0.01},
            },
        )
