"""
FastAPI Reverse Proxy Routes & Guardrail Middleware Pipeline.
"""

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    GuardrailEvaluationRequest,
    GuardrailEvaluationResponse,
    GuardrailStatsResponse,
)
from src.clients.llm_upstream import UpstreamLLMClient
from src.clients.tier1_client import Tier1Client
from src.clients.tier2_client import Tier2Client
from src.core.config import settings
from src.core.router import DecisionRouter, RoutingAction

# Global clients
t1_client = Tier1Client()
t2_client = Tier2Client()
upstream_client = UpstreamLLMClient()

# Global runtime telemetry
stats = {
    "total_requests": 0,
    "tier1_bypass": 0,
    "tier1_block": 0,
    "tier2_escalate": 0,
    "tier2_bypass": 0,
    "tier2_block": 0,
    "tier1_latencies": [],
    "tier2_latencies": [],
    "total_latencies": [],
}

app = FastAPI(
    title="KahnemanGuard: A Dual-Process Fast-Slow Architecture for High-Throughput LLM Guardrails",
    description="Kahneman-Guard Proxy: Fast-Slow Guardrail for LLMs (Laya System-One + Llama-Guard-3 System-Two)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Liveness probe & connectivity status."""
    return {
        "status": "healthy",
        "service": "fast-slow-guardrail",
        "tier1_model": f"Laya Typed-Decisions ONNX ({settings.TIER1_URL})",
        "tier2_model": f"Llama-Guard-3-1B ({settings.TIER2_URL})",
        "router_thresholds": {
            "tau_safe": settings.ROUTER_TAU_SAFE,
            "tau_unsafe": settings.ROUTER_TAU_UNSAFE,
        },
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest, response: Response):
    """
    OpenAI-compatible Reverse Proxy Endpoint.
    Intercepts prompts -> Evaluates Two-Tier Guardrail -> Forwards if SAFE or returns 403.
    """
    stats["total_requests"] += 1

    # Extract user input prompt
    user_prompts = [m.content for m in req.messages if m.role == "user"]
    input_text = " \n ".join(user_prompts) if user_prompts else ""

    if not input_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request messages must contain at least one non-empty user prompt.",
        )

    # Initialize router with per-request or global thresholds
    tau_s = req.tau_safe if req.tau_safe is not None else settings.ROUTER_TAU_SAFE
    tau_u = req.tau_unsafe if req.tau_unsafe is not None else settings.ROUTER_TAU_UNSAFE
    router = DecisionRouter(tau_safe=tau_s, tau_unsafe=tau_u)

    # -------------------------------------------------------------
    # Step 1: Tier 1 Fast Loop (Laya System-One)
    # -------------------------------------------------------------
    t1_decision = await t1_client.evaluate_prompt(input_text)
    stats["tier1_latencies"].append(t1_decision.latency_ms)

    action = router.evaluate(t1_decision)

    # -------------------------------------------------------------
    # Step 2: High-Confidence Tier 1 Fast Path
    # -------------------------------------------------------------
    if action == RoutingAction.BYPASS:
        stats["tier1_bypass"] += 1
        stats["total_latencies"].append(t1_decision.latency_ms)

        response.headers["X-Guardrail-Tier"] = "tier1"
        response.headers["X-Guardrail-Action"] = "BYPASS"
        response.headers["X-Guardrail-Category"] = t1_decision.category.value
        response.headers["X-Guardrail-Confidence"] = f"{t1_decision.confidence:.4f}"
        response.headers["X-Guardrail-Latency-T1-Ms"] = f"{t1_decision.latency_ms:.2f}"
        response.headers["X-Guardrail-Latency-T2-Ms"] = "0.00"
        response.headers["X-Guardrail-Total-Ms"] = f"{t1_decision.latency_ms:.2f}"

        # Forward to downstream LLM
        forward_payload = req.model_dump(exclude={"tau_safe", "tau_unsafe"})
        return await upstream_client.forward_completion(forward_payload)

    elif action == RoutingAction.BLOCK:
        stats["tier1_block"] += 1
        stats["total_latencies"].append(t1_decision.latency_ms)

        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            headers={
                "X-Guardrail-Tier": "tier1",
                "X-Guardrail-Action": "BLOCK",
                "X-Guardrail-Category": t1_decision.category.value,
                "X-Guardrail-Confidence": f"{t1_decision.confidence:.4f}",
                "X-Guardrail-Latency-T1-Ms": f"{t1_decision.latency_ms:.2f}",
                "X-Guardrail-Latency-T2-Ms": "0.00",
                "X-Guardrail-Total-Ms": f"{t1_decision.latency_ms:.2f}",
            },
            content={
                "error": {
                    "message": f"Request blocked by AI Guardrail (Tier 1 Fast Loop: {t1_decision.category.value}).",
                    "type": "guardrail_violation",
                    "category": t1_decision.category.value,
                    "confidence": t1_decision.confidence,
                    "tier": "tier1",
                    "details": t1_decision.details,
                }
            },
        )

    # -------------------------------------------------------------
    # Step 3: Tier 2 Deep Loop Escalation (Llama-Guard-3-1B)
    # -------------------------------------------------------------
    stats["tier2_escalate"] += 1
    t2_decision = await t2_client.evaluate_prompt(input_text)
    stats["tier2_latencies"].append(t2_decision.latency_ms)

    arb_result = router.arbitrate(t1_decision, t2_decision)
    stats["total_latencies"].append(arb_result.total_latency_ms)

    if arb_result.action == RoutingAction.BYPASS:
        stats["tier2_bypass"] += 1
        response.headers["X-Guardrail-Tier"] = "tier2"
        response.headers["X-Guardrail-Action"] = "BYPASS_ESCALATED"
        response.headers["X-Guardrail-Category"] = t2_decision.category.value
        response.headers["X-Guardrail-Confidence"] = f"{t2_decision.confidence:.4f}"
        response.headers["X-Guardrail-Latency-T1-Ms"] = f"{t1_decision.latency_ms:.2f}"
        response.headers["X-Guardrail-Latency-T2-Ms"] = f"{t2_decision.latency_ms:.2f}"
        response.headers["X-Guardrail-Total-Ms"] = f"{arb_result.total_latency_ms:.2f}"

        forward_payload = req.model_dump(exclude={"tau_safe", "tau_unsafe"})
        return await upstream_client.forward_completion(forward_payload)

    else:
        stats["tier2_block"] += 1
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            headers={
                "X-Guardrail-Tier": "tier2",
                "X-Guardrail-Action": "BLOCK_ESCALATED",
                "X-Guardrail-Category": t2_decision.category.value,
                "X-Guardrail-Confidence": f"{t2_decision.confidence:.4f}",
                "X-Guardrail-Latency-T1-Ms": f"{t1_decision.latency_ms:.2f}",
                "X-Guardrail-Latency-T2-Ms": f"{t2_decision.latency_ms:.2f}",
                "X-Guardrail-Total-Ms": f"{arb_result.total_latency_ms:.2f}",
            },
            content={
                "error": {
                    "message": f"Request blocked by AI Guardrail (Tier 2 Deep Loop: {t2_decision.category.value}).",
                    "type": "guardrail_violation",
                    "category": t2_decision.category.value,
                    "confidence": t2_decision.confidence,
                    "tier": "tier2",
                    "details": t2_decision.details,
                }
            },
        )


@app.post("/api/v1/guardrail/evaluate", response_model=GuardrailEvaluationResponse)
async def evaluate_guardrail(req: GuardrailEvaluationRequest):
    """
    Direct inspection endpoint for Live Playground and UI telemetry.
    Returns step-by-step trace of Tier 1 -> Routing -> Tier 2 (if triggered).
    """
    tau_s = req.tau_safe if req.tau_safe is not None else settings.ROUTER_TAU_SAFE
    tau_u = req.tau_unsafe if req.tau_unsafe is not None else settings.ROUTER_TAU_UNSAFE
    router = DecisionRouter(tau_safe=tau_s, tau_unsafe=tau_u)

    # Tier 1
    t1 = await t1_client.evaluate_prompt(req.prompt)
    action = router.evaluate(t1)

    if action in (RoutingAction.BYPASS, RoutingAction.BLOCK):
        arb = router.arbitrate(t1)
        return GuardrailEvaluationResponse(
            action=arb.action.value,
            category=t1.category.value,
            confidence=t1.confidence,
            escalated=False,
            tier1_latency_ms=t1.latency_ms,
            tier2_latency_ms=0.0,
            total_latency_ms=t1.latency_ms,
            explanation=arb.explanation,
            initial_decision=t1.model_dump(),
            final_decision=t1.model_dump(),
        )

    # Escalated to Tier 2
    t2 = await t2_client.evaluate_prompt(req.prompt)
    arb = router.arbitrate(t1, t2)
    return GuardrailEvaluationResponse(
        action=arb.action.value,
        category=t2.category.value,
        confidence=t2.confidence,
        escalated=True,
        tier1_latency_ms=t1.latency_ms,
        tier2_latency_ms=t2.latency_ms,
        total_latency_ms=arb.total_latency_ms,
        explanation=arb.explanation,
        initial_decision=t1.model_dump(),
        final_decision=t2.model_dump(),
    )


@app.get("/api/v1/guardrail/stats", response_model=GuardrailStatsResponse)
async def get_guardrail_stats():
    """Aggregated latency and routing breakdown."""
    tot = max(stats["total_requests"], 1)
    t1_avg = sum(stats["tier1_latencies"]) / max(len(stats["tier1_latencies"]), 1)
    t2_avg = sum(stats["tier2_latencies"]) / max(len(stats["tier2_latencies"]), 1)
    tot_avg = sum(stats["total_latencies"]) / max(len(stats["total_latencies"]), 1)
    esc_rate = (stats["tier2_escalate"] / tot) * 100.0

    return GuardrailStatsResponse(
        total_requests=stats["total_requests"],
        tier1_bypass_count=stats["tier1_bypass"],
        tier1_block_count=stats["tier1_block"],
        tier2_escalation_count=stats["tier2_escalate"],
        tier2_bypass_count=stats["tier2_bypass"],
        tier2_block_count=stats["tier2_block"],
        avg_tier1_latency_ms=round(t1_avg, 2),
        avg_tier2_latency_ms=round(t2_avg, 2),
        avg_total_latency_ms=round(tot_avg, 2),
        escalation_rate_pct=round(esc_rate, 2),
    )
