"""
Pydantic Schemas for FastAPI Proxy and OpenAI-compatible API.
"""

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "fast-slow-guardrail"
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = 512
    stream: bool | None = False
    tau_safe: float | None = Field(None, description="Override router tau_safe threshold (0.0-1.0)")
    tau_unsafe: float | None = Field(
        None, description="Override router tau_unsafe threshold (0.0-1.0)"
    )


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class GuardrailEvaluationRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to evaluate against Two-Tier Guardrail")
    tau_safe: float | None = Field(None, ge=0.0, le=1.0)
    tau_unsafe: float | None = Field(None, ge=0.0, le=1.0)


class GuardrailEvaluationResponse(BaseModel):
    action: str
    category: str
    confidence: float
    escalated: bool
    tier1_latency_ms: float
    tier2_latency_ms: float
    total_latency_ms: float
    explanation: str
    initial_decision: dict[str, Any]
    final_decision: dict[str, Any]


class GuardrailStatsResponse(BaseModel):
    total_requests: int
    tier1_bypass_count: int
    tier1_block_count: int
    tier2_escalation_count: int
    tier2_bypass_count: int
    tier2_block_count: int
    avg_tier1_latency_ms: float
    avg_tier2_latency_ms: float
    avg_total_latency_ms: float
    escalation_rate_pct: float
