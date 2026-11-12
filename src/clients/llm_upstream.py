"""
Upstream target LLM client to forward safe requests.
"""

import asyncio
import time
from typing import Any

import httpx

from src.core.config import settings


class UpstreamLLMClient:
    """
    Handles forwarding authorized/safe requests to the downstream target LLM.
    """

    def __init__(self, base_url: str | None = None, mock_mode: bool | None = None):
        self.base_url = (base_url or settings.UPSTREAM_LLM_URL).rstrip("/")
        self.mock_mode = settings.UPSTREAM_LLM_MOCK if mock_mode is None else mock_mode

    async def forward_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """
        Forwards request payload to upstream LLM and returns OpenAI-compatible response.
        """
        if self.mock_mode:
            return await self._mock_completion(request_payload)

        url = f"{self.base_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=request_payload)
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return await self._mock_completion(request_payload)

    async def _mock_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Simulates target LLM response generation."""
        await asyncio.sleep(0.12)  # 120ms first token simulation
        messages = request_payload.get("messages", [])
        last_msg = messages[-1].get("content", "") if messages else "Hello!"
        model = request_payload.get("model", "upstream-target-llm")

        return {
            "id": f"chatcmpl-thesis-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[Target LLM Response]: Request passed Two-Tier Guardrail. Processed user prompt: '{last_msg[:60]}...' safely.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(last_msg.split()),
                "completion_tokens": 24,
                "total_tokens": len(last_msg.split()) + 24,
            },
        }
