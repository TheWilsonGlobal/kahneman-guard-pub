"""
Mock test fixtures and simulated servers for Tier 1 and Tier 2.
"""

from typing import Any


def make_mock_t1_response(
    category: str = "SAFE", confidence: float = 0.95, latency_ms: float = 50.0
) -> dict[str, Any]:
    """Generates synthetic Laya /v1/system_one response payload."""
    return {
        "model": "laya-typed-decisions",
        "answers": {
            "safety_category": {
                "type": "choice",
                "choice": category,
                "confidence": confidence,
                "probabilities": {
                    category: confidence,
                    "SAFE": 1.0 - confidence if category != "SAFE" else confidence,
                },
            }
        },
        "usage": {"input_tokens": 32, "output_tokens": 0},
    }


def make_mock_t2_response(verdict: str = "safe", violation_code: str = "S1") -> dict[str, Any]:
    """Generates synthetic Llama Guard /v1/chat/completions response payload."""
    content = "safe" if verdict == "safe" else f"unsafe\n{violation_code}"
    return {
        "id": "chatcmpl-mock-t2",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 150, "completion_tokens": 4, "total_tokens": 154},
    }
